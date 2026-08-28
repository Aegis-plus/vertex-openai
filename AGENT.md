# AGENT.md - System Architecture & Developer Guide

This document provides a technical deep-dive into the design, wire-format translations, protocol bridges, and state smuggling mechanisms implemented in `vertex-openai`. It is intended for AI coding agents and developers maintaining, extending, or debugging this codebase.

---

## 🧭 Overview & Mission

`vertex-openai` is a stateless Cloudflare Worker reverse proxy that sits between standard OpenAI API clients (OpenAI SDKs, LangChain, LlamaIndex, Cursor, LibreChat, Open WebUI, etc.) and **Google Cloud Vertex AI Express Mode REST API** (`https://aiplatform.googleapis.com/v1/publishers/google/models`).

Because Google Vertex AI Express Mode uses a distinct REST wire format differing from both Google AI Studio (Gemini API) and OpenAI, this worker handles all bidirectionally required transformations in high-performance V8 edge runtime.

---

## 📐 Architecture & Pipeline

```
[OpenAI Client Request]
       │
       ▼
┌──────────────────────────────────────────────┐
│  Cloudflare Worker (`fetch` entrypoint)      │
│  - CORS & Preflight (OPTIONS -> 204)         │
│  - Auth Check (`Authorization: Bearer ...`)  │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[Chat Completions]             [Images / Models]
  ├─ `toContents(messages)`      ├─ `GET /v1/models`
  ├─ `toConfig(body)`            ├─ `POST /v1/images/generations`
  └─ `toFunctionDeclarations()`  └─ `POST /v1/images/edits`
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Upstream Request (`fetchWithRetry`)         │
│  - Endpoint: Vertex AI Express Mode REST     │
│  - Headers: `x-goog-api-key`                 │
│  - Retry: 429 quota backoff (5s, 10s, 15s)   │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[Non-Streaming JSON]            [SSE Stream (?alt=sse)]
  └─ `toOpenAIResponse()`        └─ `TransformStream`
                                    ├─ Parse Gemini SSE chunks
                                    ├─ Extract reasoning / text / tools
                                    ├─ Emit OpenAI chunk format
                                    └─ Emit final `finish_reason` & [DONE]
```

---

## 🔬 Key Protocol Bridge Mechanics

### 1. CamelCase Wire Format Requirement
Unlike the Google Python SDK which allows snake_case and translates under the hood, raw HTTP requests sent to Vertex AI Express Mode REST API **must strictly use camelCase field names**:
- `systemInstruction` (not `system_instruction`)
- `functionDeclarations` (not `function_declarations`)
- `functionCall` / `functionResponse` (not `function_call` / `function_response`)
- `maxOutputTokens`, `topP`, `stopSequences`, `presencePenalty`, `frequencyPenalty`, `candidateCount`
- `thinkingConfig`, `includeThoughts`, `thinkingBudget`
- `responseModalities`

### 2. Gemini 3 Thought Signature Smuggling (`__sig.`)
**Problem**:
Gemini 3 reasoning/thinking models produce a `thoughtSignature` on every `functionCall` part. When the client executes the tool and replies with the result on the next turn, Vertex AI requires the previous assistant turn to echo back the **exact same `thoughtSignature`** alongside `functionCall`. However, OpenAI's tool call schema has no field for proprietary metadata like thought signatures.

**Solution**:
The worker embeds the signature into the OpenAI `tool_call.id` string:
```javascript
// candidateToMessage (outbound to client)
const sig = p.thoughtSignature ? `__sig.${p.thoughtSignature}` : "";
const callId = `call_${i}_${crypto.randomUUID()}${sig}`;

// toContents (inbound from client on next turn)
const sigMatch = tc.id?.match(/__sig\.(.+)$/);
return {
  functionCall: { name: tc.function.name, args: safeParse(tc.function.arguments) },
  ...(sigMatch ? { thoughtSignature: sigMatch[1] } : {}),
};
```
Because OpenAI clients treat `tool_call_id` as an opaque identifier and return it verbatim in the `tool` message, the signature is seamlessly round-tripped without client modifications.

### 3. Tool Result Batching
**Problem**:
Gemini requires that a model turn containing $N$ function calls be answered by **a single subsequent turn** containing all $N$ `functionResponse` parts. OpenAI clients send one individual message per tool result (`role: "tool"`).

**Solution**:
`toContents()` buffers consecutive `role: "tool"` messages and flushes them into a single `{ role: "user", parts: [...] }` content turn:
```javascript
let pendingToolParts = null;
function flushToolParts() {
  if (pendingToolParts?.length) {
    contents.push({ role: "user", parts: pendingToolParts });
  }
  pendingToolParts = null;
}
```

### 4. Reasoning / Thinking Extraction
Gemini represents reasoning thoughts as parts with `thought: true`.
- Text parts where `thought !== true` are mapped to `content` (or `delta.content`).
- Parts with `thought === true` are separated and emitted as `reasoning` (or `delta.reasoning`), preventing internal chain-of-thought from contaminating output text.
- Configuration mapping:
  - `reasoning_effort: "none"` or `thinking.enabled: false` -> `{ includeThoughts: false }`
  - `reasoning_effort: "low"` -> budget 1,024 tokens
  - `reasoning_effort: "medium"` -> budget 8,192 tokens
  - `reasoning_effort: "high"` -> budget 24,576 tokens
  - `thinking.budget_tokens` -> custom `thinkingBudget`

### 5. Multimodal Vision Handling
`toParts()` handles both single-string messages and OpenAI-format content part arrays:
- Text parts (`{ type: "text", text: "..." }`) -> `{ text: "..." }`
- Image parts (`{ type: "image_url", image_url: { url: "..." } }`):
  - Data URIs (`data:image/png;base64,...`) -> decoded into `{ inlineData: { mimeType, data } }`
  - Remote HTTP/HTTPS URLs -> downloaded via `fetch()`, converted to Base64 in chunks, and wrapped as `{ inlineData: { mimeType, data } }`.

### 6. Streaming Lifecycle & Edge Cases
The SSE streaming handler (`POST /v1/chat/completions` with `stream: true`):
1. Calls upstream `streamGenerateContent?alt=sse`.
2. Chunks are buffered and split on newline boundaries to ensure full SSE data frames.
3. Every data chunk is parsed and transformed to `chat.completion.chunk`.
4. If a stream ends with zero usable output (e.g. prompt blocked by safety filters or empty finishReason), a diagnostic chunk is emitted explaining the block reason or candidate state rather than failing silently.
5. Emits a final chunk containing the proper `finish_reason` (`"tool_calls"` or `"stop"`) followed by `data: [DONE]\n\n`.

### 7. Automatic 429 Backoff Retry
Vertex AI Express Mode has dynamic rate limits. `fetchWithRetry()` intercepts HTTP `429` responses and retries up to 3 times with delays `[5000ms, 10000ms, 15000ms]` before surfacing the status to the client.

---

## 📂 Codebase File Map

| File | Purpose |
| :--- | :--- |
| [`worker.js`](file:///E:/Project/vertex-openai/worker.js) | Main entrypoint containing all request handlers, transformation helpers, streaming transforms, and CORS utilities. |
| [`wrangler.toml`](file:///E:/Project/vertex-openai/wrangler.toml) | Cloudflare Worker deployment manifest (`name`, `main`, `compatibility_date`). |
| [`.gitignore`](file:///E:/Project/vertex-openai/.gitignore) | Git ignore patterns for `.wrangler/`, local variables, logs, and artifacts. |
| [`README.md`](file:///E:/Project/vertex-openai/README.md) | User-facing documentation, setup guide, and usage examples. |
| [`AGENT.md`](file:///E:/Project/vertex-openai/AGENT.md) | This technical reference document for agents and maintainers. |

---

## 🛠️ Maintenance & Extension Guide

### Adding New Gemini Models
1. Add the model identifier to `AVAILABLE_MODELS` array in `worker.js`:
   ```javascript
   const AVAILABLE_MODELS = [
     "gemini-3.7-flash",
     // ... add new model ID here
   ];
   ```
2. If the model supports native image outputs (Nano Banana series), also register it in `IMAGE_CAPABLE_MODELS`:
   ```javascript
   const IMAGE_CAPABLE_MODELS = new Set([
     "gemini-2.5-flash-image",
     "gemini-3.1-flash-image",
     // ... add here
   ]);
   ```

### Tuning Rate-Limit Backoffs
Modify `RETRY_BACKOFF_MS` in `worker.js`:
```javascript
const RETRY_BACKOFF_MS = [5000, 10000, 15000]; // Delays in milliseconds
```

### Tuning Reasoning Effort Presets
Modify the mapping dictionary in `toThinkingConfig(body)`:
```javascript
const map = { low: 1024, medium: 8192, high: 24576 };
```

---

## 🧪 Verification & Testing Checklist

When making changes to `worker.js`, verify the following matrix:

- [ ] **Standard Chat**: Non-streaming `POST /v1/chat/completions` with user + system messages.
- [ ] **Streaming Chat**: `stream: true` yields SSE deltas and finishes with `data: [DONE]`.
- [ ] **Reasoning Output**: Verify `delta.reasoning` / `message.reasoning` on `gemini-3.7-flash`.
- [ ] **Single & Multi-Turn Tools**: Tool calls emit `id` with `__sig.`, subsequent turn receives and passes signature back to Vertex.
- [ ] **Vision / Multimodal**: Test with data URI and remote image URL.
- [ ] **Image Generation**: Test `POST /v1/images/generations` with `gemini-2.5-flash-image`.
- [ ] **Image Editing**: Test `POST /v1/images/edits` with `multipart/form-data`.
- [ ] **CORS**: `OPTIONS` preflight returns status `204` with appropriate `Access-Control-Allow-*` headers.
- [ ] **Auth Enforcement**: Requests without `Authorization: Bearer <PROXY_API_KEY>` receive `401 Unauthorized`.
