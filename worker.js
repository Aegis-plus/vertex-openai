// Cloudflare Worker: OpenAI-compatible /v1/chat/completions in front of
// Vertex AI Express Mode.
// Vertex's REST API uses camelCase field names (functionDeclarations,
// functionCall, functionResponse, systemInstruction, maxOutputTokens, etc.)
// even though the Python SDK shown in the quickstart doc accepts snake_case
// and converts it under the hood. This worker builds raw JSON, so it must
// use the camelCase wire format directly.
//
// Env (wrangler secret put):
//   VERTEX_EXPRESS_API_KEY  - your Express Mode API key
//   PROXY_API_KEY           - key clients send to THIS worker

const BASE = "https://aiplatform.googleapis.com/v1/publishers/google/models";

// When Vertex returns 429 (Express Mode quota exhausted) the worker waits
// and retries with this escalating backoff: 5s, then 10s, then 15s, after
// which it surfaces the last 429 to the caller instead of retrying forever.
const RETRY_BACKOFF_MS = [5000, 10000, 15000];

// Wraps fetch to transparently retry on HTTP 429 only -- any other status
// (including other 4xx/5xx) passes straight through. Used so transient
// rate-limit errors don't fail requests that would otherwise succeed.
async function fetchWithRetry(url, init) {
  let lastRes;
  for (let attempt = 0; attempt <= RETRY_BACKOFF_MS.length; attempt++) {
    const res = await fetch(url, init);
    lastRes = res;
    if (res.status !== 429) return res;
    const delay = RETRY_BACKOFF_MS[attempt];
    if (delay === undefined) return res;
    await new Promise(r => setTimeout(r, delay));
  }
  return lastRes;
}

// Express Mode has no model-listing endpoint in the quickstart doc, so we
// hardcode the set of models you've confirmed are available to your key.
// Add/remove entries here as your access changes.
const AVAILABLE_MODELS = [
  "gemini-3.7-flash",
  "gemini-3.6-flash",
  "gemini-3.5-flash",
  "gemini-3-flash-preview",
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
  "gemini-2.5-flash-lite",
  "gemini-2.5-flash",
  "gemini-2.5-flash-image",
  "gemini-3.1-flash-image",
  "gemini-3.1-flash-lite-image",
];

// Models that can emit image parts (the "Nano Banana" family). Requesting
// one of these auto-adds IMAGE to responseModalities so callers don't have
// to know about that Gemini-specific config field.
const IMAGE_CAPABLE_MODELS = new Set([
  "gemini-2.5-flash-image",
  "gemini-3.1-flash-image",
  "gemini-3.1-flash-lite-image",
]);

// Converts an OpenAI image_url content block into Gemini's inlineData part.
// Supports data: URIs directly, and fetches remote http(s) URLs since
// Gemini's inlineData needs the actual base64 bytes, not a reference.
async function imageUrlToPart(url) {
  const dataUriMatch = url.match(/^data:([^;]+);base64,(.+)$/);
  if (dataUriMatch) {
    return { inlineData: { mimeType: dataUriMatch[1], data: dataUriMatch[2] } };
  }
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const mimeType = res.headers.get("content-type") || "image/png";
    const buf = await res.arrayBuffer();
    return { inlineData: { mimeType, data: arrayBufferToBase64(buf) } };
  } catch {
    return null;
  }
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

// Maps OpenAI's Images API "size" strings (e.g. "1024x1024") to the closest
// Gemini aspectRatio enum value. Falls back to null (Gemini's default,
// roughly 1:1) if size is missing or unrecognized.
const GEMINI_ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"];
function sizeToAspectRatio(size) {
  if (!size || size === "auto") return null;
  const match = size.match(/^(\d+)x(\d+)$/);
  if (!match) return null;
  const w = parseInt(match[1], 10);
  const h = parseInt(match[2], 10);
  const target = w / h;
  let closest = null;
  let closestDiff = Infinity;
  for (const ratio of GEMINI_ASPECT_RATIOS) {
    const [rw, rh] = ratio.split(":").map(Number);
    const diff = Math.abs(rw / rh - target);
    if (diff < closestDiff) {
      closestDiff = diff;
      closest = ratio;
    }
  }
  return closest;
}

// ---- OpenAI -> quickstart-doc shape -----------------------------------

// Normalizes OpenAI message content, which can be either a plain string
// or an array of content blocks (e.g. [{type: "text", text: "..."}]),
// into a single string. Sending an array where Gemini expects a string
// causes: "Proto field is not repeating, cannot start list."
function extractText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter(part => typeof part?.text === "string")
      .map(part => part.text)
      .join("");
  }
  return "";
}

// Like extractText, but also converts OpenAI image_url content blocks into
// Gemini inlineData parts, so a message can carry both text and images.
async function toParts(content) {
  if (typeof content === "string") {
    return content ? [{ text: content }] : [];
  }
  if (Array.isArray(content)) {
    const parts = [];
    for (const block of content) {
      if (typeof block?.text === "string") {
        parts.push({ text: block.text });
      } else if (block?.type === "image_url" && block.image_url?.url) {
        const part = await imageUrlToPart(block.image_url.url);
        if (part) parts.push(part);
      }
    }
    return parts;
  }
  return [];
}

// contents: [{role, parts:[{text}]}] -- mirrors the doc's "contents" param
async function toContents(messages) {
  let systemInstruction;
  const contents = [];
  const toolCallIdToName = {}; // OpenAI tool-result messages carry tool_call_id, not name

  // Gemini requires that a model turn with N function calls be followed by
  // ONE turn containing all N functionResponse parts -- not N separate
  // turns. OpenAI clients send one "tool" message per call, so we batch
  // consecutive tool messages together into a single content entry.
  let pendingToolParts = null;
  function flushToolParts() {
    if (pendingToolParts?.length) {
      contents.push({ role: "user", parts: pendingToolParts });
    }
    pendingToolParts = null;
  }

  for (const m of messages) {
    if (m.role === "system") {
      flushToolParts();
      // Multiple system messages must be combined, not overwritten -- some
      // clients send several (e.g. a base prompt plus per-request context).
      const text = extractText(m.content);
      if (systemInstruction) {
        systemInstruction.parts.push({ text });
      } else {
        systemInstruction = { parts: [{ text }] };
      }
      continue;
    }
    if (m.role === "tool") {
      const name = m.name ?? toolCallIdToName[m.tool_call_id] ?? "unknown_function";
      if (!pendingToolParts) pendingToolParts = [];
      pendingToolParts.push({
        functionResponse: {
          name,
          response: { name, content: safeParse(m.content) },
        },
      });
      continue;
    }
    flushToolParts(); // any non-tool message ends a run of tool results
    if (m.role === "assistant" && m.tool_calls?.length) {
      // Vertex REST uses camelCase: functionCall. Recover any thoughtSignature
      // we smuggled into the id (see candidateToMessage) and re-attach it --
      // Vertex requires the exact same signature echoed back on this part.
      contents.push({
        role: "model",
        parts: m.tool_calls.map(tc => {
          toolCallIdToName[tc.id] = tc.function.name;
          const sigMatch = tc.id?.match(/__sig\.(.+)$/);
          return {
            functionCall: {
              name: tc.function.name,
              args: safeParse(tc.function.arguments),
            },
            ...(sigMatch ? { thoughtSignature: sigMatch[1] } : {}),
          };
        }),
      });
      continue;
    }
    contents.push({
      role: m.role === "assistant" ? "model" : "user",
      parts: await toParts(m.content),
    });
  }
  flushToolParts(); // in case the message list ends on tool results
  return { contents, systemInstruction };
}

function safeParse(v) {
  if (typeof v !== "string") return v;
  try { return JSON.parse(v); } catch { return v; }
}

// config: Vertex REST generationConfig uses camelCase field names
function toConfig(body) {
  const cfg = {};
  if (body.temperature !== undefined) cfg.temperature = body.temperature;
  if (body.top_p !== undefined) cfg.topP = body.top_p;
  if (body.max_tokens !== undefined) cfg.maxOutputTokens = body.max_tokens;
  if (body.stop !== undefined) {
    cfg.stopSequences = Array.isArray(body.stop) ? body.stop : [body.stop];
  }
  if (body.presence_penalty !== undefined) cfg.presencePenalty = body.presence_penalty;
  if (body.frequency_penalty !== undefined) cfg.frequencyPenalty = body.frequency_penalty;
  if (body.seed !== undefined) cfg.seed = body.seed;
  if (body.n !== undefined) cfg.candidateCount = body.n;

  // Thinking / reasoning. Gemini only emits thought summaries when asked via
  // thinkingConfig.includeThoughts. We request them by default so callers see
  // reasoning, but let the client opt out or tune the budget:
  //   - reasoning_effort: "none" (OpenAI-style) disables thinking
  //   - thinking: { include?: bool, budget_tokens?: number } for fine control
  const thinkingConfig = toThinkingConfig(body);
  if (thinkingConfig) cfg.thinkingConfig = thinkingConfig;

  return cfg;
}

function toThinkingConfig(body) {
  const t = body.thinking;
  const effort = body.reasoning_effort;

  // Explicit opt-out.
  if (effort === "none" || t?.include === false || t?.enabled === false) {
    return { includeThoughts: false };
  }

  const cfg = { includeThoughts: true };

  // Explicit token budget wins over effort-based mapping.
  const budget = t?.budget_tokens ?? t?.thinkingBudget;
  if (typeof budget === "number") {
    cfg.thinkingBudget = budget;
  } else if (typeof effort === "string") {
    const map = { low: 1024, medium: 8192, high: 24576 };
    if (map[effort] !== undefined) cfg.thinkingBudget = map[effort];
  }

  return cfg;
}

// tools: OpenAI tools -> Vertex REST's camelCase functionDeclarations
function toFunctionDeclarations(tools) {
  if (!tools?.length) return undefined;
  return tools
    .filter(t => t.type === "function")
    .map(t => ({
      name: t.function.name,
      description: t.function.description,
      parameters: t.function.parameters,
    }));
}

// ---- quickstart-doc response -> OpenAI shape ---------------------------

function candidateToMessage(candidate) {
  const parts = candidate?.content?.parts ?? [];
  // Gemini "thinking" models mark reasoning parts with `thought: true`.
  // Keep those out of `content` so answer text isn't polluted with the
  // model's internal reasoning; expose them separately as `reasoning`.
  const textParts = parts.filter(p => p.text !== undefined && p.thought !== true).map(p => p.text);
  const thoughtParts = parts.filter(p => p.text !== undefined && p.thought === true).map(p => p.text);
  const fnParts = parts.filter(p => p.functionCall);
  const imageParts = parts.filter(p => p.inlineData);

  const message = { role: "assistant", content: textParts.join("") || null };

  if (thoughtParts.length) message.reasoning = thoughtParts.join("");

  // Generated images go in a separate `images` array on the message, each
  // shaped like an OpenAI vision image_url block -- this matches the
  // de-facto convention used by OpenRouter and consumed by clients like
  // Open WebUI / LibreChat, rather than inlining as markdown in content.
  if (imageParts.length) {
    message.images = imageParts.map(p => ({
      type: "image_url",
      image_url: { url: `data:${p.inlineData.mimeType};base64,${p.inlineData.data}` },
    }));
  }

  if (fnParts.length) {
    message.tool_calls = fnParts.map((p, i) => {
      // Gemini 3 "thinking" models attach a thoughtSignature to each
      // functionCall part. Vertex requires that exact signature to be
      // echoed back on the next turn, but OpenAI's tool_calls schema has
      // nowhere to carry it. We smuggle it inside `id`, since clients
      // treat tool_call ids as opaque and pass them back unmodified.
      const sig = p.thoughtSignature ? `__sig.${p.thoughtSignature}` : "";
      return {
        id: `call_${i}_${crypto.randomUUID()}${sig}`,
        type: "function",
        function: {
          name: p.functionCall.name,
          arguments: JSON.stringify(p.functionCall.args ?? {}),
        },
      };
    });
  }
  return message;
}

// Converts one Vertex candidate into an OpenAI choice at the given index.
function candidateToChoice(candidate, index) {
  const message = candidateToMessage(candidate);
  const hasUsableOutput = !!message.content || !!message.tool_calls?.length || !!message.images?.length;

  // A candidate can technically have non-empty `parts` while still having
  // nothing usable in them -- e.g. a lone internal marker part with no
  // text and no functionCall. Checking parts.length alone missed this;
  // we now check whether candidateToMessage actually extracted anything.
  if (!hasUsableOutput) {
    const rawPartShapes = (candidate.content?.parts ?? []).map(p => Object.keys(p).join("+")) || [];
    const safety = candidate.safetyRatings?.filter(r => r.blocked).map(r => r.category) ?? [];
    const detail = [
      `finishReason=${candidate.finishReason ?? "unknown"}`,
      rawPartShapes.length ? `raw part shapes=[${rawPartShapes.join(", ")}]` : "raw parts=[] (none at all)",
      safety.length ? `blocked safety categories=[${safety.join(", ")}]` : null,
    ].filter(Boolean).join(", ");
    return {
      index,
      message: { role: "assistant", content: `[No response: Vertex returned a candidate with no usable text or tool call (${detail})]` },
      finish_reason: candidate.finishReason && candidate.finishReason !== "STOP"
        ? candidate.finishReason.toLowerCase()
        : "stop",
    };
  }

  return {
    index,
    message,
    finish_reason: message.tool_calls?.length
      ? "tool_calls"
      : (candidate.finishReason ?? "stop").toLowerCase(),
  };
}

function toOpenAIResponse(json, model) {
  const candidates = json.candidates ?? [];

  // If Vertex fully blocks a request (safety, recitation, etc.) it can
  // return an EMPTY candidates array rather than a candidate with a
  // reason attached. The real reason lives in promptFeedback.blockReason.
  if (!candidates.length) {
    const blockReason = json.promptFeedback?.blockReason;
    return emptyResponse(model, json,
      blockReason
        ? `Vertex blocked this request, reason: ${blockReason}`
        : "Vertex returned no candidates and no block reason",
      "content_filter");
  }

  // Map every candidate to a choice so n>1 (candidateCount) is honored,
  // instead of silently dropping all but the first.
  return {
    id: "chatcmpl-" + crypto.randomUUID(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: candidates.map((c, i) => candidateToChoice(c, i)),
    usage: {
      prompt_tokens: json.usageMetadata?.promptTokenCount ?? 0,
      completion_tokens: json.usageMetadata?.candidatesTokenCount ?? 0,
      total_tokens: json.usageMetadata?.totalTokenCount ?? 0,
    },
  };
}

function emptyResponse(model, json, detailMessage, finishReason) {
  return {
    id: "chatcmpl-" + crypto.randomUUID(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: `[No response: ${detailMessage}]` },
      finish_reason: finishReason,
    }],
    usage: {
      prompt_tokens: json.usageMetadata?.promptTokenCount ?? 0,
      completion_tokens: json.usageMetadata?.candidatesTokenCount ?? 0,
      total_tokens: json.usageMetadata?.totalTokenCount ?? 0,
    },
  };
}

// Shared image-generation call, used by both /v1/images/generations (with
// an optional input image for editing) and /v1/images/edits (multipart).
// inputParts is an array of extra Gemini parts (inlineData) to prepend
// before the text prompt, i.e. the source image(s) for image-to-image.
async function generateImages({ env, model, prompt, aspectRatio, inputParts = [], n = 1 }) {
  const requestBody = {
    contents: [{ role: "user", parts: [...inputParts, { text: prompt ?? "" }] }],
    generationConfig: {
      responseModalities: ["TEXT", "IMAGE"],
      ...(aspectRatio ? { imageConfig: { aspectRatio } } : {}),
    },
  };

  const callOnce = async () => {
    const res = await fetchWithRetry(`${BASE}/${model}:generateContent`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": env.VERTEX_EXPRESS_API_KEY,
      },
      body: JSON.stringify(requestBody),
    });
    if (!res.ok) return { ok: false, status: res.status, text: await res.text() };
    const json = await res.json();
    const parts = json.candidates?.[0]?.content?.parts ?? [];
    return { ok: true, images: parts.filter(p => p.inlineData).map(p => ({ b64_json: p.inlineData.data })) };
  };

  const first = await callOnce();
  if (!first.ok) return first;
  const images = first.images;

  // Gemini only returns one image per call regardless of n; repeat the
  // request if more than one was asked for, since there's no native
  // batch parameter for image count on this endpoint.
  while (images.length < n && images.length > 0) {
    const extra = await callOnce();
    if (!extra.ok || !extra.images.length) break;
    images.push(...extra.images);
  }

  return { ok: true, images: images.slice(0, n) };
}

// ---- CORS ---------------------------------------------------------------

// Allowed origin is configurable via the CORS_ALLOW_ORIGIN secret/var.
// Defaults to "*" so browser clients work out of the box; set it to a
// specific origin (e.g. https://app.example.com) to lock the proxy down.
function corsHeaders(env, request) {
  const allow = env.CORS_ALLOW_ORIGIN || "*";
  // When echoing a specific origin, only reflect it if it matches; also
  // send Vary so caches don't mix responses across origins.
  const origin = request.headers.get("Origin");
  const allowOrigin = allow === "*"
    ? "*"
    : (origin && origin === allow ? origin : allow);
  return {
    "Access-Control-Allow-Origin": allowOrigin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
  };
}

// ---- Worker -------------------------------------------------------------

export default {
  async fetch(request, env) {
    const cors = corsHeaders(env, request);

    // Preflight requests never carry Authorization, so answer them before
    // the auth check with just the CORS headers.
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }

    const response = await handle(request, env);

    // Attach CORS headers to whatever the handler produced. Rebuild the
    // Response so streaming bodies pass through untouched.
    const headers = new Headers(response.headers);
    for (const [k, v] of Object.entries(cors)) headers.set(k, v);
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};

async function handle(request, env) {
    const url = new URL(request.url);

    if (request.headers.get("Authorization") !== `Bearer ${env.PROXY_API_KEY}`) {
      return new Response("Unauthorized", { status: 401 });
    }

    // GET /v1/models - lets OpenAI-compatible clients populate a model dropdown
    if (request.method === "GET" && url.pathname === "/v1/models") {
      const now = Math.floor(Date.now() / 1000);
      return new Response(JSON.stringify({
        object: "list",
        data: AVAILABLE_MODELS.map(id => ({
          id,
          object: "model",
          created: now,
          owned_by: "google",
        })),
      }), { headers: { "Content-Type": "application/json" } });
    }

    // POST /v1/images/generations - OpenAI Images API shape: {data: [{b64_json}]}
    // Accepts an optional `image` (data URI or bare base64) for image-to-image.
    if (request.method === "POST" && url.pathname.endsWith("/v1/images/generations")) {
      const body = await request.json();
      const model = body.model && IMAGE_CAPABLE_MODELS.has(body.model)
        ? body.model
        : "gemini-2.5-flash-image"; // sane default if an unsupported/text model is passed

      // Accept a direct aspect_ratio (e.g. "16:9") or OpenAI-style size
      // (e.g. "1792x1024"), mapped to the nearest supported Gemini ratio.
      const aspectRatio = body.aspect_ratio ?? sizeToAspectRatio(body.size);

      const inputParts = [];
      const rawImages = body.image ? [body.image] : (body.images ?? []);
      for (const img of rawImages) {
        const part = typeof img === "string" && img.startsWith("data:")
          ? await imageUrlToPart(img)
          : { inlineData: { mimeType: "image/png", data: img } }; // assume bare base64
        if (part) inputParts.push(part);
      }

      const result = await generateImages({
        env, model, prompt: body.prompt, aspectRatio, inputParts, n: body.n ?? 1,
      });

      if (!result.ok) return new Response(result.text, { status: result.status });
      if (!result.images.length) {
        return new Response(JSON.stringify({ error: { message: "No image returned by Vertex", code: 502 } }), { status: 502 });
      }
      return new Response(JSON.stringify({
        created: Math.floor(Date.now() / 1000),
        data: result.images,
      }), { headers: { "Content-Type": "application/json" } });
    }

    // POST /v1/images/edits - OpenAI's actual image-editing endpoint, which
    // real OpenAI clients send as multipart/form-data with an `image` file
    // and a `prompt` field, rather than JSON.
    if (request.method === "POST" && url.pathname.endsWith("/v1/images/edits")) {
      const form = await request.formData();
      const prompt = form.get("prompt") ?? "";
      const model = form.get("model") && IMAGE_CAPABLE_MODELS.has(form.get("model"))
        ? form.get("model")
        : "gemini-2.5-flash-image";
      const aspectRatio = form.get("aspect_ratio") ?? sizeToAspectRatio(form.get("size"));
      const n = Number(form.get("n") ?? 1);

      const inputParts = [];
      // OpenAI's edits endpoint allows one `image` file or multiple under
      // `image[]`; support both since clients vary in which they send.
      const files = [...form.getAll("image"), ...form.getAll("image[]")].filter(f => f instanceof File);
      for (const file of files) {
        const buf = await file.arrayBuffer();
        inputParts.push({
          inlineData: { mimeType: file.type || "image/png", data: arrayBufferToBase64(buf) },
        });
      }

      const result = await generateImages({ env, model, prompt, aspectRatio, inputParts, n });

      if (!result.ok) return new Response(result.text, { status: result.status });
      if (!result.images.length) {
        return new Response(JSON.stringify({ error: { message: "No image returned by Vertex", code: 502 } }), { status: 502 });
      }
      return new Response(JSON.stringify({
        created: Math.floor(Date.now() / 1000),
        data: result.images,
      }), { headers: { "Content-Type": "application/json" } });
    }

    if (request.method !== "POST" || !url.pathname.endsWith("/v1/chat/completions")) {
      return new Response("Not found", { status: 404 });
    }

    const body = await request.json();
    const { model, messages, stream, tools } = body;
    const { contents, systemInstruction } = await toContents(messages);
    const functionDeclarations = toFunctionDeclarations(tools);
    const generationConfig = toConfig(body);
    if (IMAGE_CAPABLE_MODELS.has(model)) {
      generationConfig.responseModalities = ["TEXT", "IMAGE"];
    }

    const requestBody = {
      contents,
      ...(systemInstruction ? { systemInstruction } : {}),
      ...(Object.keys(generationConfig).length ? { generationConfig } : {}),
      ...(functionDeclarations ? { tools: [{ functionDeclarations }] } : {}),
    };

    const method = stream ? "streamGenerateContent" : "generateContent";
    const upstreamUrl = `${BASE}/${model}:${method}${stream ? "?alt=sse" : ""}`;

    const upstream = await fetchWithRetry(upstreamUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": env.VERTEX_EXPRESS_API_KEY,
      },
      body: JSON.stringify(requestBody),
    });

    if (!upstream.ok) {
      const errText = await upstream.text();
      return new Response(errText, { status: upstream.status });
    }

    if (!stream) {
      const json = await upstream.json();
      return new Response(JSON.stringify(toOpenAIResponse(json, model)), {
        headers: { "Content-Type": "application/json" },
      });
    }

    // streamGenerateContent -> re-wrap each chunk as an OpenAI SSE delta
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();
    const reader = upstream.body.getReader();
    const decoder = new TextDecoder();
    const id = "chatcmpl-" + crypto.randomUUID();

    (async () => {
      let buf = "";
      let sawAnyOutput = false;
      let sawToolCalls = false;
      let lastChunk = null;
      let lastFinishReason = null;

      // Process one already-parsed Vertex chunk into an OpenAI SSE delta.
      const handleChunk = async (chunk) => {
        lastChunk = chunk;
        const candidate = chunk.candidates?.[0];
        if (candidate?.finishReason) lastFinishReason = candidate.finishReason;
        const message = candidateToMessage(candidate);
        const hasOutput = !!message.content || !!message.tool_calls?.length || !!message.images?.length || !!message.reasoning;
        if (!hasOutput) return; // don't emit empty deltas, but keep reading
        sawAnyOutput = true;
        const delta = {};
        if (message.reasoning) delta.reasoning = message.reasoning;
        if (message.content) delta.content = message.content;
        if (message.tool_calls?.length) {
          sawToolCalls = true;
          // OpenAI streaming requires an `index` on each tool_call delta so
          // clients can reassemble them across chunks.
          delta.tool_calls = message.tool_calls.map((tc, i) => ({ index: i, ...tc }));
        }
        if (message.images?.length) delta.images = message.images;
        const openaiChunk = {
          id, object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model, choices: [{ index: 0, delta, finish_reason: null }],
        };
        await writer.write(encoder.encode(`data: ${JSON.stringify(openaiChunk)}\n\n`));
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;
          await handleChunk(JSON.parse(jsonStr));
        }
      }

      // Flush any trailing buffered SSE line that lacked a final newline.
      const tail = buf.trim();
      if (tail.startsWith("data:")) {
        const jsonStr = tail.slice(5).trim();
        if (jsonStr && jsonStr !== "[DONE]") {
          try { await handleChunk(JSON.parse(jsonStr)); } catch {}
        }
      }

      // A full stream can complete with finishReason STOP and zero usable
      // output. Surface that instead of silently sending only [DONE].
      if (!sawAnyOutput) {
        const candidate = lastChunk?.candidates?.[0];
        const blockReason = lastChunk?.promptFeedback?.blockReason;
        const rawPartShapes = (candidate?.content?.parts ?? []).map(p => Object.keys(p).join("+"));
        const safety = candidate?.safetyRatings?.filter(r => r.blocked).map(r => r.category) ?? [];
        const detail = blockReason
          ? `Vertex blocked this request, reason: ${blockReason}`
          : [
              `Vertex returned finishReason=${candidate?.finishReason ?? "unknown"} with no usable text or tool call`,
              rawPartShapes.length ? `raw part shapes=[${rawPartShapes.join(", ")}]` : "raw parts=[] (none at all)",
              safety.length ? `blocked safety categories=[${safety.join(", ")}]` : null,
            ].filter(Boolean).join(", ");
        const diagChunk = {
          id, object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model, choices: [{ index: 0, delta: { content: `[No response: ${detail}]` }, finish_reason: null }],
        };
        await writer.write(encoder.encode(`data: ${JSON.stringify(diagChunk)}\n\n`));
      }

      // OpenAI's SSE contract requires a final chunk with a non-null
      // finish_reason (before [DONE]) so clients know the turn ended and,
      // for tool calls, that they should now execute them.
      const finishReason = sawToolCalls
        ? "tool_calls"
        : (lastFinishReason ?? "stop").toLowerCase();
      const finalChunk = {
        id, object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model, choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
      };
      await writer.write(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`));

      await writer.write(encoder.encode("data: [DONE]\n\n"));
      await writer.close();
    })();

    return new Response(readable, {
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
    });
}
