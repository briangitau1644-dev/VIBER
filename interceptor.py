#!/usr/bin/env python3
"""
Ollama OpenAI-Compatible Proxy v4.0
- Full OpenAI API compatibility (chat, completions, responses, embeddings, models)
- Streaming support (SSE)
- Robust model mapping (any model name → actual Ollama model)
- CORS support
- Health checks & debug endpoints
- Graceful fallback: /api/chat → /api/generate
"""

import os
import sys
import json
import time
import logging
import requests
import uuid
import re
from flask import Flask, request, jsonify, Response, stream_with_context

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG_MODE", "false").lower() == "true" else logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("proxy")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
PROXY_PORT   = int(os.getenv("PROXY_PORT", "1234"))
OLLAMA_PORT  = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "127.0.0.1")
REAL_MODEL   = os.getenv("MODEL", "gemma4:e4b")
CTX_SIZE     = int(os.getenv("CTX_SIZE", "8192"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.3"))
OLLAMA_URL   = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_TIMEOUT = 600  # seconds

# Model mapping: any incoming model name → REAL_MODEL
MODEL_ALIASES = {
    # OpenAI GPT family
    "gpt-4o":                    REAL_MODEL,
    "gpt-4o-mini":               REAL_MODEL,
    "gpt-4-turbo":               REAL_MODEL,
    "gpt-4-turbo-preview":       REAL_MODEL,
    "gpt-4":                     REAL_MODEL,
    "gpt-3.5-turbo":             REAL_MODEL,
    "gpt-3.5-turbo-16k":         REAL_MODEL,
    "gpt-5.1-codex-max":         REAL_MODEL,
    "o1":                        REAL_MODEL,
    "o1-mini":                   REAL_MODEL,
    "o3":                        REAL_MODEL,
    "o3-mini":                   REAL_MODEL,
    # Anthropic Claude family
    "claude-3-opus":             REAL_MODEL,
    "claude-3-sonnet":           REAL_MODEL,
    "claude-3-haiku":            REAL_MODEL,
    "claude-3-5-sonnet":         REAL_MODEL,
    "claude-3-5-haiku":          REAL_MODEL,
    # Meta Llama family
    "llama2":                    REAL_MODEL,
    "llama3":                    REAL_MODEL,
    "llama3.1":                  REAL_MODEL,
    "llama3.2":                  REAL_MODEL,
    # Mistral / Mixtral
    "mistral":                   REAL_MODEL,
    "mixtral":                   REAL_MODEL,
    "mixtral-8x7b":              REAL_MODEL,
    # Google Gemma
    "gemma":                     REAL_MODEL,
    "gemma2":                    REAL_MODEL,
    # Explicit real model
    REAL_MODEL:                  REAL_MODEL,
}

def resolve_model(name: str) -> str:
    """Return the actual Ollama model for any requested name."""
    if not name:
        return REAL_MODEL
    return MODEL_ALIASES.get(name, REAL_MODEL)  # default to real model for unknowns


# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── CORS ──────────────────────────────────────────────────────
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD"
    resp.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization, Accept, X-Requested-With, "
        "OpenAI-Beta, X-API-Key, api-key"
    )
    resp.headers["Access-Control-Expose-Headers"] = "Content-Type, X-Request-Id"
    return resp


@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return "", 204


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def ollama_is_up() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def safe_json(req) -> dict:
    """Parse JSON body; return {} on failure."""
    try:
        return req.get_json(force=True, silent=True) or {}
    except Exception:
        return {}


def build_openai_chat_response(
    content: str,
    requested_model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict:
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   requested_model,
        "choices": [{
            "index":   0,
            "message": {"role": "assistant", "content": content or "(empty response)"},
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens or len(content.split()),
            "total_tokens":      prompt_tokens + (completion_tokens or len(content.split())),
        },
        "system_fingerprint": f"fp_{uuid.uuid4().hex[:8]}",
    }


def messages_to_prompt(messages: list) -> str:
    """Flatten a messages array to a plain-text prompt for /api/generate."""
    parts = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):          # multimodal blocks
            content = " ".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role, role.title())
        parts.append(f"{label}: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# STREAMING HELPER
# ─────────────────────────────────────────────────────────────

def stream_chat(payload: dict, requested_model: str):
    """
    Call Ollama's streaming /api/chat and forward SSE chunks
    in OpenAI format.
    """
    def generate():
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={**payload, "stream": True},
                stream=True,
                timeout=OLLAMA_TIMEOUT,
            ) as r:
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk    = json.loads(raw_line)
                        delta    = chunk.get("message", {}).get("content", "")
                        done     = chunk.get("done", False)
                        sse_data = {
                            "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object":  "chat.completion.chunk",
                            "created": int(time.time()),
                            "model":   requested_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": delta} if delta else {},
                                "finish_reason": "stop" if done else None,
                            }],
                        }
                        yield f"data: {json.dumps(sse_data)}\n\n"
                        if done:
                            yield "data: [DONE]\n\n"
                            return
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            err = {"error": {"message": str(exc), "type": "proxy_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":       "no-cache",
            "X-Accel-Buffering":   "no",
            "Connection":          "keep-alive",
            "Transfer-Encoding":   "chunked",
        },
    )


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

# ── / ─────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    base = request.url_root.rstrip("/")
    return jsonify({
        "service":        "Ollama OpenAI-Compatible Proxy",
        "version":        "4.0.0",
        "actual_model":   REAL_MODEL,
        "ollama_backend": OLLAMA_URL,
        "endpoints": {
            "health":           f"{base}/health",
            "debug":            f"{base}/v1/debug",
            "models_list":      f"{base}/v1/models",
            "chat_completions": f"{base}/v1/chat/completions",
            "completions":      f"{base}/v1/completions",
            "responses":        f"{base}/v1/responses",
            "embeddings":       f"{base}/v1/embeddings",
        },
        "quick_start": {
            "provider":  "openai",
            "baseUrl":   f"{base}/v1",
            "model":     "gpt-4",
            "apiKey":    "any-value",
        },
        "curl_example": (
            f"curl -X POST {base}/v1/chat/completions "
            "-H 'Content-Type: application/json' "
            "-d '{\"model\":\"gpt-4\","
            "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
        ),
    })


# ── /health ───────────────────────────────────────────────────
@app.route("/health", methods=["GET", "HEAD"])
def health():
    if request.method == "HEAD":
        return "", 200
    status = "connected" if ollama_is_up() else "disconnected"
    code   = 200 if status == "connected" else 503
    return jsonify({
        "status":        "healthy" if code == 200 else "degraded",
        "proxy":         "running",
        "ollama":        status,
        "model":         REAL_MODEL,
        "ctx_size":      CTX_SIZE,
        "temperature":   TEMPERATURE,
    }), code


# ── /v1 ───────────────────────────────────────────────────────
@app.route("/v1", methods=["GET"])
def v1_index():
    base = request.url_root.rstrip("/")
    return jsonify({
        "message":      "OpenAI API v1 compatibility layer for Ollama",
        "baseUrl":      f"{base}/v1",
        "chat":         f"{base}/v1/chat/completions",
        "completions":  f"{base}/v1/completions",
        "responses":    f"{base}/v1/responses",
        "models":       f"{base}/v1/models",
        "embeddings":   f"{base}/v1/embeddings",
    })


# ── /v1/debug ─────────────────────────────────────────────────
@app.route("/v1/debug", methods=["GET"])
@app.route("/api/debug", methods=["GET"])
def debug():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_info = {
            "connected":   True,
            "status_code": r.status_code,
            "models":      [m.get("name") for m in r.json().get("models", [])],
        }
    except Exception as e:
        ollama_info = {"connected": False, "error": str(e)}

    return jsonify({
        "ollama_url":     OLLAMA_URL,
        "ollama":         ollama_info,
        "actual_model":   REAL_MODEL,
        "model_aliases":  list(MODEL_ALIASES.keys()),
        "proxy_port":     PROXY_PORT,
        "ctx_size":       CTX_SIZE,
        "temperature":    TEMPERATURE,
        "debug_mode":     os.getenv("DEBUG_MODE", "false"),
    })


# ── /v1/models ────────────────────────────────────────────────
@app.route("/v1/models", methods=["GET"])
def list_models():
    ts = int(time.time())
    data = [
        {"id": REAL_MODEL,          "object": "model", "created": ts, "owned_by": "ollama"},
        {"id": "gpt-4o",            "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "gpt-4o-mini",       "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "gpt-4-turbo",       "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "gpt-4",             "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "gpt-3.5-turbo",     "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "gpt-5.1-codex-max", "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "claude-3-opus",     "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "claude-3-sonnet",   "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "llama3",            "object": "model", "created": ts, "owned_by": "proxy"},
        {"id": "mistral",           "object": "model", "created": ts, "owned_by": "proxy"},
    ]
    return jsonify({"object": "list", "data": data})


@app.route("/v1/models/<path:model_id>", methods=["GET"])
def get_model(model_id):
    actual = resolve_model(model_id)
    return jsonify({
        "id":         model_id,
        "object":     "model",
        "created":    int(time.time()),
        "owned_by":   "ollama" if model_id == REAL_MODEL else "proxy",
        "mapped_to":  actual,
    })


# ── /v1/chat/completions ─────────────────────────────────────
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data            = safe_json(request)
    requested_model = data.get("model", "gpt-4")
    actual_model    = resolve_model(requested_model)

    if requested_model != actual_model:
        logger.info("model map: %s → %s", requested_model, actual_model)

    messages = data.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": "Say hello"}]

    # Inject system prompt if missing
    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": "You are a helpful, concise assistant."}] + messages

    stream      = data.get("stream", False)
    max_tokens  = data.get("max_tokens") or data.get("max_completion_tokens") or 1024
    temperature = data.get("temperature", TEMPERATURE)
    top_p       = data.get("top_p", 0.95)

    payload = {
        "model":    actual_model,
        "messages": messages,
        "stream":   stream,
        "options": {
            "temperature": temperature,
            "top_p":       top_p,
            "num_predict": max_tokens,
            "num_ctx":     CTX_SIZE,
        },
    }

    if stream:
        return stream_chat(payload, requested_model)

    # ── Non-streaming ─────────────────────────────────────────
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        if resp.status_code == 200:
            result  = resp.json()
            content = result.get("message", {}).get("content", "").strip()
            return jsonify(build_openai_chat_response(
                content         = content,
                requested_model = requested_model,
                prompt_tokens   = result.get("prompt_eval_count", 0),
                completion_tokens = result.get("eval_count", 0),
            ))

        logger.warning("/api/chat returned %s — falling back to /api/generate", resp.status_code)

    except requests.exceptions.ConnectionError:
        logger.error("Cannot reach Ollama at %s", OLLAMA_URL)
        return jsonify({"error": {"message": f"Ollama unreachable at {OLLAMA_URL}", "type": "connection_error"}}), 503
    except Exception as exc:
        logger.exception("Unexpected error in /api/chat")
        return jsonify({"error": {"message": str(exc), "type": "server_error"}}), 500

    # ── Fallback: /api/generate ───────────────────────────────
    try:
        gen_payload = {
            "model":   actual_model,
            "prompt":  messages_to_prompt(messages),
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": CTX_SIZE},
        }
        r2 = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
        r2.raise_for_status()
        content = r2.json().get("response", "").strip()
        return jsonify(build_openai_chat_response(content, requested_model))
    except Exception as exc:
        logger.exception("Fallback /api/generate also failed")
        return jsonify({"error": {"message": str(exc), "type": "server_error"}}), 500


# ── /v1/completions (legacy text completions) ─────────────────
@app.route("/v1/completions", methods=["POST"])
def completions():
    data            = safe_json(request)
    requested_model = data.get("model", "gpt-3.5-turbo")
    actual_model    = resolve_model(requested_model)

    prompt = data.get("prompt", "Say hello")
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)

    max_tokens  = data.get("max_tokens", 256)
    temperature = data.get("temperature", TEMPERATURE)

    payload = {
        "model":   actual_model,
        "prompt":  f"User: {prompt}\nAssistant:",
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": CTX_SIZE},
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return jsonify({
            "id":      f"cmpl-{uuid.uuid4().hex[:12]}",
            "object":  "text_completion",
            "created": int(time.time()),
            "model":   requested_model,
            "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
            "usage":   {"prompt_tokens": 0, "completion_tokens": len(text.split()), "total_tokens": 0},
        })
    except Exception as exc:
        logger.exception("Error in /v1/completions")
        return jsonify({"error": {"message": str(exc), "type": "server_error"}}), 500


# ── /v1/responses (Responses API — OpenAI beta) ───────────────
@app.route("/v1/responses", methods=["POST"])
def responses():
    data            = safe_json(request)
    requested_model = data.get("model", "gpt-4o")
    actual_model    = resolve_model(requested_model)

    raw_input = data.get("input", "")
    if isinstance(raw_input, dict):
        raw_input = raw_input.get("content", "")
    elif isinstance(raw_input, list):
        raw_input = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw_input
        )
    if not raw_input:
        raw_input = "Say hello"

    instructions    = data.get("instructions", "")
    max_out_tokens  = data.get("max_output_tokens", 1024)
    temperature     = data.get("temperature", TEMPERATURE)

    full_prompt = (f"{instructions}\n\n" if instructions else "") + f"User: {raw_input}\nAssistant:"

    payload = {
        "model":   actual_model,
        "prompt":  full_prompt,
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": max_out_tokens, "num_ctx": CTX_SIZE},
    }

    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        content = resp.json().get("response", "").strip()
        return jsonify({
            "id":      f"resp_{uuid.uuid4().hex[:12]}",
            "object":  "response",
            "created": int(time.time()),
            "model":   requested_model,
            "output": [{
                "type":    "message",
                "role":    "assistant",
                "content": [{"type": "output_text", "text": content}],
            }],
            "usage": {
                "input_tokens":  0,
                "output_tokens": len(content.split()),
                "total_tokens":  0,
            },
        })
    except Exception as exc:
        logger.exception("Error in /v1/responses")
        return jsonify({"error": {"message": str(exc), "type": "server_error"}}), 500


# ── /v1/embeddings (stub — returns zero vectors) ──────────────
@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    data  = safe_json(request)
    inp   = data.get("input", "")
    texts = [inp] if isinstance(inp, str) else inp
    DIM   = 768
    return jsonify({
        "object": "list",
        "data": [
            {
                "object":    "embedding",
                "index":     i,
                "embedding": [0.0] * DIM,   # stub — Ollama doesn't expose embeddings on all models
            }
            for i, _ in enumerate(texts)
        ],
        "model": data.get("model", REAL_MODEL),
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })


# ── Catch-all ─────────────────────────────────────────────────
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def catch_all(path):
    return jsonify({
        "error":   f"Endpoint '/{path}' not found",
        "hint":    "Available: /v1/chat/completions, /v1/completions, /v1/responses, /v1/models, /v1/embeddings, /health",
        "baseUrl": request.url_root.rstrip("/") + "/v1",
    }), 404


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banner = "=" * 68
    logger.info(banner)
    logger.info("🚀  Ollama Proxy v4.0 — Full OpenAI Compatibility")
    logger.info(banner)
    logger.info("📍  Proxy:    http://0.0.0.0:%d", PROXY_PORT)
    logger.info("🔗  Ollama:   %s", OLLAMA_URL)
    logger.info("🤖  Model:    %s", REAL_MODEL)
    logger.info("📐  Ctx:      %d tokens", CTX_SIZE)
    logger.info("🌡️   Temp:     %.1f", TEMPERATURE)
    logger.info(banner)
    logger.info("✅  Roo / Cline config:")
    logger.info("    provider: openai")
    logger.info("    baseUrl:  http://localhost:%d/v1", PROXY_PORT)
    logger.info("    model:    gpt-4  (any name works)")
    logger.info("    apiKey:   any-value")
    logger.info(banner)
    app.run(host="0.0.0.0", port=PROXY_PORT, threaded=True, debug=False)
