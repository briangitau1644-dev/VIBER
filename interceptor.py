#!/usr/bin/env python3
"""Flask-based Ollama proxy for gemma4:e4b - robust HTTP handling with /api/chat + /api/generate fallback"""
import os, sys, json, time, logging, requests
from flask import Flask, request, jsonify, Response

# Configure logging to stdout immediately
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Hardcoded config - gemma4:e4b ONLY
PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = 'gemma4:e4b'  # FIXED: Only this model
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
CTX_SIZE = int(os.getenv('CTX_SIZE', '8192'))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_TIMEOUT = 600  # 10 minutes for large model generation

# CORS helper
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    return response

@app.after_request
def after_request(response):
    return add_cors(response)

@app.before_request
def log_incoming():
    logger.info(f"→ {request.method} {request.path} from {request.remote_addr}")
    if request.data:
        logger.debug(f"→ Body ({len(request.data)} bytes): {request.data[:200]}...")

@app.route('/health', methods=['GET'])
def health():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        return jsonify({"status": "healthy" if resp.status_code == 200 else "degraded", "model": MODEL}), 200 if resp.status_code == 200 else 503
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 503

@app.route('/api/tags', methods=['GET'])
def list_tags():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))
    except Exception as e:
        logger.error(f"Tags fetch failed: {e}")
        return jsonify({"error": str(e)}), 502

@app.route('/v1/models', methods=['GET'])
def openai_models():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        data = resp.json()
        models = [{"id": m.get('name', MODEL), "object": "model", "created": int(time.time()), "owned_by": "ollama"} for m in data.get('models', [])]
        if not any(m['id'] == MODEL for m in models):
            models.append({"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"})
        return jsonify({"object": "list", "data": models})
    except Exception as e:
        logger.error(f"Models list failed: {e}")
        return jsonify({"error": str(e)}), 502

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content-type: {request.content_type}")
        
        if request.is_json:
            req_data = request.get_json(force=True, silent=True) or {}
        else:
            try:
                req_data = json.loads(request.data) if request.data else {}
            except json.JSONDecodeError:
                logger.error(f"Failed to parse request body: {request.data[:200]}")
                return jsonify({"error": "Invalid JSON in request body"}), 400
    except Exception as e:
        logger.error(f"Request parse error: {e}")
        return jsonify({"error": f"Failed to parse request: {str(e)}"}), 400

    # Force gemma4:e4b regardless of request
    messages = req_data.get('messages', [])
    stream = req_data.get('stream', False)
    max_tok = req_data.get('max_tokens', req_data.get('max_completion_tokens', 2048))
    temp = req_data.get('temperature', TEMPERATURE)

    # Transform to Ollama format
    ollama_msgs = [{'role': m.get('role','user'), 'content': m.get('content','')} for m in messages]
    payload = {
        'model': MODEL,
        'messages': ollama_msgs,
        'stream': stream,
        'options': {'temperature': temp, 'num_ctx': CTX_SIZE, 'num_predict': max_tok}
    }

    logger.info(f"→ Forwarding to Ollama: model={MODEL}, msgs={len(ollama_msgs)}, max_tok={max_tok}")

    try:
        if stream:
            def generate():
                try:
                    # Try /api/chat first for streaming
                    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as resp:
                        if resp.status_code == 404:
                            logger.warning("/api/chat not found, streaming not supported on legacy Ollama")
                            yield f'data: {json.dumps({"error": "Streaming requires Ollama with /api/chat endpoint"})}\n\n'.encode()
                            return
                        for line in resp.iter_lines():
                            if line:
                                yield line + b'\n'
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404:
                        logger.warning("/api/chat endpoint not found - Ollama may be outdated")
                        yield f'data: {json.dumps({"error": "Endpoint /api/chat not found. Please upgrade Ollama."})}\n\n'.encode()
                    else:
                        logger.error(f"Stream HTTP error: {e}")
                        yield f'data: {json.dumps({"error": str(e)})}\n\n'.encode()
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    yield f'data: {json.dumps({"error": str(e)})}\n\n'.encode()
            
            return Response(generate(), mimetype='text/event-stream', headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            })
        else:
            # Non-streaming: try /api/chat first, fallback to /api/generate
            logger.info(f"⏳ Calling Ollama /api/chat (timeout={OLLAMA_TIMEOUT}s)...")
            start = time.time()
            content = None
            finish_reason = "stop"
            prompt_tokens = 0
            completion_tokens = 0
            
            try:
                # === ATTEMPT 1: Modern /api/chat endpoint ===
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
                
                if resp.status_code == 404:
                    logger.warning("/api/chat returned 404 - falling back to /api/generate")
                    raise requests.exceptions.HTTPError("Endpoint not found", response=resp)
                
                resp.raise_for_status()
                elapsed = time.time() - start
                logger.info(f"← /api/chat responded in {elapsed:.1f}s (HTTP {resp.status_code})")
                
                raw = resp.text
                logger.debug(f"← Raw response ({len(raw)} bytes): {raw[:500]}...")
                
                try:
                    res = resp.json()
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse Ollama JSON: {raw[:300]}")
                    return jsonify({
                        "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                        "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": raw[:4000]}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": len(raw), "total_tokens": len(raw)}
                    })

                # Extract content from /api/chat format
                if isinstance(res, dict):
                    if 'message' in res and isinstance(res['message'], dict):
                        content = res['message'].get('content', '')
                    elif 'content' in res:
                        content = res['content']
                    prompt_tokens = res.get('prompt_eval_count', res.get('prompt_tokens', 0))
                    completion_tokens = res.get('eval_count', res.get('completion_tokens', 0))
                    finish_reason = res.get('done_reason', res.get('finish_reason', 'stop'))
                
                if content is None:
                    logger.warning(f"Content not found in /api/chat response: {json.dumps(res)[:300]}")
                    content = ''
                    
            except (requests.exceptions.HTTPError, KeyError) as e:
                # === FALLBACK: Legacy /api/generate endpoint ===
                logger.warning(f"/api/chat failed ({e}), trying /api/generate fallback...")
                
                # Build prompt from messages for legacy endpoint
                prompt_parts = []
                for m in ollama_msgs:
                    if m['role'] == 'system':
                        prompt_parts.append(f"System: {m['content']}")
                    elif m['role'] == 'user':
                        prompt_parts.append(f"User: {m['content']}")
                    elif m['role'] == 'assistant':
                        prompt_parts.append(f"Assistant: {m['content']}")
                prompt = '\n'.join(prompt_parts) + "\nAssistant:"
                
                gen_payload = {
                    'model': MODEL,
                    'prompt': prompt,
                    'stream': False,
                    'options': payload['options']
                }
                
                logger.info(f"⏳ Calling Ollama /api/generate fallback (timeout={OLLAMA_TIMEOUT}s)...")
                start = time.time()
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                elapsed = time.time() - start
                logger.info(f"← /api/generate responded in {elapsed:.1f}s (HTTP {resp.status_code})")
                
                raw = resp.text
                logger.debug(f"← Raw generate response ({len(raw)} bytes): {raw[:500]}...")
                
                try:
                    res = resp.json()
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse generate JSON: {raw[:300]}")
                    return jsonify({
                        "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                        "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": raw[:4000]}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": len(raw), "total_tokens": len(raw)}
                    })
                
                # Extract content from /api/generate format
                content = res.get('response', '')
                prompt_tokens = res.get('prompt_eval_count', 0)
                completion_tokens = res.get('eval_count', 0)
                finish_reason = res.get('done_reason', 'stop')
                
                if content is None:
                    logger.warning(f"Content not found in /api/generate response: {json.dumps(res)[:300]}")
                    content = ''
            
            logger.info(f"← Final content length: {len(content) if content else 0}")

            return jsonify({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion", 
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content or ''},
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            })
            
    except requests.exceptions.Timeout:
        logger.error(f"Ollama timeout after {OLLAMA_TIMEOUT}s")
        return jsonify({"error": f"Timeout: {MODEL} is still processing (try increasing timeout)"}), 504
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        return jsonify({"error": f"Ollama unavailable at {OLLAMA_URL}: {str(e)}"}), 502
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else None
        logger.error(f"Ollama HTTP {status}: {e}")
        if status == 404:
            return jsonify({
                "error": f"Endpoint not found: {OLLAMA_URL}/api/chat or /api/generate. Please upgrade Ollama to latest version.",
                "hint": "Run: curl -fsSL https://ollama.com/install.sh | sh"
            }), 502
        return jsonify({"error": f"Ollama HTTP {status}: {str(e)}"}), 502
    except Exception as e:
        logger.error(f"Proxy error: {type(e).__name__}: {e}", exc_info=True)
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:300]}"}), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "service": "Ollama Proxy",
        "model": MODEL,
        "endpoints": ["/health", "/api/tags", "/v1/models", "/v1/chat/completions"],
        "cors": "enabled",
        "ollama_url": OLLAMA_URL
    })

@app.errorhandler(404)
def not_found(e):
    logger.warning(f"404: {request.path}")
    return jsonify({"error": "Not found", "path": request.path, "available_endpoints": ["/health", "/api/tags", "/v1/models", "/v1/chat/completions"]}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500: {e}", exc_info=True)
    return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    logger.info(f"🚀 Flask proxy starting on 0.0.0.0:{PROXY_PORT}")
    logger.info(f"   → Model: {MODEL} (hardcoded)")
    logger.info(f"   → Ollama: {OLLAMA_URL} | Temp: {TEMPERATURE} | Ctx: {CTX_SIZE} | Timeout: {OLLAMA_TIMEOUT}s")
    logger.info(f"   → Endpoints: /api/chat (primary) + /api/generate (fallback)")
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True, debug=False)
