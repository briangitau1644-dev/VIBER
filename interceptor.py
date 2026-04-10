#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - supports ALL common OpenAI endpoints"""
import os, sys, json, time, logging, requests
from flask import Flask, request, jsonify, Response

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
app = Flask(__name__)

PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = 'gemma4:e4b'  # HARD CODED
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
CTX_SIZE = int(os.getenv('CTX_SIZE', '8192'))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_TIMEOUT = 600

def cors(r):
    r.headers.update({'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept'})
    return r

@app.after_request
def after(r): return cors(r)

@app.route('/health')
def health():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        return jsonify({"status": "healthy", "model": MODEL}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503

# === OPENAI-COMPATIBLE ENDPOINTS - ALL ROUTE TO CHAT LOGIC ===

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@app.route('/v1/completions', methods=['POST', 'OPTIONS'])  # Legacy endpoint
@app.route('/v1/responses', methods=['POST', 'OPTIONS'])     # New Responses API
def handle_chat_request():
    """Handle all chat-style endpoints with automatic routing"""
    if request.method == 'OPTIONS': return '', 204
    
    endpoint = request.path
    logger.info(f"→ Request to {endpoint}")
    
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Extract params (handle both chat and completions formats)
    messages = data.get('messages')
    prompt = data.get('prompt')  # For /v1/completions
    stream = data.get('stream', False)
    max_tok = data.get('max_tokens', data.get('max_completion_tokens', 2048))
    temp = data.get('temperature', TEMPERATURE)
    
    # Convert prompt to messages if needed (for /v1/completions)
    if not messages and prompt:
        messages = [{'role': 'user', 'content': prompt}]
    if not messages:
        return jsonify({"error": "Missing 'messages' or 'prompt' in request"}), 400
    
    ollama_msgs = [{'role': m.get('role','user'), 'content': m.get('content','')} for m in messages]
    payload = {'model': MODEL, 'messages': ollama_msgs, 'stream': stream, 'options': {'temperature': temp, 'num_ctx': CTX_SIZE, 'num_predict': max_tok}}
    
    try:
        if stream:
            def gen():
                try:
                    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
                        if r.status_code == 404:
                            yield f' {json.dumps({"error": "Streaming requires /api/chat - upgrade Ollama"})}\n\n'.encode(); return
                        for line in r.iter_lines():
                            if line: yield line + b'\n'
                except Exception as e:
                    yield f' {json.dumps({"error": str(e)})}\n\n'.encode()
            return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})
        else:
            content, finish_reason, pt, ct = None, "stop", 0, 0
            try:
                # Try /api/chat first
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 404: raise requests.exceptions.HTTPError("404", response=resp)
                resp.raise_for_status()
                res = resp.json()
                if isinstance(res, dict):
                    if 'message' in res and isinstance(res['message'], dict): content = res['message'].get('content', '')
                    elif 'content' in res: content = res['content']
                    pt, ct = res.get('prompt_eval_count', 0), res.get('eval_count', 0)
                    finish_reason = res.get('done_reason', 'stop')
            except:
                # Fallback to /api/generate
                prompt_text = '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in ollama_msgs]) + "\nAssistant:"
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json={'model': MODEL, 'prompt': prompt_text, 'stream': False, 'options': payload['options']}, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                res = resp.json()
                content = res.get('response', '')
                pt, ct = res.get('prompt_eval_count', 0), res.get('eval_count', 0)
            
            if content is None: content = ''
            
            # Return format based on endpoint
            if '/responses' in endpoint:
                # Responses API format
                return jsonify({
                    "id": f"resp-{int(time.time())}",
                    "object": "response",
                    "created_at": int(time.time()),
                    "status": "completed",
                    "model": MODEL,
                    "output": [{"type": "message", "content": [{"type": "input_text", "text": content}]}],
                    "usage": {"input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct}
                })
            elif '/completions' in endpoint and '/chat' not in endpoint:
                # Legacy completions format
                return jsonify({
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [{"text": content, "index": 0, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
                })
            else:
                # Standard chat completions format
                return jsonify({
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
                })
    except requests.exceptions.Timeout:
        return jsonify({"error": "Timeout: model is still processing"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Ollama unavailable"}), 502
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return jsonify({"error": str(e)[:300]}), 500

# === OTHER OPENAI ENDPOINTS ===

@app.route('/v1/models', methods=['GET'])
@app.route('/models', methods=['GET'])
def list_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        data = r.json()
        models = [{"id": m.get('name', MODEL), "object": "model", "created": int(time.time()), "owned_by": "ollama"} for m in data.get('models', [])]
        if not any(m['id'] == MODEL for m in models):
            models.append({"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"})
        return jsonify({"object": "list", "data": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    # Ollama doesn't support embeddings via API yet - return helpful error
    return jsonify({
        "error": "Embeddings not supported via this proxy. Use Ollama's /api/embeddings endpoint directly.",
        "hint": f"POST http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
    }), 501

@app.route('/api/tags', methods=['GET'])
def ollama_tags():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        return Response(r.content, status=r.status_code, content_type=r.headers.get('Content-Type', 'application/json'))
    except Exception as e:
        return jsonify({"error": str(e)}), 502

# === CATCH-ALL FOR DEBUGGING ===

@app.route('/v1/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def catch_all(path):
    """Catch undefined routes and return helpful 404"""
    logger.warning(f"404: Unknown endpoint /{path}")
    return jsonify({
        "error": f"Endpoint /{path} not supported",
        "available_endpoints": [
            "/v1/chat/completions",
            "/v1/completions", 
            "/v1/responses",
            "/v1/models",
            "/health",
            "/api/tags"
        ],
        "hint": "Use provider: 'openai' with baseUrl ending in /v1"
    }), 404

@app.route('/')
def root():
    return jsonify({
        "service": "Ollama Proxy",
        "model": MODEL,
        "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/models", "/health"],
        "cors": "enabled"
    })

if __name__ == '__main__':
    logger.info(f"🚀 Proxy: 0.0.0.0:{PROXY_PORT} → Model: {MODEL}")
    logger.info(f"   Endpoints: /v1/chat/completions, /v1/completions, /v1/responses, /v1/models")
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True)
