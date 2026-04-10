#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - /api/chat + /api/generate fallback"""
import os, sys, json, time, logging, requests
from flask import Flask, request, jsonify, Response

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
app = Flask(__name__)

PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = 'gemma4:e4b'
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
CTX_SIZE = int(os.getenv('CTX_SIZE', '8192'))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_TIMEOUT = 600

def cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
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

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS': return '', 204
    try:
        data = request.get_json(force=True, silent=True) or {}
    except:
        return jsonify({"error": "Invalid JSON"}), 400
    
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    max_tok = data.get('max_tokens', data.get('max_completion_tokens', 2048))
    temp = data.get('temperature', TEMPERATURE)
    ollama_msgs = [{'role': m.get('role','user'), 'content': m.get('content','')} for m in messages]
    payload = {'model': MODEL, 'messages': ollama_msgs, 'stream': stream, 'options': {'temperature': temp, 'num_ctx': CTX_SIZE, 'num_predict': max_tok}}
    
    try:
        if stream:
            def gen():
                try:
                    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
                        if r.status_code == 404:
                            yield f' {json.dumps({"error": "Streaming requires /api/chat endpoint - upgrade Ollama"})}\n\n'.encode()
                            return
                        for line in r.iter_lines():
                            if line: yield line + b'\n'
                except Exception as e:
                    yield f' {json.dumps({"error": str(e)})}\n\n'.encode()
            return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})
        else:
            content, finish_reason, prompt_tok, comp_tok = None, "stop", 0, 0
            try:
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 404:
                    raise requests.exceptions.HTTPError("Endpoint not found", response=resp)
                resp.raise_for_status()
                res = resp.json()
                if isinstance(res, dict):
                    if 'message' in res and isinstance(res['message'], dict):
                        content = res['message'].get('content', '')
                    elif 'content' in res:
                        content = res['content']
                    prompt_tok = res.get('prompt_eval_count', res.get('prompt_tokens', 0))
                    comp_tok = res.get('eval_count', res.get('completion_tokens', 0))
                    finish_reason = res.get('done_reason', res.get('finish_reason', 'stop'))
            except (requests.exceptions.HTTPError, KeyError):
                logger.warning("/api/chat failed, trying /api/generate fallback...")
                prompt = '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in ollama_msgs]) + "\nAssistant:"
                gen_payload = {'model': MODEL, 'prompt': prompt, 'stream': False, 'options': payload['options']}
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                res = resp.json()
                content = res.get('response', '')
                prompt_tok = res.get('prompt_eval_count', 0)
                comp_tok = res.get('eval_count', 0)
                finish_reason = res.get('done_reason', 'stop')
            if content is None: content = ''
            return jsonify({
                "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": prompt_tok, "completion_tokens": comp_tok, "total_tokens": prompt_tok + comp_tok}
            })
    except requests.exceptions.Timeout:
        return jsonify({"error": f"Timeout: {MODEL} is still processing"}), 504
    except requests.exceptions.ConnectionError as e:
        return jsonify({"error": f"Ollama unavailable: {str(e)}"}), 502
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else None
        if status == 404:
            return jsonify({"error": "Endpoint not found - please upgrade Ollama", "hint": "curl -fsSL https://ollama.com/install.sh | sh"}), 502
        return jsonify({"error": f"Ollama HTTP {status}: {str(e)}"}), 502
    except Exception as e:
        logger.error(f"Proxy error: {type(e).__name__}: {e}")
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:300]}"}), 500

@app.route('/api/tags')
def tags():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        return Response(r.content, status=r.status_code, content_type=r.headers.get('Content-Type', 'application/json'))
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route('/v1/models')
def models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        data = r.json()
        models = [{"id": m.get('name', MODEL), "object": "model", "created": int(time.time()), "owned_by": "ollama"} for m in data.get('models', [])]
        if not any(m['id'] == MODEL for m in models):
            models.append({"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"})
        return jsonify({"object": "list", "data": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route('/')
def root():
    return jsonify({"service": "Ollama Proxy", "model": MODEL, "endpoints": ["/health", "/api/tags", "/v1/models", "/v1/chat/completions"]})

if __name__ == '__main__':
    logger.info(f"🚀 Proxy starting: 0.0.0.0:{PROXY_PORT} → Model: {MODEL}")
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True)
