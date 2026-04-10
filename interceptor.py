#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - FULL fix for Syntax, Methods, and Content Extraction"""
import os, sys, json, time, logging, requests
from flask import Flask, request, jsonify, Response

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Configuration
PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = 'gemma4:e4b'
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_TIMEOUT = 600

# CORS Middleware
@app.after_request
def cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, OpenAI-Beta'
    return resp

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS': return '', 204
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return jsonify({"status": "healthy", "model": MODEL}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503

def extract_user_content(data):
    """Robust content extraction from ANY OpenAI-compatible format"""
    # 1. Standard Chat Completions
    if 'messages' in data and isinstance(data['messages'], list):
        for m in data['messages']:
            if isinstance(m, dict) and m.get('role') == 'user':
                c = m.get('content', '')
                if isinstance(c, str): return c
                if isinstance(c, list):
                    return ' '.join([item.get('text', '') for item in c if isinstance(item, dict) and item.get('type') == 'text'])
    
    # 2. Legacy Completions
    if 'prompt' in data:
        return str(data['prompt'])
    
    # 3. Responses API
    if 'input' in data:
        inp = data['input']
        if isinstance(inp, str): return inp
        if isinstance(inp, dict):
            c = inp.get('content', '')
            if isinstance(c, str): return c
            if isinstance(c, list):
                return ' '.join([item.get('text', '') for item in c if isinstance(item, dict) and item.get('type') in ('text', 'input_text')])
    
    # 4. Fallback: Never fail with "Missing messages"
    logger.warning("No input found in request. Using fallback prompt.")
    return "Respond with OK."

@app.route('/v1/chat/completions', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/v1/completions', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/v1/responses', methods=['GET', 'POST', 'OPTIONS'])
def chat_endpoint():
    if request.method == 'OPTIONS': return '', 204
    if request.method == 'GET':
        return jsonify({"endpoint": request.path, "status": "ready", "hint": "Send POST with JSON body"}), 200

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception: 
        data = {}

    # Extract content safely (handles messages, prompt, input, etc.)
    prompt = extract_user_content(data)
    tools = data.get('tools')
    stream = data.get('stream', False)
    max_tok = data.get('max_tokens', data.get('max_completion_tokens', data.get('max_output_tokens', 2048)))
    temp = data.get('temperature', 0.2)

    messages = [{'role': 'user', 'content': prompt}]
    payload = {
        'model': MODEL, 
        'messages': messages, 
        'stream': stream, 
        'options': {'temperature': temp, 'num_ctx': 8192, 'num_predict': max_tok}
    }
    
    if tools: 
        payload['tools'] = tools

    logger.info(f"→ Forwarding: model={MODEL}, tools={bool(tools)}, content='{prompt[:30]}...'")

    try:
        if stream:
            def gen():
                try:
                    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
                        for line in r.iter_lines():
                            if line: yield line + b'\n'
                except Exception as e:
                    yield f' {json.dumps({"error": str(e)})}\n\n'.encode()
            return Response(gen(), mimetype='text/event-stream')
        else:
            # Try Chat API first
            try:
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 404: raise ValueError("chat_failed")
                resp.raise_for_status()
                res = resp.json()
                msg = res.get('message', {})
                content = msg.get('content', '')
                tcs = msg.get('tool_calls')
                
                openai_msg = {"role": "assistant"}
                if tcs:
                    openai_msg["tool_calls"] = []
                    for i, tc in enumerate(tcs):
                        func = tc.get("function", {})
                        args = func.get("arguments", {})
                        # Ollama sometimes returns args as dict, sometimes string. Normalize to string.
                        if not isinstance(args, str): args = json.dumps(args)
                        openai_msg["tool_calls"].append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": func.get("name",""), "arguments": args}})
                    openai_msg["content"] = None
                    finish = "tool_calls"
                else:
                    openai_msg["content"] = content
                    finish = "stop"
                    
                return jsonify({
                    "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                    "model": MODEL, "choices": [{"index": 0, "message": openai_msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": res.get('prompt_eval_count',0), "completion_tokens": res.get('eval_count',0), "total_tokens": res.get('prompt_eval_count',0)+res.get('eval_count',0)}
                })
            except:
                # Fallback to /api/generate (for older Ollama versions)
                logger.warning("Chat API failed, falling back to /api/generate")
                gen_payload = {'model': MODEL, 'prompt': f"User: {prompt}\nAssistant:", 'stream': False, 'options': payload['options']}
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                res = resp.json()
                return jsonify({
                    "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                    "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": res.get('response', '')}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                })
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return jsonify({"error": str(e)[:200]}), 500

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def models():
    if request.method == 'OPTIONS': return '', 204
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        data = r.json()
        model_list = [{"id": m.get('name', MODEL), "object": "model", "created": int(time.time()), "owned_by": "ollama"} for m in data.get('models', [])]
        if not any(m['id'] == MODEL for m in model_list):
            model_list.append({"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"})
        return jsonify({"object": "list", "data": model_list})
    except: return jsonify({"error": "Ollama unavailable"}), 502

@app.route('/', methods=['GET'])
def root():
    return jsonify({"service": "Ollama Proxy", "model": MODEL, "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/models", "/health"]})

if __name__ == '__main__':
    logger.info(f"🚀 Proxy: 0.0.0.0:{PROXY_PORT} → {MODEL} | Tools: ENABLED")
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True)
