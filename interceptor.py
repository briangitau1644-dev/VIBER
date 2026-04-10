#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - FULL tool calling support + OpenAI format compliance"""
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
    r.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept, OpenAI-Beta'
    })
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
@app.route('/v1/completions', methods=['POST', 'OPTIONS'])
@app.route('/v1/responses', methods=['POST', 'OPTIONS'])
def handle_chat_request():
    if request.method == 'OPTIONS': return '', 204
    
    endpoint = request.path
    logger.info(f"→ {request.method} {endpoint}")
    
    # Parse request
    try:
        data = request.get_json(force=True, silent=True) or {}
    except: data = {}
    
    # Extract params
    messages = data.get('messages', [])
    prompt = data.get('prompt')
    tools = data.get('tools')
    stream = data.get('stream', False)
    max_tok = data.get('max_tokens', data.get('max_completion_tokens', data.get('max_output_tokens', 2048)))
    temp = data.get('temperature', TEMPERATURE)
    
    # Build messages
    if not messages and prompt:
        messages = [{'role': 'user', 'content': prompt}]
    if not messages:
        return jsonify({"error": "Missing 'messages' or 'prompt'"}), 400
    
    # Format messages for Ollama
    ollama_msgs = []
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get('role', 'user'), m.get('content', '')
            # Handle content arrays (multimodal)
            if isinstance(content, list):
                text_parts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
                content = ' '.join(text_parts)
            ollama_msgs.append({'role': role, 'content': content or ''})
    
    # Build Ollama payload
    ollama_payload = {
        'model': MODEL,
        'messages': ollama_msgs,
        'stream': stream,
        'options': {'temperature': temp, 'num_ctx': CTX_SIZE, 'num_predict': max_tok}
    }
    if tools:
        ollama_payload['tools'] = tools  # Ollama v0.1.25+ supports OpenAI tool format
    
    logger.info(f"→ Forwarding to Ollama: model={MODEL}, tools={bool(tools)}")
    
    try:
        if stream:
            def gen():
                try:
                    with requests.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
                        for line in r.iter_lines():
                            if line: yield line + b'\n'
                except Exception as e:
                    yield f' {json.dumps({"error": str(e)})}\n\n'.encode()
            return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})
        else:
            # Non-streaming
            try:
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload, timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 404:
                    raise requests.exceptions.HTTPError("404", response=resp)
                resp.raise_for_status()
                res = resp.json()
            except Exception:
                # Fallback to /api/generate
                prompt_text = '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in ollama_msgs]) + "\nAssistant:"
                resp = requests.post(f"{OLLAMA_URL}/api/generate", 
                    json={'model': MODEL, 'prompt': prompt_text, 'stream': False, 'options': ollama_payload['options']}, 
                    timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                res = resp.json()
                # Generate doesn't support tools, return as text
                return jsonify({
                    "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                    "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": res.get('response', '')}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                })
            
            # Extract response
            message = res.get('message', {})
            content = message.get('content')
            tool_calls = message.get('tool_calls')
            
            # Build OpenAI-compatible message
            openai_message = {"role": "assistant"}
            
            # Handle tool calls (OpenAI format)
            if tool_calls:
                openai_message["tool_calls"] = []
                for i, tc in enumerate(tool_calls):
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    # Ollama sometimes returns args as dict, sometimes as JSON string
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    openai_message["tool_calls"].append({
                        "id": f"call_{int(time.time())}_{i}",
                        "type": "function",
                        "function": {"name": func.get("name", ""), "arguments": args if args else "{}"}
                    })
                openai_message["content"] = None  # OpenAI spec: content is null when tool_calls exist
                finish_reason = "tool_calls"
            else:
                openai_message["content"] = content if content is not None else ""
                finish_reason = res.get('done_reason', 'stop')
            
            pt = res.get('prompt_eval_count', 0)
            ct = res.get('eval_count', 0)
            
            return jsonify({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{"index": 0, "message": openai_message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
            })
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Timeout: model is still processing"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Ollama unavailable at {OLLAMA_URL}"}), 502
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:200]}"}), 500

@app.route('/v1/models', methods=['GET'])
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

@app.route('/api/tags', methods=['GET'])
def ollama_tags():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        return Response(r.content, status=r.status_code, content_type=r.headers.get('Content-Type', 'application/json'))
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route('/')
def root():
    return jsonify({"service": "Ollama Proxy", "model": MODEL, "tool_support": True})

if __name__ == '__main__':
    logger.info(f"🚀 Proxy: 0.0.0.0:{PROXY_PORT} → Model: {MODEL} | Tool Calls: ENABLED")
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True)
