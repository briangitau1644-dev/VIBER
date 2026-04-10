#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - handles ALL OpenAI client request formats"""
import os, sys, json, time, logging, requests
from flask import Flask, request, jsonify, Response

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
app = Flask(__name__)

PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = 'gemma4:e4b'  # HARD CODED - ONLY THIS MODEL
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

def extract_user_content(data):
    """Extract user prompt from ANY OpenAI-compatible request format"""
    # Format 1: Standard chat completions
    if 'messages' in data and isinstance(data['messages'], list):
        for msg in data['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content')
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle content array: [{"type": "text", "text": "..."}]
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            return item.get('text', '')
    
    # Format 2: Legacy completions
    if 'prompt' in data:
        prompt = data['prompt']
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list) and len(prompt) > 0:
            return prompt[0] if isinstance(prompt[0], str) else str(prompt[0])
    
    # Format 3: Responses API (new)
    if 'input' in data:
        inp = data['input']
        if isinstance(inp, str):
            return inp
        elif isinstance(inp, dict):
            # {"type": "message", "content": "..."}
            if 'content' in inp:
                content = inp['content']
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'input_text':
                            return item.get('text', '')
        elif isinstance(inp, list):
            # [{"type": "message", "content": "..."}]
            for item in inp:
                if isinstance(item, dict):
                    if 'content' in item:
                        content = item['content']
                        if isinstance(content, str):
                            return content
                        elif isinstance(content, list):
                            for sub in content:
                                if isinstance(sub, dict) and sub.get('type') == 'input_text':
                                    return sub.get('text', '')
    
    # Format 4: Fallback - just use any string value found
    for key in ['query', 'question', 'text', 'user_input', 'prompt_text']:
        if key in data and isinstance(data[key], str):
            return data[key]
    
    # Format 5: Last resort - stringify the whole request
    logger.warning(f"Could not extract content from request: {json.dumps(data)[:200]}")
    return "Hello"  # Default fallback prompt

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@app.route('/v1/completions', methods=['POST', 'OPTIONS'])
@app.route('/v1/responses', methods=['POST', 'OPTIONS'])
def handle_chat_request():
    """Handle all chat-style endpoints with maximum format compatibility"""
    if request.method == 'OPTIONS': 
        return '', 204
    
    endpoint = request.path
    logger.info(f"→ {request.method} {endpoint} from {request.remote_addr}")
    
    # Parse request body with maximum tolerance
    try:
        if request.is_json:
            data = request.get_json(force=True, silent=True) or {}
        elif request.data:
            data = json.loads(request.data) if request.data else {}
        else:
            data = {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Request parse error: {e}")
        data = {}
    
    logger.debug(f"Request data keys: {list(data.keys())}")
    
    # Extract content using robust helper
    user_content = extract_user_content(data)
    logger.info(f"Extracted user content ({len(user_content)} chars): {user_content[:100]}...")
    
    # Extract other params with defaults
    stream = data.get('stream', False)
    max_tok = data.get('max_tokens', data.get('max_completion_tokens', data.get('max_output_tokens', 2048)))
    temp = data.get('temperature', TEMPERATURE)
    
    # Build Ollama payload
    ollama_msgs = [{'role': 'user', 'content': user_content}]
    payload = {
        'model': MODEL,
        'messages': ollama_msgs,
        'stream': stream,
        'options': {'temperature': temp, 'num_ctx': CTX_SIZE, 'num_predict': max_tok}
    }
    
    logger.info(f"→ Forwarding to Ollama: model={MODEL}, content_len={len(user_content)}")
    
    try:
        if stream:
            def gen():
                try:
                    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
                        if r.status_code == 404:
                            yield f' {json.dumps({"error": "Streaming requires /api/chat - upgrade Ollama"})}\n\n'.encode()
                            return
                        for line in r.iter_lines():
                            if line: yield line + b'\n'
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    yield f' {json.dumps({"error": str(e)})}\n\n'.encode()
            return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'})
        else:
            content, finish_reason, pt, ct = None, "stop", 0, 0
            try:
                # Try /api/chat first (modern)
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 404:
                    raise requests.exceptions.HTTPError("404", response=resp)
                resp.raise_for_status()
                res = resp.json()
                if isinstance(res, dict):
                    if 'message' in res and isinstance(res['message'], dict):
                        content = res['message'].get('content', '')
                    elif 'content' in res:
                        content = res['content']
                    pt = res.get('prompt_eval_count', res.get('prompt_tokens', 0))
                    ct = res.get('eval_count', res.get('completion_tokens', 0))
                    finish_reason = res.get('done_reason', res.get('finish_reason', 'stop'))
            except Exception as chat_err:
                logger.warning(f"/api/chat failed ({chat_err}), trying /api/generate fallback...")
                # Fallback to legacy /api/generate
                prompt_text = f"User: {user_content}\nAssistant:"
                gen_payload = {
                    'model': MODEL,
                    'prompt': prompt_text,
                    'stream': False,
                    'options': payload['options']
                }
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                res = resp.json()
                content = res.get('response', '')
                pt = res.get('prompt_eval_count', 0)
                ct = res.get('eval_count', 0)
                finish_reason = res.get('done_reason', 'stop')
            
            if content is None: 
                content = ''
            logger.info(f"← Response content length: {len(content)}")
            
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
                # Standard chat completions format (default)
                return jsonify({
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
                })
                
    except requests.exceptions.Timeout:
        logger.error("Ollama timeout")
        return jsonify({"error": "Timeout: model is still processing"}), 504
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        return jsonify({"error": f"Ollama unavailable at {OLLAMA_URL}"}), 502
    except Exception as e:
        logger.error(f"Proxy error: {type(e).__name__}: {e}", exc_info=True)
        return jsonify({"error": f"{type(e).__name__}: {str(e)[:300]}"}), 500

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

@app.route('/api/tags', methods=['GET'])
def ollama_tags():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        return Response(r.content, status=r.status_code, content_type=r.headers.get('Content-Type', 'application/json'))
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route('/v1/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def catch_all(path):
    """Helpful 404 for unknown endpoints"""
    logger.warning(f"404: Unknown endpoint /{path}")
    return jsonify({
        "error": f"Endpoint /{path} not supported",
        "available_endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/models", "/health"],
        "hint": "Ensure your client uses provider: 'openai' with baseUrl ending in /v1"
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
