#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - FIXED PATHS + OLLAMA COMPATIBILITY"""
import os, sys, json, time, logging, requests, uuid, socket
from flask import Flask, request, jsonify, Response, stream_with_context

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

# Verify Ollama endpoints are available
def check_ollama_endpoints():
    """Check which Ollama endpoints are available"""
    endpoints = {
        'chat': f"{OLLAMA_URL}/api/chat",
        'generate': f"{OLLAMA_URL}/api/generate",
        'tags': f"{OLLAMA_URL}/api/tags"
    }
    
    available = []
    for name, url in endpoints.items():
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code in [200, 405]:  # 405 means endpoint exists but method not allowed
                available.append(name)
                logger.info(f"✅ Ollama {name} endpoint available at {url}")
            else:
                logger.warning(f"⚠️ Ollama {name} endpoint returned {resp.status_code}")
        except:
            logger.warning(f"❌ Ollama {name} endpoint not reachable at {url}")
    
    return available

# Run check on startup
OLLAMA_ENDPOINTS = check_ollama_endpoints()

# CORS Middleware
@app.after_request
def cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, OpenAI-Beta, X-Requested-With'
    resp.headers['Access-Control-Expose-Headers'] = 'Content-Type'
    return resp

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        "status": "healthy",
        "model": MODEL,
        "proxy": "running",
        "ollama_endpoints": OLLAMA_ENDPOINTS,
        "ollama_url": OLLAMA_URL
    }), 200

def extract_user_content(data):
    """Extract content from any format"""
    if 'messages' in data and data['messages']:
        for msg in data['messages']:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if content:
                    return content
    if 'prompt' in data:
        return data['prompt']
    if 'input' in data:
        inp = data['input']
        if isinstance(inp, str):
            return inp
        if isinstance(inp, dict):
            return inp.get('content', '')
    return "Please respond with a brief acknowledgment."

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    """Chat completions endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        data = {}
    
    prompt = extract_user_content(data)
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    max_tokens = data.get('max_tokens', 500)
    temperature = data.get('temperature', 0.3)
    
    # Build Ollama messages
    ollama_messages = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get('content'):
            ollama_messages.append({
                'role': msg.get('role', 'user'),
                'content': str(msg.get('content', ''))
            })
    
    if not ollama_messages:
        ollama_messages = [{'role': 'user', 'content': prompt}]
    
    # Add system prompt
    if not any(m.get('role') == 'system' for m in ollama_messages):
        ollama_messages.insert(0, {
            'role': 'system',
            'content': 'You are a helpful assistant. Always provide complete, non-empty responses.'
        })
    
    payload = {
        'model': MODEL,
        'messages': ollama_messages,
        'stream': stream,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
            'num_ctx': 8192
        }
    }
    
    logger.info(f"Chat request: {len(ollama_messages)} messages, stream={stream}")
    
    try:
        # Try chat endpoint first
        if 'chat' in OLLAMA_ENDPOINTS:
            resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
            if resp.status_code == 200:
                result = resp.json()
                content = result.get('message', {}).get('content', '')
                if not content:
                    content = "I'm ready to help."
                
                return jsonify({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": result.get('prompt_eval_count', 0),
                        "completion_tokens": result.get('eval_count', 0),
                        "total_tokens": result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                    }
                })
        
        # Fallback to generate endpoint
        logger.info("Using generate endpoint fallback")
        gen_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in ollama_messages])
        gen_payload = {
            'model': MODEL,
            'prompt': gen_prompt + "\nassistant:",
            'stream': False,
            'options': payload['options']
        }
        
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            result = resp.json()
            content = result.get('response', '')
            if not content:
                content = "I'm ready to help."
            
            return jsonify({
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(content),
                    "total_tokens": 0
                }
            })
        else:
            return jsonify({"error": f"Ollama returned {resp.status_code}", "details": resp.text}), 500
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/responses', methods=['POST', 'OPTIONS'])
def responses():
    """Responses API endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        data = {}
    
    # Extract input
    input_content = data.get('input', '')
    if isinstance(input_content, dict):
        input_content = input_content.get('content', '')
    elif isinstance(input_content, list):
        input_content = ' '.join([str(i) for i in input_content if i])
    
    if not input_content:
        input_content = "Say hello"
    
    model = data.get('model', MODEL)
    instructions = data.get('instructions', '')
    max_output_tokens = data.get('max_output_tokens', 500)
    temperature = data.get('temperature', 0.3)
    
    # Build prompt
    if instructions:
        full_prompt = f"{instructions}\n\nUser: {input_content}\nAssistant:"
    else:
        full_prompt = f"User: {input_content}\nAssistant:"
    
    logger.info(f"Responses request: input='{input_content[:50]}...'")
    
    try:
        # Try generate endpoint (more compatible)
        gen_payload = {
            'model': model,
            'prompt': full_prompt,
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_output_tokens
            }
        }
        
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
        
        if resp.status_code == 200:
            result = resp.json()
            content = result.get('response', '')
            if not content:
                content = "OK"
            
            return jsonify({
                "id": f"resp_{uuid.uuid4().hex[:8]}",
                "object": "response",
                "created": int(time.time()),
                "model": model,
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}]
                }],
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": len(content),
                    "total_tokens": 0
                }
            })
        else:
            # Try chat endpoint as fallback
            chat_payload = {
                'model': model,
                'messages': [
                    {'role': 'user', 'content': input_content}
                ],
                'options': gen_payload['options']
            }
            resp = requests.post(f"{OLLAMA_URL}/api/chat", json=chat_payload, timeout=OLLAMA_TIMEOUT)
            if resp.status_code == 200:
                result = resp.json()
                content = result.get('message', {}).get('content', 'OK')
                return jsonify({
                    "id": f"resp_{uuid.uuid4().hex[:8]}",
                    "object": "response",
                    "created": int(time.time()),
                    "model": model,
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}]
                    }],
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": len(content),
                        "total_tokens": 0
                    }
                })
            else:
                return jsonify({"error": f"Ollama returned {resp.status_code}", "details": resp.text}), 500
                
    except Exception as e:
        logger.error(f"Responses error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/completions', methods=['POST', 'OPTIONS'])
def completions():
    """Legacy completions endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True) or {}
    except:
        data = {}
    
    prompt = data.get('prompt', 'Say hello')
    if isinstance(prompt, list):
        prompt = ' '.join(prompt)
    
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.3)
    
    payload = {
        'model': MODEL,
        'prompt': f"User: {prompt}\nAssistant:",
        'stream': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens
        }
    }
    
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        if resp.status_code == 200:
            result = resp.json()
            content = result.get('response', 'OK')
            
            return jsonify({
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{
                    "text": content,
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(content),
                    "total_tokens": 0
                }
            })
        else:
            return jsonify({"error": f"Ollama returned {resp.status_code}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "object": "list",
        "data": [{
            "id": MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ollama"
        }]
    })

@app.route('/', methods=['GET'])
def root():
    base_url = request.url_root.rstrip('/')
    return jsonify({
        "service": "Ollama OpenAI Proxy",
        "model": MODEL,
        "base_url": base_url,
        "important": "DO NOT add /v1 twice in your base URL!",
        "endpoints": {
            "chat": f"{base_url}/v1/chat/completions",
            "responses": f"{base_url}/v1/responses",
            "completions": f"{base_url}/v1/completions",
            "models": f"{base_url}/v1/models"
        },
        "config_for_roo": {
            "provider": "openai",
            "baseUrl": f"{base_url}/v1",
            "model": MODEL,
            "apiKey": "roo"
        }
    })

@app.route('/api/debug', methods=['GET'])
def debug():
    """Debug endpoint to test Ollama connectivity"""
    results = {
        "ollama_url": OLLAMA_URL,
        "endpoints_checked": OLLAMA_ENDPOINTS,
        "model": MODEL
    }
    
    # Test each endpoint
    for endpoint in ['chat', 'generate', 'tags']:
        url = f"{OLLAMA_URL}/api/{endpoint}"
        try:
            if endpoint == 'tags':
                resp = requests.get(url, timeout=2)
            else:
                resp = requests.post(url, json={'model': MODEL}, timeout=2)
            results[f"test_{endpoint}"] = {
                "status": resp.status_code,
                "working": resp.status_code < 500
            }
        except Exception as e:
            results[f"test_{endpoint}"] = {"error": str(e), "working": False}
    
    return jsonify(results)

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Ollama Proxy - PATH FIXED")
    logger.info("=" * 60)
    logger.info(f"📍 Proxy URL: http://0.0.0.0:{PROXY_PORT}")
    logger.info(f"🤖 Model: {MODEL}")
    logger.info(f"🔗 Ollama: {OLLAMA_URL}")
    logger.info(f"📡 Available Ollama endpoints: {OLLAMA_ENDPOINTS}")
    logger.info("=" * 60)
    logger.info("✅ CORRECT CONFIGURATION FOR ROO:")
    logger.info(f"   baseUrl: http://localhost:{PROXY_PORT}/v1")
    logger.info(f"   (DO NOT add /v1 again in your requests)")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True, debug=False)
