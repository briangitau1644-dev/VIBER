#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - ALL ENDPOINTS WORKING"""
import os, sys, json, time, logging, requests, uuid, socket
from flask import Flask, request, jsonify, Response, stream_with_context
from urllib.parse import urlparse

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
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, OpenAI-Beta, X-Requested-With'
    resp.headers['Access-Control-Expose-Headers'] = 'Content-Type'
    return resp

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({"status": "healthy", "model": MODEL, "proxy": "running"}), 200

def extract_user_content(data):
    """Extract content from any format"""
    if 'messages' in data:
        for msg in data['messages']:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if content:
                    return content
    if 'prompt' in data:
        return data['prompt']
    if 'input' in data:
        return data['input']
    return "Please respond with a brief acknowledgment."

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    """Chat completions endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True) or {}
    except:
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
    
    # Add system prompt to ensure responses
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
        if stream:
            def generate():
                with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
                    for line in r.iter_lines():
                        if line:
                            yield f"data: {line.decode('utf-8')}\n\n"
                yield "data: [DONE]\n\n"
            return Response(generate(), mimetype='text/event-stream')
        
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
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
    except Exception as e:
        logger.error(f"Chat error: {e}")
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
        resp.raise_for_status()
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/responses', methods=['POST', 'OPTIONS'])
def responses():
    """Responses API endpoint (OpenAI new format)"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True) or {}
    except:
        data = {}
    
    # Extract input - handles various formats
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
    
    payload = {
        'model': model,
        'prompt': full_prompt,
        'stream': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_output_tokens
        }
    }
    
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
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
    except Exception as e:
        logger.error(f"Responses error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    """List available models"""
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

@app.route('/v1/models/<model_id>', methods=['GET', 'OPTIONS'])
def get_model(model_id):
    """Get specific model"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ollama"
    })

@app.route('/v1', methods=['GET'])
def v1_root():
    """V1 endpoint info"""
    base_url = request.url_root.rstrip('/')
    return jsonify({
        "message": "OpenAI API v1 compatibility layer",
        "available_endpoints": [
            f"{base_url}/v1/chat/completions",
            f"{base_url}/v1/completions", 
            f"{base_url}/v1/responses",
            f"{base_url}/v1/models"
        ],
        "model": MODEL,
        "docs": "Send POST requests to specific endpoints"
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    base_url = request.url_root.rstrip('/')
    return jsonify({
        "service": "Ollama OpenAI Proxy",
        "model": MODEL,
        "base_url": base_url,
        "endpoints": {
            "chat": f"{base_url}/v1/chat/completions",
            "completions": f"{base_url}/v1/completions",
            "responses": f"{base_url}/v1/responses",
            "models": f"{base_url}/v1/models",
            "health": f"{base_url}/health"
        },
        "quick_start": {
            "curl": f"curl -X POST {base_url}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"model\":\"{MODEL}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'"
        }
    })

@app.route('/api/urls', methods=['GET'])
def api_urls():
    """Get all working URLs"""
    base_url = request.url_root.rstrip('/')
    return jsonify({
        "base_url": base_url,
        "endpoints": {
            "chat": f"{base_url}/v1/chat/completions",
            "completions": f"{base_url}/v1/completions",
            "responses": f"{base_url}/v1/responses",
            "models": f"{base_url}/v1/models"
        },
        "test_commands": {
            "chat": f"curl -X POST {base_url}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"model\":\"{MODEL}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}'",
            "responses": f"curl -X POST {base_url}/v1/responses -H 'Content-Type: application/json' -d '{{\"model\":\"{MODEL}\",\"input\":\"Hello\"}}'"
        }
    })

# Catch-all for debugging - shows available endpoints instead of 404
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def catch_all(path):
    """Catch-all that shows available endpoints"""
    if request.method == 'OPTIONS':
        return '', 204
    
    # Don't interfere with v1 endpoints that have handlers
    if path.startswith('v1/'):
        return jsonify({"error": f"Endpoint '/{path}' not found", "available_v1_endpoints": ["chat/completions", "completions", "responses", "models"]}), 404
    
    base_url = request.url_root.rstrip('/')
    return jsonify({
        "message": f"Path '/{path}' not found",
        "available_endpoints": [
            "/",
            "/health",
            "/api/urls",
            "/v1",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/models"
        ],
        "base_url": base_url,
        "try_these": {
            "chat": f"{base_url}/v1/chat/completions",
            "responses": f"{base_url}/v1/responses"
        }
    }), 404

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Ollama Proxy - ALL ENDPOINTS ACTIVE")
    logger.info("=" * 60)
    logger.info(f"📍 Port: {PROXY_PORT}")
    logger.info(f"🤖 Model: {MODEL}")
    logger.info(f"📡 Endpoints:")
    logger.info(f"   POST /v1/chat/completions")
    logger.info(f"   POST /v1/completions")
    logger.info(f"   POST /v1/responses")
    logger.info(f"   GET  /v1/models")
    logger.info(f"   GET  /health")
    logger.info(f"   GET  /api/urls")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True, debug=False)
