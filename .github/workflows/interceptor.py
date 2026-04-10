#!/usr/bin/env python3
"""
interceptor.py - OpenAI-compatible proxy for Ollama with tool call interception
Handles CORS, request transformation, and deterministic generation settings.
"""

import os
import sys
import json
import time
import logging
import requests
from flask import Flask, request, jsonify, Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration from environment
PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = os.getenv('MODEL', 'gemma3:4b')
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
CTX_SIZE = int(os.getenv('CTX_SIZE', '8192'))

OLLAMA_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# CORS headers helper
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.before_request
def log_request():
    logger.info(f"{request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.status_code == 200:
            return jsonify({"status": "healthy", "ollama": "connected"}), 200
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
    return jsonify({"status": "degraded", "error": str(e)}), 503

@app.route('/api/tags', methods=['GET'])
def list_models():
    """Proxy Ollama's model list endpoint"""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=30)
        return Response(resp.content, status=resp.status_code, content_type=resp.headers.get('Content-Type', 'application/json'))
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return jsonify({"error": str(e)}), 502

@app.route('/v1/models', methods=['GET'])
def openai_list_models():
    """OpenAI-compatible model listing"""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=30)
        data = resp.json()
        models = []
        for m in data.get('models', []):
            models.append({
                "id": m.get('name', MODEL),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama"
            })
        # Ensure our target model is listed
        if not any(m['id'] == MODEL for m in models):
            models.append({
                "id": MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama"
            })
        return jsonify({"object": "list", "data": models})
    except Exception as e:
        logger.error(f"Failed to list OpenAI models: {e}")
        return jsonify({"error": str(e)}), 502

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint with tool interception"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        req_data = request.get_json(force=True)
    except Exception as e:
        logger.error(f"Invalid JSON: {e}")
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Extract and normalize parameters
    model = req_data.get('model', MODEL)
    messages = req_data.get('messages', [])
    stream = req_data.get('stream', False)
    max_tokens = req_data.get('max_tokens', req_data.get('max_completion_tokens', 2048))
    temperature = req_data.get('temperature', TEMPERATURE)
    tools = req_data.get('tools', None)
    
    # Transform OpenAI messages to Ollama format
    ollama_messages = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        # Handle tool calls if present
        if role == 'assistant' and 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                ollama_messages.append({
                    'role': 'assistant',
                    'content': f"[Tool call: {tc.get('function', {}).get('name')}]"
                })
        elif role == 'tool':
            # Convert tool response to user message for Ollama
            ollama_messages.append({
                'role': 'user',
                'content': f"[Tool response]: {content}"
            })
        else:
            ollama_messages.append({'role': role, 'content': content})
    
    # Build Ollama request
    ollama_payload = {
        'model': model,
        'messages': ollama_messages,
        'stream': stream,
        'options': {
            'temperature': temperature,
            'num_ctx': CTX_SIZE,
            'num_predict': max_tokens
        }
    }
    
    # Tool interception logic: if tools requested, we may intercept certain patterns
    intercepted_response = None
    if tools:
        intercepted_response = _intercept_tool_calls(messages, tools)
    
    if intercepted_response:
        logger.info("🎯 Tool call intercepted - returning synthetic response")
        result = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": intercepted_response},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(intercepted_response.split()), "total_tokens": 0}
        }
        return jsonify(result)
    
    try:
        # Forward to Ollama
        ollama_url = f"{OLLAMA_BASE}/v1/chat/completions" if stream else f"{OLLAMA_BASE}/api/chat"
        
        if stream:
            # Streaming: proxy Ollama's streaming response
            def generate():
                try:
                    with requests.post(ollama_url, json=ollama_payload, stream=True, timeout=120) as resp:
                        for line in resp.iter_lines():
                            if line:
                                yield line + b'\n'
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f'data: {json.dumps({"error": str(e)})}\n\n'.encode()
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            # Non-streaming: transform Ollama response to OpenAI format
            resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=ollama_payload, timeout=180)
            resp.raise_for_status()
            ollama_result = resp.json()
            
            content = ollama_result.get('message', {}).get('content', '')
            
            openai_result = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": ollama_result.get('done_reason', 'stop')
                }],
                "usage": {
                    "prompt_tokens": ollama_result.get('prompt_eval_count', 0),
                    "completion_tokens": ollama_result.get('eval_count', 0),
                    "total_tokens": ollama_result.get('prompt_eval_count', 0) + ollama_result.get('eval_count', 0)
                }
            }
            return jsonify(openai_result)
            
    except requests.exceptions.Timeout:
        logger.error("Ollama request timeout")
        return jsonify({"error": "Gateway timeout"}), 504
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama")
        return jsonify({"error": "Ollama unavailable"}), 502
    except Exception as e:
        logger.error(f"Proxy error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def _intercept_tool_calls(messages, tools):
    """
    Intercept specific tool call patterns for deterministic responses.
    Returns a string response if intercepted, None otherwise.
    """
    if not messages:
        return None
    
    last_msg = messages[-1].get('content', '').lower() if isinstance(messages[-1], dict) else str(messages[-1]).lower()
    
    # Example intercept patterns for senior engineer tasks
    intercept_patterns = {
        'create file': 'I can help with that. To create a file in this environment, use: `echo "content" > filename` or Python `open("filename", "w").write("content")`. What file would you like to create?',
        'run command': 'For safety, commands should be reviewed first. What command would you like to run? I can help validate it.',
        'read file': 'To read a file: `cat filename` or in Python `open("filename").read()`. Which file?',
        'install package': 'Use `pip install package-name` for Python or `sudo apt install package` for system packages. Which package?',
    }
    
    for pattern, response in intercept_patterns.items():
        if pattern in last_msg:
            return response
    
    return None

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API info"""
    return jsonify({
        "service": "Ollama Interceptor Proxy",
        "model": MODEL,
        "endpoints": [
            "/health",
            "/api/tags",
            "/v1/models", 
            "/v1/chat/completions"
        ],
        "cors": "enabled"
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "path": request.path}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info(f"🚀 Starting interceptor proxy on port {PROXY_PORT}")
    logger.info(f"   → Ollama: {OLLAMA_HOST}:{OLLAMA_PORT}")
    logger.info(f"   → Model: {MODEL}")
    logger.info(f"   → Context: {CTX_SIZE}, Temperature: {TEMPERATURE}")
    
    # Run with threaded=True for better concurrency
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True)
