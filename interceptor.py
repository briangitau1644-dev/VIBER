#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - FIXED URL HANDLING + DNS + TUNNEL SUPPORT"""
import os, sys, json, time, logging, requests, uuid, socket
from flask import Flask, request, jsonify, Response, stream_with_context
from urllib.parse import urlparse, urljoin

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

# Force IPv4 for better compatibility
try:
    socket.setdefaulttimeout(30)
    # Pre-resolve Ollama host
    socket.gethostbyname(OLLAMA_HOST)
except:
    pass

# CORS Middleware with proper headers for tunnel
@app.after_request
def cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, OpenAI-Beta, X-Requested-With, X-Forwarded-For, X-Forwarded-Proto, CF-*'
    resp.headers['Access-Control-Expose-Headers'] = 'Content-Type, Authorization, X-Request-Id'
    resp.headers['X-Proxy-Version'] = '2.0'
    return resp

@app.before_request
def log_request():
    """Log all incoming requests for debugging"""
    logger.info(f"→ {request.method} {request.path} from {request.remote_addr}")
    if request.method == 'POST' and request.path == '/v1/chat/completions':
        try:
            data = request.get_json(silent=True)
            if data:
                msg_preview = str(data.get('messages', [{}])[0].get('content', ''))[:50]
                logger.info(f"  Content preview: {msg_preview}...")
        except:
            pass

@app.route('/health', methods=['GET', 'HEAD', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    if request.method == 'HEAD':
        return '', 200
    
    try:
        # Test Ollama connectivity
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return jsonify({
                "status": "healthy",
                "model": MODEL,
                "ollama": "connected",
                "proxy_url": f"http://0.0.0.0:{PROXY_PORT}",
                "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/models"]
            }), 200
        return jsonify({"status": "degraded", "error": "Ollama response non-200"}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503

@app.route('/api/debug/dns', methods=['GET'])
def debug_dns():
    """Debug endpoint to check DNS resolution"""
    import dns.resolver
    results = {}
    
    # Get the actual request host
    host = request.headers.get('Host', 'unknown')
    results['request_host'] = host
    results['request_url'] = request.url
    
    # Try to resolve common domains
    for domain in [host.split(':')[0], 'localhost', '127.0.0.1', OLLAMA_HOST]:
        if domain and domain != 'unknown':
            try:
                answers = socket.gethostbyname(domain)
                results[f'dns_{domain}'] = answers
            except Exception as e:
                results[f'dns_{domain}'] = str(e)
    
    return jsonify(results)

def extract_user_content(data):
    """Aggressive content extraction - guarantees non-empty response"""
    # 1. Direct messages array
    if 'messages' in data and isinstance(data['messages'], list):
        for msg in data['messages']:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif item.get('type') == 'input_text':
                                text_parts.append(item.get('text', ''))
                    if text_parts:
                        combined = ' '.join(text_parts)
                        if combined.strip():
                            return combined
    
    # 2. Legacy completions
    if 'prompt' in data:
        prompt = data['prompt']
        if isinstance(prompt, str) and prompt.strip():
            return prompt
        if isinstance(prompt, list) and prompt:
            return str(prompt[0])
    
    # 3. Responses API
    if 'input' in data:
        inp = data['input']
        if isinstance(inp, str) and inp.strip():
            return inp
        if isinstance(inp, dict):
            content = inp.get('content', '')
            if content and isinstance(content, str):
                return content
    
    # 4. Force a default that will get a response
    logger.warning("No valid input found, using default prompt")
    return "Please respond with a brief acknowledgment like 'OK' or 'Hello'."

def force_non_empty_response(content, original_prompt):
    """Ensure response isn't empty - retry with different prompt if needed"""
    if content and content.strip():
        return content
    
    logger.warning(f"Empty response detected for prompt: '{original_prompt[:50]}...'")
    
    # Retry with explicit instructions
    retry_prompts = [
        "Say exactly: 'OK'",
        "Respond with a single word: Hello",
        "Reply with: I am ready"
    ]
    
    for retry_prompt in retry_prompts:
        try:
            retry_payload = {
                'model': MODEL,
                'messages': [{'role': 'user', 'content': retry_prompt}],
                'stream': False,
                'options': {'temperature': 0.5, 'num_predict': 20}
            }
            retry_resp = requests.post(f"{OLLAMA_URL}/api/chat", json=retry_payload, timeout=30)
            if retry_resp.status_code == 200:
                retry_data = retry_resp.json()
                retry_content = retry_data.get('message', {}).get('content', '')
                if retry_content and retry_content.strip():
                    logger.info(f"Retry successful with prompt: '{retry_prompt}'")
                    return retry_content
        except Exception as e:
            logger.warning(f"Retry failed: {e}")
            continue
    
    return "I am ready to help."

@app.route('/v1/chat/completions', methods=['GET', 'POST', 'PUT', 'OPTIONS'])
def chat_completions():
    """Primary chat endpoint with robust URL and response handling"""
    if request.method == 'OPTIONS':
        return '', 204
    
    if request.method in ['GET', 'PUT']:
        return jsonify({
            "endpoint": "/v1/chat/completions",
            "method": request.method,
            "supported_methods": ["POST", "OPTIONS"],
            "base_url": f"{request.url_root}v1",
            "model": MODEL,
            "example": {
                "curl": f"curl -X POST {request.url_root}v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer roo' -d '{{\"model\":\"{MODEL}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'"
            }
        }), 200
    
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        data = {}
    
    # Extract content
    prompt = extract_user_content(data)
    messages = data.get('messages', [])
    tools = data.get('tools', [])
    stream = data.get('stream', False)
    max_tokens = data.get('max_tokens', data.get('max_completion_tokens', 500))
    temperature = data.get('temperature', 0.3)
    
    # Build Ollama messages
    ollama_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join([c.get('text', '') for c in content if isinstance(c, dict)])
            if content:  # Only add non-empty messages
                ollama_messages.append({'role': role, 'content': str(content)})
    
    if not ollama_messages:
        ollama_messages = [{'role': 'user', 'content': prompt}]
    
    # Ensure system message for better responses
    has_system = any(m.get('role') == 'system' for m in ollama_messages)
    if not has_system:
        ollama_messages.insert(0, {
            'role': 'system',
            'content': 'You are a helpful assistant. Always provide complete, non-empty responses. Never respond with just whitespace or empty strings.'
        })
    
    payload = {
        'model': MODEL,
        'messages': ollama_messages,
        'stream': stream,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
            'num_ctx': 8192,
            'stop': ['<end_of_turn>', '<eos>']
        }
    }
    
    # Add tools if present
    if tools:
        payload['tools'] = tools
        logger.info(f"🔧 Tools: {len(tools)} functions")
    
    logger.info(f"📤 Request to Ollama: {len(ollama_messages)} messages, stream={stream}, tools={bool(tools)}")
    
    try:
        if stream:
            def generate():
                try:
                    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as ollama_resp:
                        for line in ollama_resp.iter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line.decode('utf-8'))
                                    sse_chunk = {
                                        'id': f"chatcmpl-{uuid.uuid4().hex[:8]}",
                                        'object': 'chat.completion.chunk',
                                        'created': int(time.time()),
                                        'model': MODEL,
                                        'choices': [{
                                            'index': 0,
                                            'delta': {},
                                            'finish_reason': None
                                        }]
                                    }
                                    
                                    if 'message' in chunk:
                                        msg = chunk['message']
                                        if 'content' in msg and msg['content']:
                                            sse_chunk['choices'][0]['delta']['content'] = msg['content']
                                        if 'tool_calls' in msg:
                                            sse_chunk['choices'][0]['delta']['tool_calls'] = msg['tool_calls']
                                    
                                    if chunk.get('done'):
                                        sse_chunk['choices'][0]['finish_reason'] = 'stop'
                                    
                                    yield f"data: {json.dumps(sse_chunk)}\n\n"
                                except json.JSONDecodeError:
                                    continue
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_chunk = {'error': str(e)}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
        else:
            # Non-streaming
            try:
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 404:
                    raise ValueError("Chat API not available")
                resp.raise_for_status()
                ollama_res = resp.json()
                
                # Extract and validate content
                content = ollama_res.get('message', {}).get('content', '')
                
                # Force non-empty response if needed
                if not content or not content.strip():
                    content = force_non_empty_response(content, prompt)
                
                response_message = {
                    "role": "assistant",
                    "content": content
                }
                
                # Handle tool calls if present
                tool_calls = ollama_res.get('message', {}).get('tool_calls')
                if tool_calls:
                    response_message["tool_calls"] = tool_calls
                    response_message["content"] = None
                    finish_reason = "tool_calls"
                else:
                    finish_reason = "stop"
                
                return jsonify({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [{
                        "index": 0,
                        "message": response_message,
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": ollama_res.get('prompt_eval_count', 0),
                        "completion_tokens": ollama_res.get('eval_count', len(content)),
                        "total_tokens": ollama_res.get('prompt_eval_count', 0) + ollama_res.get('eval_count', len(content))
                    }
                })
                
            except Exception as chat_error:
                # Fallback to generate API
                logger.warning(f"Chat API failed: {chat_error}, using generate API")
                gen_payload = {
                    'model': MODEL,
                    'prompt': f"System: You are a helpful assistant. Always respond.\n\nUser: {prompt}\nAssistant:",
                    'stream': False,
                    'options': payload['options']
                }
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                gen_res = resp.json()
                
                content = gen_res.get('response', '')
                if not content or not content.strip():
                    content = "I'm ready to help. What would you like to know?"
                
                return jsonify({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
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
                
    except requests.exceptions.Timeout:
        logger.error("Ollama timeout")
        return jsonify({"error": "Request timeout", "type": "timeout_error"}), 504
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return jsonify({"error": f"Cannot connect to Ollama at {OLLAMA_URL}", "type": "connection_error"}), 503
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": str(e), "type": "internal_error"}), 500

@app.route('/v1/completions', methods=['POST', 'OPTIONS'])
def completions():
    """Legacy completions endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True) or {}
    except:
        data = {}
    
    prompt = data.get('prompt', '')
    if isinstance(prompt, list):
        prompt = ' '.join(prompt)
    
    if not prompt:
        prompt = "Say hello"
    
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.3)
    
    payload = {
        'model': MODEL,
        'prompt': f"System: Always respond with something.\n\nUser: {prompt}\nAssistant:",
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
        
        content = result.get('response', '')
        if not content or not content.strip():
            content = "OK"
        
        return jsonify({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
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
            "owned_by": "ollama",
            "permission": []
        }]
    })

@app.route('/v1/models/<model_id>', methods=['GET', 'OPTIONS'])
def get_model(model_id):
    """Get specific model info"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ollama"
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with full service info and working URLs"""
    base_url = request.url_root.rstrip('/')
    
    return jsonify({
        "service": "Ollama OpenAI-Compatible Proxy",
        "version": "2.0.0",
        "model": MODEL,
        "base_url": base_url,
        "endpoints": {
            "chat": f"{base_url}/v1/chat/completions",
            "completions": f"{base_url}/v1/completions",
            "models": f"{base_url}/v1/models",
            "health": f"{base_url}/health",
            "debug_dns": f"{base_url}/api/debug/dns"
        },
        "quick_start": {
            "curl_example": f"curl -X POST {base_url}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"model\":\"{MODEL}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'",
            "python_example": f"""
import openai
client = openai.OpenAI(
    base_url="{base_url}/v1",
    api_key="roo"
)
response = client.chat.completions.create(
    model="{MODEL}",
    messages=[{{"role": "user", "content": "Hello"}}]
)
print(response.choices[0].message.content)
"""
        },
        "features": {
            "tool_calls": True,
            "streaming": True,
            "function_calling": True
        }
    })

@app.route('/api/urls', methods=['GET'])
def get_urls():
    """Helper endpoint to get all working URLs"""
    base_url = request.url_root.rstrip('/')
    
    # Get tunnel URL from environment if available
    tunnel_url = os.getenv('CLOUDFLARE_URL', '')
    
    return jsonify({
        "local_access": {
            "proxy": f"http://localhost:{PROXY_PORT}",
            "ollama": OLLAMA_URL,
            "chat_endpoint": f"http://localhost:{PROXY_PORT}/v1/chat/completions"
        },
        "tunnel_access": {
            "url": tunnel_url,
            "chat_endpoint": f"{tunnel_url}/v1/chat/completions" if tunnel_url else None
        },
        "current_request": {
            "base_url": base_url,
            "full_url": request.url,
            "headers": dict(request.headers)
        },
        "dns_info": {
            "proxy_host": socket.gethostbyname('localhost'),
            "ollama_host": socket.gethostbyname(OLLAMA_HOST) if OLLAMA_HOST != '127.0.0.1' else '127.0.0.1'
        }
    })

# Catch-all for debugging
@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def catch_all(path):
    """Catch-all to help debug routing"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "message": f"Path '/{path}' not found",
        "available_endpoints": [
            "/",
            "/health",
            "/api/urls",
            "/api/debug/dns",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/models"
        ],
        "current_url": request.url,
        "base_url": request.url_root
    }), 404

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🚀 Ollama Proxy Server v2.0 - URL FIXED")
    logger.info("=" * 70)
    logger.info(f"📍 Proxy URL: http://0.0.0.0:{PROXY_PORT}")
    logger.info(f"🤖 Model: {MODEL}")
    logger.info(f"🔗 Ollama Backend: {OLLAMA_URL}")
    logger.info(f"🔧 Tool Calls: ENABLED")
    logger.info(f"📡 Streaming: ENABLED")
    logger.info(f"🌐 CORS: ENABLED for tunnels")
    logger.info("=" * 70)
    logger.info("📋 Quick Test:")
    logger.info(f"   curl http://localhost:{PROXY_PORT}/health")
    logger.info(f"   curl http://localhost:{PROXY_PORT}/api/urls")
    logger.info("=" * 70)
    
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True, debug=False)
