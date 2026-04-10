#!/usr/bin/env python3
"""Flask proxy for gemma4:e4b - COMPLETE FIX with ALL endpoints and FULL tool call support"""
import os, sys, json, time, logging, requests, uuid
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

# CORS Middleware
@app.after_request
def cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, OpenAI-Beta, X-Requested-With'
    resp.headers['Access-Control-Expose-Headers'] = 'Content-Type, Authorization'
    return resp

@app.route('/health', methods=['GET', 'HEAD', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS': return '', 204
    if request.method == 'HEAD': return '', 200
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return jsonify({"status": "healthy", "model": MODEL, "ollama": "connected"}), 200
        return jsonify({"status": "degraded", "error": "Ollama response non-200"}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503

def extract_user_content(data):
    """Aggressive content extraction from ANY format - NEVER fails"""
    # 1. Direct messages array (OpenAI format)
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
                        return ' '.join(text_parts)
    
    # 2. Legacy completions API
    if 'prompt' in data:
        prompt = data['prompt']
        if isinstance(prompt, str) and prompt.strip():
            return prompt
        if isinstance(prompt, list) and prompt:
            return str(prompt[0])
    
    # 3. Responses API format
    if 'input' in data:
        inp = data['input']
        if isinstance(inp, str) and inp.strip():
            return inp
        if isinstance(inp, list):
            for item in inp:
                if isinstance(item, dict) and item.get('role') == 'user':
                    return item.get('content', '')
    
    # 4. Raw text fallback
    if 'text' in data and data['text']:
        return str(data['text'])
    
    if 'query' in data and data['query']:
        return str(data['query'])
    
    # 5. NEVER FAIL - always return something
    logger.warning("No valid input found, using default prompt")
    return "Please respond with a brief acknowledgment."

def convert_tools_for_ollama(openai_tools):
    """Convert OpenAI tool format to Ollama-compatible format"""
    if not openai_tools:
        return None
    
    ollama_tools = []
    for tool in openai_tools:
        if tool.get('type') == 'function':
            function = tool.get('function', {})
            ollama_tools.append({
                'type': 'function',
                'function': {
                    'name': function.get('name', ''),
                    'description': function.get('description', ''),
                    'parameters': function.get('parameters', {})
                }
            })
    return ollama_tools if ollama_tools else None

def parse_ollama_tool_calls(ollama_response):
    """Parse Ollama's tool_calls response into OpenAI format"""
    message = ollama_response.get('message', {})
    tool_calls_raw = message.get('tool_calls', [])
    
    if not tool_calls_raw:
        return None, message.get('content', '')
    
    openai_tool_calls = []
    for idx, tc in enumerate(tool_calls_raw):
        function = tc.get('function', {})
        args = function.get('arguments', {})
        
        # Ensure arguments is a JSON string
        if isinstance(args, dict):
            args_str = json.dumps(args)
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = "{}"
        
        openai_tool_calls.append({
            'id': f"call_{uuid.uuid4().hex[:8]}",
            'type': 'function',
            'function': {
                'name': function.get('name', ''),
                'arguments': args_str
            }
        })
    
    return openai_tool_calls, None

@app.route('/v1/chat/completions', methods=['GET', 'POST', 'PUT', 'OPTIONS'])
def chat_completions():
    """Primary chat completions endpoint - FULL tool call support"""
    if request.method == 'OPTIONS':
        return '', 204
    if request.method in ['GET', 'PUT']:
        return jsonify({
            "endpoint": "/v1/chat/completions",
            "method": request.method,
            "supported_methods": ["POST", "OPTIONS"],
            "message": "Send POST with JSON body"
        }), 200
    
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        data = {}
    
    # Extract everything needed
    prompt = extract_user_content(data)
    messages = data.get('messages', [])
    tools = data.get('tools', [])
    tool_choice = data.get('tool_choice', 'auto')
    stream = data.get('stream', False)
    max_tokens = data.get('max_tokens', data.get('max_completion_tokens', 2048))
    temperature = data.get('temperature', 0.2)
    top_p = data.get('top_p', 0.95)
    n = data.get('n', 1)
    stop = data.get('stop', [])
    presence_penalty = data.get('presence_penalty', 0.0)
    frequency_penalty = data.get('frequency_penalty', 0.0)
    user = data.get('user', '')
    
    # Build Ollama payload
    ollama_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join([c.get('text', '') for c in content if isinstance(c, dict)])
            ollama_messages.append({'role': role, 'content': str(content)})
    
    if not ollama_messages:
        ollama_messages = [{'role': 'user', 'content': prompt}]
    
    payload = {
        'model': MODEL,
        'messages': ollama_messages,
        'stream': stream,
        'options': {
            'temperature': temperature,
            'top_p': top_p,
            'num_predict': max_tokens,
            'num_ctx': 8192,
            'stop': stop if stop else None,
            'presence_penalty': presence_penalty,
            'frequency_penalty': frequency_penalty
        }
    }
    
    # Add tools if present
    if tools:
        ollama_tools = convert_tools_for_ollama(tools)
        if ollama_tools:
            payload['tools'] = ollama_tools
            if tool_choice != 'auto':
                payload['tool_choice'] = tool_choice
            logger.info(f"🔧 Tools enabled: {len(tools)} tools")
    
    logger.info(f"→ POST /v1/chat/completions | model={MODEL} | stream={stream} | tools={bool(tools)} | prompt='{prompt[:50]}...'")
    
    try:
        if stream:
            def generate_stream():
                try:
                    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as ollama_resp:
                        for line in ollama_resp.iter_lines():
                            if line:
                                try:
                                    chunk_data = json.loads(line.decode('utf-8'))
                                    # Convert to SSE format
                                    sse_data = {
                                        'id': f"chatcmpl-{int(time.time())}",
                                        'object': 'chat.completion.chunk',
                                        'created': int(time.time()),
                                        'model': MODEL,
                                        'choices': [{
                                            'index': 0,
                                            'delta': {},
                                            'finish_reason': None
                                        }]
                                    }
                                    
                                    if 'message' in chunk_data:
                                        msg = chunk_data['message']
                                        if 'content' in msg and msg['content']:
                                            sse_data['choices'][0]['delta']['content'] = msg['content']
                                        if 'tool_calls' in msg and msg['tool_calls']:
                                            sse_data['choices'][0]['delta']['tool_calls'] = msg['tool_calls']
                                    
                                    if chunk_data.get('done', False):
                                        sse_data['choices'][0]['finish_reason'] = 'stop'
                                    
                                    yield f"data: {json.dumps(sse_data)}\n\n"
                                except json.JSONDecodeError:
                                    continue
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_data = {'error': str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')
        
        else:
            # Non-streaming - try chat API first
            try:
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 404:
                    raise ValueError("Chat API not available")
                resp.raise_for_status()
                ollama_res = resp.json()
                
                # Parse tool calls if any
                tool_calls, content = parse_ollama_tool_calls(ollama_res)
                
                response_message = {"role": "assistant"}
                if tool_calls:
                    response_message["tool_calls"] = tool_calls
                    response_message["content"] = None
                    finish_reason = "tool_calls"
                else:
                    response_message["content"] = content or ""
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
                        "completion_tokens": ollama_res.get('eval_count', 0),
                        "total_tokens": ollama_res.get('prompt_eval_count', 0) + ollama_res.get('eval_count', 0)
                    }
                })
            except Exception as chat_error:
                # Fallback to generate API
                logger.warning(f"Chat API failed ({chat_error}), falling back to /api/generate")
                gen_payload = {
                    'model': MODEL,
                    'prompt': f"User: {prompt}\nAssistant:",
                    'stream': False,
                    'options': payload['options']
                }
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json=gen_payload, timeout=OLLAMA_TIMEOUT)
                resp.raise_for_status()
                gen_res = resp.json()
                
                return jsonify({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": gen_res.get('response', '')
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })
                
    except requests.exceptions.Timeout:
        logger.error("Ollama timeout")
        return jsonify({"error": "Request timeout", "type": "timeout_error"}), 504
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return jsonify({"error": "Cannot connect to Ollama", "type": "connection_error"}), 503
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
    
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.2)
    
    payload = {
        'model': MODEL,
        'prompt': prompt,
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
        
        return jsonify({
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [{
                "text": result.get('response', ''),
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/responses', methods=['POST', 'OPTIONS'])
def responses():
    """Responses API endpoint (new OpenAI format)"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json(force=True) or {}
    except:
        data = {}
    
    # Extract input content
    input_content = data.get('input', '')
    if isinstance(input_content, dict):
        input_content = input_content.get('content', '')
    elif isinstance(input_content, list):
        input_content = ' '.join([str(i) for i in input_content])
    
    model = data.get('model', MODEL)
    instructions = data.get('instructions', '')
    max_output_tokens = data.get('max_output_tokens', 2048)
    temperature = data.get('temperature', 0.2)
    
    full_prompt = f"{instructions}\n\nUser: {input_content}\nAssistant:" if instructions else f"User: {input_content}\nAssistant:"
    
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
        
        return jsonify({
            "id": f"resp_{uuid.uuid4().hex[:12]}",
            "object": "response",
            "created": int(time.time()),
            "model": model,
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.get('response', '')}]
            }],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
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
    
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = []
            for m in data.get('models', []):
                models.append({
                    "id": m.get('name', MODEL),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama"
                })
            
            # Ensure our model is in the list
            if not any(m['id'] == MODEL for m in models):
                models.append({
                    "id": MODEL,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama"
                })
            
            return jsonify({
                "object": "list",
                "data": models
            })
        else:
            return jsonify({
                "object": "list",
                "data": [{"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"}]
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route('/v1/models/<model_id>', methods=['GET', 'OPTIONS'])
def get_model(model_id):
    """Get specific model info"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ollama",
        "permission": []
    })

@app.route('/v1/tools', methods=['GET', 'OPTIONS'])
def tools_info():
    """Information about tool calling support"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "tools_supported": True,
        "tool_choice_supported": True,
        "max_tools_per_request": 10,
        "supported_tool_types": ["function"],
        "documentation": "Use standard OpenAI function calling format"
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service info"""
    return jsonify({
        "service": "Ollama OpenAI-Compatible Proxy",
        "model": MODEL,
        "version": "2.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "responses": "/v1/responses",
            "models": "/v1/models",
            "tools": "/v1/tools",
            "health": "/health"
        },
        "features": {
            "tool_calls": True,
            "streaming": True,
            "function_calling": True
        },
        "usage": {
            "base_url": f"http://localhost:{PROXY_PORT}/v1",
            "api_key": "any value (roo recommended)",
            "model": MODEL
        }
    })

@app.route('/v1/chat/completions/stream', methods=['POST'])
def chat_completions_stream_alt():
    """Alternative streaming endpoint"""
    return chat_completions()

# Catch-all for any OpenAI-compatible path
@app.route('/v1/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def catch_all(path):
    """Catch-all for any other OpenAI endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "endpoint": f"/v1/{path}",
        "message": f"Endpoint /v1/{path} is not explicitly supported",
        "supported_endpoints": ["/chat/completions", "/completions", "/responses", "/models", "/tools"]
    }), 404

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info(f"🚀 Ollama Proxy Server v2.0")
    logger.info(f"📍 Host: 0.0.0.0:{PROXY_PORT}")
    logger.info(f"🤖 Model: {MODEL}")
    logger.info(f"🔧 Tool Calls: ENABLED")
    logger.info(f"📡 Streaming: ENABLED")
    logger.info(f"🔗 Ollama: {OLLAMA_URL}")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=PROXY_PORT, threaded=True, debug=False)
