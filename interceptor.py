#!/usr/bin/env python3
"""Ollama proxy for gemma4:e4b - extended timeouts, raw fallback, verbose logging"""
import os, sys, json, time, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

# Hardcoded config - gemma4:e4b ONLY
PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = 'gemma4:e4b'  # FIXED: Only this model
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
CTX_SIZE = int(os.getenv('CTX_SIZE', '8192'))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Extended timeouts for large/slow models
OLLAMA_TIMEOUT = 600  # 10 minutes for generation
HEALTH_TIMEOUT = 10

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, format, *args): log(f"HTTP: {args[0]}")
    
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    
    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/health':
            try:
                urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=HEALTH_TIMEOUT)
                self._json(200, {"status": "healthy", "model": MODEL})
            except Exception as e:
                log(f"❌ HEALTH: {e}")
                self._json(503, {"status": "error", "error": str(e)})
        elif path == '/api/tags':
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=30) as r:
                    self.send_response(r.status)
                    self.send_header('Content-Type', r.headers.get('Content-Type', 'application/json'))
                    self._cors(); self.end_headers(); self.wfile.write(r.read())
            except Exception as e: 
                log(f"❌ TAGS: {e}")
                self._json(502, {"error": str(e)})
        elif path == '/v1/models':
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=30) as r:
                    data = json.loads(r.read())
                    models = [{"id": m.get('name', MODEL), "object": "model", "created": int(time.time()), "owned_by": "ollama"} for m in data.get('models', [])]
                    if not any(m['id'] == MODEL for m in models):
                        models.append({"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"})
                    self._json(200, {"object": "list", "data": models})
            except Exception as e: 
                log(f"❌ MODELS: {e}")
                self._json(502, {"error": str(e)})
        elif path == '/':
            self._json(200, {"service": "Ollama Proxy", "model": MODEL, "port": PROXY_PORT})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path.split('?')[0] != '/v1/chat/completions':
            return self._json(404, {"error": "Not found"})
        
        try:
            length = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(length))
        except Exception as e: 
            log(f"❌ PARSE: {e}")
            return self._json(400, {"error": f"Invalid JSON: {e}"})

        # Force gemma4:e4b regardless of request
        messages = req.get('messages', [])
        stream = req.get('stream', False)
        max_tok = req.get('max_tokens', req.get('max_completion_tokens', 2048))
        temp = req.get('temperature', TEMPERATURE)

        ollama_msgs = [{'role': m.get('role','user'), 'content': m.get('content','')} for m in messages]
        payload = {
            'model': MODEL,  # HARD CODED
            'messages': ollama_msgs, 
            'stream': stream, 
            'options': {'temperature': temp, 'num_ctx': CTX_SIZE, 'num_predict': max_tok}
        }

        log(f"→ Forwarding to Ollama: model={MODEL}, msgs={len(ollama_msgs)}, max_tok={max_tok}")

        try:
            ollama_req = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat", 
                data=json.dumps(payload).encode(), 
                headers={'Content-Type': 'application/json'}
            )
            
            if stream:
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache'); self._cors(); self.end_headers()
                with urllib.request.urlopen(ollama_req, timeout=OLLAMA_TIMEOUT) as resp:
                    for line in resp:
                        if line: self.wfile.write(line)
                        self.wfile.flush()
            else:
                log(f"⏳ Waiting for Ollama response (timeout: {OLLAMA_TIMEOUT}s)...")
                start = time.time()
                with urllib.request.urlopen(ollama_req, timeout=OLLAMA_TIMEOUT) as resp:
                    elapsed = time.time() - start
                    raw = resp.read()
                    log(f"← Ollama responded in {elapsed:.1f}s ({len(raw)} bytes)")
                    
                    # Try to parse, but fallback to raw if needed
                    try:
                        res = json.loads(raw)
                        log(f"← Parsed JSON: keys={list(res.keys())}")
                        
                        # Multiple ways to extract content (handle different Ollama versions)
                        content = None
                        if isinstance(res, dict):
                            # Standard format
                            if 'message' in res and isinstance(res['message'], dict):
                                content = res['message'].get('content', '')
                            # Alternative: direct content field
                            elif 'content' in res:
                                content = res['content']
                            # Debug: show what we got
                            if content is None:
                                log(f"⚠️ Content not found in standard locations. Full response: {json.dumps(res)[:500]}")
                                content = ''
                        else:
                            log(f"⚠️ Response is not a dict: {type(res)}")
                            content = ''
                        
                        # Fallback: if still empty, try to extract from raw string
                        if not content and raw:
                            log(f"⚠️ Empty content, attempting raw fallback...")
                            # Try to find content in raw JSON string
                            import re
                            match = re.search(r'"content"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw.decode())
                            if match:
                                content = match.group(1).replace('\\"', '"').replace('\\n', '\n')
                                log(f"← Extracted content via regex: {content[:100]}...")
                        
                        log(f"← Final content length: {len(content) if content else 0}")
                        
                        self._json(200, {
                            "id": f"chatcmpl-{int(time.time())}", 
                            "object": "chat.completion", 
                            "created": int(time.time()),
                            "model": MODEL, 
                            "choices": [{
                                "index": 0, 
                                "message": {"role": "assistant", "content": content or ''}, 
                                "finish_reason": res.get('done_reason', res.get('finish_reason', 'stop'))
                            }],
                            "usage": {
                                "prompt_tokens": res.get('prompt_eval_count', res.get('prompt_tokens', 0)), 
                                "completion_tokens": res.get('eval_count', res.get('completion_tokens', 0)), 
                                "total_tokens": res.get('prompt_eval_count', 0) + res.get('eval_count', 0)
                            }
                        })
                    except json.JSONDecodeError as je:
                        log(f"❌ JSON decode failed: {je}")
                        log(f"← Raw response (first 500 chars): {raw[:500]}")
                        # Fallback: return raw as text content
                        self._json(200, {
                            "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                            "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": raw.decode()[:4000]}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 0, "completion_tokens": len(raw), "total_tokens": len(raw)}
                        })
                        
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if hasattr(e, 'read') else ''
            log(f"❌ Ollama HTTP {e.code}: {error_body[:300]}")
            self._json(e.code, {"error": f"Ollama HTTP {e.code}: {error_body[:500]}"})
        except urllib.error.URLError as e:
            log(f"❌ Connection: {e.reason}")
            self._json(502, {"error": f"Cannot reach Ollama: {e.reason}"})
        except TimeoutError as e:
            log(f"❌ TIMEOUT after {OLLAMA_TIMEOUT}s - model may be loading or generating slowly")
            self._json(504, {"error": f"Timeout: gemma4:e4b is still processing (try increasing timeout)"})
        except Exception as e: 
            log(f"❌ CRASH: {type(e).__name__}: {e}", exc_info=True)
            self._json(500, {"error": f"{type(e).__name__}: {str(e)[:300]}", "model": MODEL})

if __name__ == '__main__':
    log(f"🚀 PROXY START: 0.0.0.0:{PROXY_PORT}")
    log(f"   → MODEL: {MODEL} (HARD CODED)")
    log(f"   → OLLAMA: {OLLAMA_URL} | Temp: {TEMPERATURE} | Ctx: {CTX_SIZE} | Timeout: {OLLAMA_TIMEOUT}s")
    try:
        server = HTTPServer(('0.0.0.0', PROXY_PORT), Handler)
        log("✅ LISTENING")
        server.serve_forever()
    except OSError as e:
        log(f"❌ BIND FAILED: {e}")
        sys.exit(1)
