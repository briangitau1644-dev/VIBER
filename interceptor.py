#!/usr/bin/env python3
"""Minimal Ollama proxy - designed for GitHub Actions reliability"""

import os, sys, json, time, logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.request, urllib.error

# Config
PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = os.getenv('MODEL', 'gemma3:4b')
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
CTX_SIZE = int(os.getenv('CTX_SIZE', '8192'))

OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Simple logging
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    
    def log_message(self, format, *args):
        log(f"HTTP: {args[0]}")
    
    def _send_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    
    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)
    
    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/health':
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
                    self._json_response(200 if r.status == 200 else 503, {"status": "healthy" if r.status == 200 else "degraded"})
            except Exception as e:
                self._json_response(503, {"status": "error", "error": str(e)})
        elif path == '/api/tags':
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=30) as r:
                    self.send_response(r.status)
                    self.send_header('Content-Type', r.headers.get('Content-Type', 'application/json'))
                    self._send_cors()
                    self.end_headers()
                    self.wfile.write(r.read())
            except Exception as e:
                self._json_response(502, {"error": str(e)})
        elif path == '/v1/models':
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=30) as r:
                    data = json.loads(r.read())
                    models = [{"id": m.get('name', MODEL), "object": "model", "created": int(time.time()), "owned_by": "ollama"} for m in data.get('models', [])]
                    if not any(m['id'] == MODEL for m in models):
                        models.append({"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"})
                    self._json_response(200, {"object": "list", "data": models})
            except Exception as e:
                self._json_response(502, {"error": str(e)})
        elif path == '/':
            self._json_response(200, {"service": "Ollama Proxy", "model": MODEL, "port": PROXY_PORT})
        else:
            self._json_response(404, {"error": "Not found"})
    
    def do_POST(self):
        path = urlparse(self.path).path
        if path != '/v1/chat/completions':
            self._json_response(404, {"error": "Not found"})
            return
        
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            req = json.loads(body)
        except Exception as e:
            self._json_response(400, {"error": f"Invalid JSON: {e}"})
            return
        
        # Extract params
        model = req.get('model', MODEL)
        messages = req.get('messages', [])
        stream = req.get('stream', False)
        max_tokens = req.get('max_tokens', req.get('max_completion_tokens', 2048))
        temperature = req.get('temperature', TEMPERATURE)
        
        # Transform to Ollama format
        ollama_msgs = [{'role': m.get('role','user'), 'content': m.get('content','')} for m in messages]
        payload = {
            'model': model,
            'messages': ollama_msgs,
            'stream': stream,
            'options': {'temperature': temperature, 'num_ctx': CTX_SIZE, 'num_predict': max_tokens}
        }
        
        try:
            if stream:
                # Simple streaming proxy
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self._send_cors()
                self.end_headers()
                with urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}), timeout=120) as resp:
                    for line in resp:
                        if line: self.wfile.write(line)
                        self.wfile.flush()
            else:
                with urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}), timeout=180) as resp:
                    result = json.loads(resp.read())
                    content = result.get('message', {}).get('content', '')
                    self._json_response(200, {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": result.get('done_reason', 'stop')}],
                        "usage": {
                            "prompt_tokens": result.get('prompt_eval_count', 0),
                            "completion_tokens": result.get('eval_count', 0),
                            "total_tokens": result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                        }
                    })
        except urllib.error.HTTPError as e:
            self._json_response(e.code, {"error": f"Ollama HTTP {e.code}"})
        except urllib.error.URLError as e:
            self._json_response(502, {"error": f"Cannot reach Ollama: {e.reason}"})
        except Exception as e:
            log(f"ERROR: {e}")
            self._json_response(500, {"error": str(e)})

def main():
    log(f"🚀 Starting minimal proxy on 0.0.0.0:{PROXY_PORT}")
    log(f"   → Ollama: {OLLAMA_URL}, Model: {MODEL}")
    try:
        server = HTTPServer(('0.0.0.0', PROXY_PORT), ProxyHandler)
        log(f"✅ Server running")
        server.serve_forever()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            log(f"❌ Port {PROXY_PORT} already in use!")
        else:
            log(f"❌ Server error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        log("👋 Shutting down")
        sys.exit(0)

if __name__ == '__main__':
    main()
