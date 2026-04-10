#!/usr/bin/env python3
"""Minimal Ollama proxy for GitHub Actions - uses only Python stdlib"""
import os, sys, json, time, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

PROXY_PORT = int(os.getenv('PROXY_PORT', '1234'))
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', '127.0.0.1')
MODEL = os.getenv('MODEL', 'gemma4:e4b')
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
CTX_SIZE = int(os.getenv('CTX_SIZE', '8192'))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

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
                urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
                self._json(200, {"status": "healthy", "model": MODEL})
            except Exception as e:
                self._json(503, {"status": "error", "error": str(e)})
        elif path == '/api/tags':
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=30) as r:
                    self.send_response(r.status)
                    self.send_header('Content-Type', r.headers.get('Content-Type', 'application/json'))
                    self._cors(); self.end_headers(); self.wfile.write(r.read())
            except Exception as e: self._json(502, {"error": str(e)})
        elif path == '/v1/models':
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=30) as r:
                    data = json.loads(r.read())
                    models = [{"id": m.get('name', MODEL), "object": "model", "created": int(time.time()), "owned_by": "ollama"} for m in data.get('models', [])]
                    if not any(m['id'] == MODEL for m in models):
                        models.append({"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"})
                    self._json(200, {"object": "list", "data": models})
            except Exception as e: self._json(502, {"error": str(e)})
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
        except Exception as e: return self._json(400, {"error": f"Invalid JSON: {e}"})

        model = req.get('model', MODEL)
        messages = req.get('messages', [])
        stream = req.get('stream', False)
        max_tok = req.get('max_tokens', req.get('max_completion_tokens', 2048))
        temp = req.get('temperature', TEMPERATURE)

        ollama_msgs = [{'role': m.get('role','user'), 'content': m.get('content','')} for m in messages]
        payload = {'model': model, 'messages': ollama_msgs, 'stream': stream, 'options': {'temperature': temp, 'num_ctx': CTX_SIZE, 'num_predict': max_tok}}

        try:
            if stream:
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache'); self._cors(); self.end_headers()
                with urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}), timeout=120) as resp:
                    for line in resp:
                        if line: self.wfile.write(line)
                        self.wfile.flush()
            else:
                with urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}), timeout=180) as resp:
                    res = json.loads(resp.read())
                    self._json(200, {
                        "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
                        "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": res.get('message',{}).get('content','')}, "finish_reason": res.get('done_reason','stop')}],
                        "usage": {"prompt_tokens": res.get('prompt_eval_count',0), "completion_tokens": res.get('eval_count',0), "total_tokens": res.get('prompt_eval_count',0)+res.get('eval_count',0)}
                    })
        except urllib.error.HTTPError as e: self._json(e.code, {"error": f"Ollama HTTP {e.code}"})
        except urllib.error.URLError as e: self._json(502, {"error": f"Cannot reach Ollama: {e.reason}"})
        except Exception as e: log(f"ERROR: {e}"); self._json(500, {"error": str(e)})

if __name__ == '__main__':
    log(f"🚀 Starting proxy on 0.0.0.0:{PROXY_PORT}")
    log(f"   → Model: {MODEL}")
    log(f"   → Ollama: {OLLAMA_URL} | Temp: {TEMPERATURE} | Ctx: {CTX_SIZE}")
    try:
        server = HTTPServer(('0.0.0.0', PROXY_PORT), Handler)
        log("✅ SERVER BOUND AND LISTENING")
        server.serve_forever()
    except OSError as e:
        log(f"❌ FAILED TO BIND: {e}")
        sys.exit(1)
