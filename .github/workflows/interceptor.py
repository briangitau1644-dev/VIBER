import http.server, json, os, re, uuid, urllib.request, urllib.error, traceback

BACKEND = "http://127.0.0.1:" + os.environ.get("OLLAMA_PORT", "11434")
PORT    = int(os.environ.get("PROXY_PORT", "1234"))
MODEL   = os.environ.get("MODEL", "gemma3:4b")

SYSTEM_PROMPT = (
    "You are a Senior Software Developer with 15 years of full-stack experience. "
    "Languages: Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, Kotlin, Swift, PHP, Ruby, Bash, SQL, GraphQL. "
    "Frontends: React, Vue, Svelte, HTML/CSS/Tailwind. "
    "Backends: REST, gRPC, microservices, serverless, message queues. "
    "Databases: PostgreSQL, MySQL, SQLite, MongoDB, Redis, Cassandra, Elasticsearch. "
    "Cloud: AWS, GCP, Azure. Containers: Docker, Kubernetes. "
    "You are running inside Roo Code (VS Code extension). You have access to tools.\n\n"
    "TOOL CALL PROTOCOL\n"
    "When you need to use a tool output ONLY a JSON function call in this exact format:\n"
    "{\"tool_use\": {\"name\": \"<tool_name>\", \"parameters\": {<key-value pairs>}}}\n\n"
    "Available tools:\n"
    "  read_file        : {\"path\": \"<path>\"}\n"
    "  write_file       : {\"path\": \"<path>\", \"content\": \"<full file content>\"}\n"
    "  create_directory : {\"path\": \"<dir path>\"}\n"
    "  list_directory   : {\"path\": \"<dir path>\"}\n"
    "  execute_command  : {\"command\": \"<shell command>\", \"cwd\": \"<optional>\"}\n"
    "  search_files     : {\"pattern\": \"<glob or regex>\", \"path\": \"<root dir>\"}\n"
    "  web_search       : {\"query\": \"<search query>\"}\n\n"
    "Rules:\n"
    "- Call ONE tool at a time. Wait for the result before calling the next.\n"
    "- Tool results arrive as messages with role tool.\n"
    "- Write complete production-quality code with error handling. No placeholders.\n"
    "- Be direct. Talk like a senior colleague.\n"
)

RE_XML_TOOL  = re.compile(r"<tool_call[^>]*>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
RE_JSON_TOOL = re.compile(r"```(?:json)?\s*(\{[^`]*\"tool_use\"[^`]*\})\s*```", re.DOTALL)


def extract_tool_call(text):
    try:
        obj = json.loads(text.strip())
        if "tool_use" in obj:
            tu = obj["tool_use"]
            return tu.get("name"), tu.get("parameters", {})
    except Exception:
        pass
    m = RE_JSON_TOOL.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            tu = obj.get("tool_use", {})
            return tu.get("name"), tu.get("parameters", {})
        except Exception:
            pass
    m = RE_XML_TOOL.search(text)
    if m:
        inner = m.group(1).strip()
        try:
            obj = json.loads(inner)
            name   = obj.get("name") or obj.get("tool") or obj.get("function")
            params = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
            if name:
                return name, params
        except Exception:
            pass
        tag = re.search(r"<([a-z_]+)>(.*?)</\1>", inner, re.DOTALL)
        if tag:
            name = tag.group(1)
            try:
                params = json.loads(tag.group(2).strip())
            except Exception:
                params = {"value": tag.group(2).strip()}
            return name, params
    return None, None


def build_tool_call_response(name, params, original):
    call_id = "call_" + uuid.uuid4().hex[:8]
    return {
        "id": original.get("id", "chatcmpl-intercepted"),
        "object": "chat.completion",
        "created": original.get("created", 0),
        "model": original.get("model", MODEL),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(params)
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": original.get("usage", {})
    }


def inject_system_prompt(messages):
    if not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    existing = messages[0].get("content", "")
    if "TOOL CALL PROTOCOL" not in existing:
        messages = list(messages)
        messages[0] = {"role": "system", "content": existing + "\n\n" + SYSTEM_PROMPT}
    return messages


class Interceptor(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, X-Requested-With, Accept, "
            "X-Api-Key, api-key, x-stainless-os, x-stainless-lang, "
            "x-stainless-package-version, x-stainless-runtime, "
            "x-stainless-runtime-version, x-stainless-arch",
        )
        self.send_header("Access-Control-Max-Age", "86400")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            try:
                r = urllib.request.urlopen(BACKEND + "/api/tags", timeout=5)
                if r.status == 200:
                    body = b'{"status":"ok"}'
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except Exception:
                pass
            body = b'{"status":"unavailable"}'
            self.send_response(503)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._fwd()

    def do_DELETE(self):
        self._fwd()

    def _fwd(self, body=None):
        url  = BACKEND + self.path
        hdrs = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "authorization", "api-key", "x-api-key")
        }
        hdrs["Host"] = "127.0.0.1:" + os.environ.get("OLLAMA_PORT", "11434")
        if body is not None:
            hdrs["Content-Length"] = str(len(body))
        req = urllib.request.Request(url, data=body, headers=hdrs, method=self.command)
        try:
            r = urllib.request.urlopen(req, timeout=600)
            st, rh, rb = r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            st, rh, rb = e.code, e.headers, e.read()
        except Exception as e:
            print("[interceptor] forward error: " + str(e), flush=True)
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(st)
        self._cors()
        for k, v in rh.items():
            if k.lower() == "content-type":
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(rb)))
        self.end_headers()
        self.wfile.write(rb)

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        raw     = self.rfile.read(length)
        is_chat = "/chat/completions" in self.path or "/api/chat" in self.path

        if not is_chat:
            self._fwd(raw)
            return

        try:
            payload = json.loads(raw)
            payload["messages"] = inject_system_prompt(payload.get("messages", []))
            payload["model"]    = MODEL
            payload.setdefault("temperature", float(os.environ.get("TEMPERATURE", "0.2")))
            payload.setdefault("options", {})
            payload["options"].setdefault("num_ctx",        int(os.environ.get("CTX_SIZE", "8192")))
            payload["options"].setdefault("repeat_penalty", 1.05)
            payload["options"].setdefault("top_p",          0.95)
            payload["options"].setdefault("top_k",          64)

            if payload.get("stream", False):
                self._fwd(json.dumps(payload).encode())
                return

            raw  = json.dumps(payload).encode()
            url  = BACKEND + self.path
            hdrs = {
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "authorization", "api-key", "x-api-key")
            }
            hdrs["Host"]           = "127.0.0.1:" + os.environ.get("OLLAMA_PORT", "11434")
            hdrs["Content-Length"] = str(len(raw))

            req = urllib.request.Request(url, data=raw, headers=hdrs, method="POST")
            try:
                r = urllib.request.urlopen(req, timeout=600)
                st, rh, rb = r.status, r.headers, r.read()
            except urllib.error.HTTPError as e:
                st, rh, rb = e.code, e.headers, e.read()
                self._reply(st, rb)
                return
            except Exception as e:
                print("[interceptor] ollama error: " + str(e), flush=True)
                body = json.dumps({"error": {"message": str(e), "type": "upstream_error"}}).encode()
                self._reply(502, body)
                return

            try:
                resp_obj = json.loads(rb)
            except Exception:
                self._fwd_raw(st, rh, rb)
                return

            assistant_text = ""
            try:
                assistant_text = resp_obj["choices"][0]["message"]["content"] or ""
            except Exception:
                pass

            tool_name, tool_params = extract_tool_call(assistant_text)
            if tool_name:
                print("[interceptor] tool call detected: " + tool_name, flush=True)
                body = json.dumps(
                    build_tool_call_response(tool_name, tool_params, resp_obj)
                ).encode()
                self._reply(200, body)
                return

            self._fwd_raw(200, rh, rb)

        except Exception:
            traceback.print_exc()
            try:
                self._fwd(raw)
            except Exception:
                pass

    def _fwd_raw(self, status, headers, body):
        self.send_response(status)
        self._cors()
        for k, v in headers.items():
            if k.lower() == "content-type":
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reply(self, status, body):
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


srv = http.server.HTTPServer(("0.0.0.0", PORT), Interceptor)
print("[interceptor] :" + str(PORT) + " -> " + BACKEND, flush=True)
srv.serve_forever()
