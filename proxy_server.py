#!/usr/bin/env python3
"""
Ollama Proxy Server with Full Tool Calling Support
GitHub Actions Edition - CPU Optimized
Supports: OpenAI /v1/chat/completions + Anthropic /v1/messages
"""

import os, sys, json, time, uuid, threading, random, subprocess, requests
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL = os.getenv('MODEL', 'sorc/qwen3.5-claude-4.6-opus:4b')
OLLAMA_BASE = f"http://127.0.0.1:{os.getenv('OLLAMA_PORT', '11434')}"
PROXY_PORT = int(os.getenv('PROXY_PORT', '8000'))
RAM_PRESSURE_ENABLED = os.getenv('RAM_PRESSURE_ENABLED', 'true').lower() == 'true'
RAM_PRESSURE_MB = int(os.getenv('RAM_PRESSURE_MB', '256'))
ACTIVITY_LOG_INTERVAL = float(os.getenv('ACTIVITY_LOG_INTERVAL', '30'))
KEEPALIVE_INTERVAL = float(os.getenv('KEEPALIVE_INTERVAL', '10'))

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"⚡ Proxy Starting - Model: {MODEL}")

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": "get_current_time", "description": "Get current date/time in ISO format.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "calculate", "description": "Perform math: +, -, *, /, **, %, //", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search for information (simulated).", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for location (simulated).", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
]

# ─────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    @staticmethod
    def execute(name: str, arguments: Dict[str, Any]) -> str:
        try:
            fn = getattr(ToolExecutor, f"_{name}", None)
            if fn is None:
                return json.dumps({"error": f"Unknown tool: {name}"})
            return json.dumps(fn(**arguments), ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
    
    @staticmethod
    def _get_current_time() -> Dict:
        return {"timestamp": datetime.now().isoformat(), "timezone": "UTC"}
    
    @staticmethod
    def _calculate(expression: str) -> Dict:
        try:
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return {"error": "Invalid characters"}
            return {"expression": expression, "result": eval(expression, {"__builtins__": {}}, {})}
        except Exception as e:
            return {"error": f"Calculation failed: {e}"}
    
    @staticmethod
    def _search_web(query: str) -> Dict:
        return {"query": query, "results": [{"title": f"Result for '{query}'", "url": "https://example.com"}], "note": "Simulated"}
    
    @staticmethod
    def _get_weather(location: str) -> Dict:
        import random
        return {"location": location, "temperature_c": random.randint(-10, 35), "condition": random.choice(["sunny", "cloudy", "rainy"]), "note": "Simulated"}

tool_executor = ToolExecutor()

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE KEEP-ALIVE
# ─────────────────────────────────────────────────────────────────────────────

_ram_buffer = None

def _ram_pressure_thread():
    global _ram_buffer
    if not RAM_PRESSURE_ENABLED:
        return
    try:
        _ram_buffer = bytearray(RAM_PRESSURE_MB * 1024 * 1024)
        for i in range(0, len(_ram_buffer), 4096):
            _ram_buffer[i] = 1
        log(f"💾 Reserved {RAM_PRESSURE_MB} MB RAM")
    except:
        pass
    while True:
        if _ram_buffer:
            _ram_buffer[0] = _ram_buffer[-1] = 1
        time.sleep(30)

def _activity_logger():
    import psutil
    while True:
        try:
            ram = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.5)
            log(f"📊 CPU: {cpu}% | RAM: {ram.percent}%")
        except:
            pass
        time.sleep(ACTIVITY_LOG_INTERVAL)

def _ollama_keepalive():
    session = requests.Session()
    while True:
        try:
            resp = session.post(f"{OLLAMA_BASE}/api/generate",
                json={"model": MODEL, "prompt": ".", "keep_alive": -1, "stream": False, "options": {"num_predict": 1}},
                timeout=(5, 20))
            if resp.status_code == 200:
                log(f"💓 Model ping OK")
        except Exception as e:
            log(f"⚠️ Keepalive: {e}")
        time.sleep(KEEPALIVE_INTERVAL + random.uniform(-1, 1))

def _start_background_threads():
    for fn, name in [(_ram_pressure_thread, "ram"), (_activity_logger, "logger"), (_ollama_keepalive, "keepalive")]:
        threading.Thread(target=fn, daemon=True, name=name).start()
    log("✅ Background threads started")

# ─────────────────────────────────────────────────────────────────────────────
# FORMAT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_tool_params(params: Optional[Dict]) -> Dict:
    if not params or not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    return params if "type" in params else {**params, "type": "object"}

def _oai_tools_to_ollama(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    if tools is None:
        return None
    result = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        result.append({"type": "function", "function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": _normalize_tool_params(fn.get("parameters"))
        }})
    return result if result else None

def _oai_messages_to_ollama(messages: List[Dict]) -> List[Dict]:
    result = []
    for msg in messages:
        role, content = msg.get("role", ""), msg.get("content") or ""
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        c = block.get("content", "")
                        if isinstance(c, list):
                            c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                        parts.append(str(c))
                else:
                    parts.append(str(block))
            content = "\n".join(parts)
        if role == "system":
            result.append({"role": "system", "content": content})
        elif role == "user":
            result.append({"role": "user", "content": content})
        elif role == "assistant":
            oai_msg = {"role": "assistant", "content": content}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                ollama_tcs = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    ollama_tcs.append({"function": {"name": fn.get("name", ""), "arguments": args}})
                oai_msg["tool_calls"] = ollama_tcs
            result.append(oai_msg)
        elif role == "tool":
            tool_msg = {"role": "tool", "content": str(content)}
            if msg.get("tool_call_id"):
                tool_msg["tool_call_id"] = msg["tool_call_id"]
            result.append(tool_msg)
    return result

def _ollama_tool_calls_to_oai(ollama_tcs: List[Dict]) -> List[Dict]:
    result = []
    for tc in ollama_tcs:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        result.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args_str}
        })
    return result

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA COMMUNICATION
# ─────────────────────────────────────────────────────────────────────────────

def _make_ollama_session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504, 429])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.headers.update({"Connection": "keep-alive"})
    return session

ollama_session = _make_ollama_session()

def _ollama_chat_sync(messages: List[Dict], tools: Optional[List[Dict]] = None, max_tokens: int = 2048, temperature: float = 0.7) -> Dict:
    payload = {"model": MODEL, "stream": False, "keep_alive": -1, "messages": _oai_messages_to_ollama(messages), "options": {"num_predict": max_tokens, "temperature": temperature}}
    if tools is not None:
        payload["tools"] = _oai_tools_to_ollama(tools)
    try:
        resp = ollama_session.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=(10, 300))
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        raise HTTPException(504, "Ollama timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(503, f"Ollama error: {e}")

def _ollama_chat_stream(messages: List[Dict], tools: Optional[List[Dict]] = None, max_tokens: int = 2048, temperature: float = 0.7):
    payload = {"model": MODEL, "stream": True, "keep_alive": -1, "messages": _oai_messages_to_ollama(messages), "options": {"num_predict": max_tokens, "temperature": temperature}}
    if tools is not None:
        payload["tools"] = _oai_tools_to_ollama(tools)
    return ollama_session.post(f"{OLLAMA_BASE}/api/chat", json=payload, stream=True, timeout=(10, 300))

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _generate_openai_stream(messages: List[Dict], tools: Optional[List[Dict]], max_tokens: int, temperature: float):
    yield ": stream-start\n\n"
    try:
        resp = _ollama_chat_stream(messages, tools, max_tokens, temperature)
    except Exception as e:
        yield f" {json.dumps({'error': str(e)})}\n\n"
        yield " [DONE]\n\n"
        return
    last_ping = time.time()
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            if time.time() - last_ping > 15:
                yield ": keep-alive\n\n"
                last_ping = time.time()
            try:
                chunk = json.loads(line)
            except:
                continue
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            done = chunk.get("done", False)
            delta = {"content": content} if content else {}
            if done and msg.get("tool_calls"):
                oai_tcs = _ollama_tool_calls_to_oai(msg["tool_calls"])
                tool_chunk = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "delta": {"tool_calls": oai_tcs}, "finish_reason": None}]}
                yield f" {json.dumps(tool_chunk)}\n\n"
                finish_chunk = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                yield f" {json.dumps(finish_chunk)}\n\n"
                yield " [DONE]\n\n"
                return
            oai_chunk = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if done else None}]}
            yield f" {json.dumps(oai_chunk)}\n\n"
            if done:
                yield " [DONE]\n\n"
                return
    except Exception as e:
        yield f" {json.dumps({'error': f'Stream error: {e}'})}\n\n"
        yield " [DONE]\n\n"

def _generate_anthropic_stream(messages: List[Dict], tools: Optional[List[Dict]], max_tokens: int, temperature: float):
    msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    yield "event: message_start\n "
    yield json.dumps({"type": "message_start", "message": {"id": msg_id, "type": "message", "role": "assistant", "content": [], "model": MODEL, "stop_reason": None, "usage": {"input_tokens": 0, "output_tokens": 0}}}) + "\n\n"
    yield "event: content_block_start\n "
    yield json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}) + "\n\n"
    try:
        resp = _ollama_chat_stream(messages, tools, max_tokens, temperature)
    except Exception as e:
        yield f"event: error\n {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        return
    last_ping = time.time()
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            if time.time() - last_ping > 15:
                yield "event: ping\n {\"type\":\"ping\"}\n\n"
                last_ping = time.time()
            try:
                chunk = json.loads(line)
            except:
                continue
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            done = chunk.get("done", False)
            if content:
                yield "event: content_block_delta\n "
                yield json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": content}}) + "\n\n"
            if done:
                yield "event: content_block_stop\n "
                yield json.dumps({"type": "content_block_stop", "index": 0}) + "\n\n"
                stop_reason = "end_turn"
                if msg.get("tool_calls"):
                    stop_reason = "tool_use"
                    for idx, tc in enumerate(msg["tool_calls"], 1):
                        fn = tc.get("function", {})
                        args = fn.get("arguments", {})
                        tu_id = f"toolu_{uuid.uuid4().hex[:8]}"
                        yield "event: content_block_start\n "
                        yield json.dumps({"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": tu_id, "name": fn.get("name", ""), "input": {}}}) + "\n\n"
                        yield "event: content_block_delta\n "
                        yield json.dumps({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": json.dumps(args if isinstance(args, dict) else {})}}) + "\n\n"
                        yield "event: content_block_stop\ndata: "
                        yield json.dumps({"type": "content_block_stop", "index": idx}) + "\n\n"
                yield "event: message_delta\n "
                yield json.dumps({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 0}}) + "\n\n"
                yield "event: message_stop\n "
                yield json.dumps({"type": "message_stop"}) + "\n\n"
                return
    except Exception as e:
        yield f"event: error\n {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Ollama Proxy with Tool Calling", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict]] = ""
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = "auto"

@app.get("/")
async def root():
    return {"service": "Ollama Proxy with Tool Calling", "version": "1.0.0", "model": MODEL}

@app.get("/health")
async def health():
    try:
        resp = ollama_session.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return {"status": "ready" if MODEL in models else "degraded", "model": MODEL, "model_loaded": MODEL in models, "ollama_ok": True}
    except:
        pass
    return {"status": "down", "model": MODEL}

@app.get("/v1/models")
async def list_models():
    try:
        resp = ollama_session.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = [{"id": m.get("name", ""), "object": "model", "owned_by": "ollama"} for m in resp.json().get("models", [])]
            return {"object": "list", "data": data if data else [{"id": MODEL, "object": "model", "owned_by": "ollama"}]}
    except:
        pass
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "ollama"}]}

@app.get("/v1/tools")
async def list_tools():
    return {"tools": TOOL_DEFINITIONS, "count": len(TOOL_DEFINITIONS)}

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(400, "Invalid JSON")
    try:
        req = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(400, f"Validation failed: {e}")
    messages = [m.model_dump() for m in req.messages]
    tool_choice = req.tool_choice if isinstance(req.tool_choice, str) else (req.tool_choice or {}).get("type", "auto")
    effective_tools = None if tool_choice == "none" else (req.tools if req.tools is not None else TOOL_DEFINITIONS)
    if req.stream:
        return StreamingResponse(_generate_openai_stream(messages, effective_tools, req.max_tokens, req.temperature), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    try:
        result = _ollama_chat_sync(messages, effective_tools, req.max_tokens, req.temperature)
        msg = result.get("message", {})
        content = msg.get("content") or ""
        ollama_tcs = msg.get("tool_calls") or []
        response_msg = {"role": "assistant", "content": content or None}
        finish_reason = "stop"
        if ollama_tcs:
            response_msg["tool_calls"] = _ollama_tool_calls_to_oai(ollama_tcs)
            finish_reason = "tool_calls"
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "message": response_msg, "finish_reason": finish_reason}], "usage": {"prompt_tokens": result.get("prompt_eval_count", 0), "completion_tokens": result.get("eval_count", 0), "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Chat failed: {e}")

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(400, "Invalid JSON")
    system = body.get("system", "")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 2048)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)
    tools = body.get("tools")
    tool_choice = body.get("tool_choice", {"type": "auto"})
    tc_type = tool_choice.get("type", "auto") if isinstance(tool_choice, dict) else str(tool_choice)
    full_messages = [{"role": "system", "content": system}] if system else []
    full_messages.extend(messages)
    effective_tools = None if tc_type == "none" else (_anthropic_tools_to_ollama(tools) if tools else _oai_tools_to_ollama(TOOL_DEFINITIONS))
    if stream:
        return StreamingResponse(_generate_anthropic_stream(full_messages, effective_tools, max_tokens, temperature), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    try:
        result = _ollama_chat_sync(full_messages, effective_tools, max_tokens, temperature)
        msg = result.get("message", {})
        content = msg.get("content") or ""
        ollama_tcs = msg.get("tool_calls") or []
        content_blocks = [{"type": "text", "text": content}] if content else []
        stop_reason = "end_turn"
        for tc in ollama_tcs:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            content_blocks.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:8]}", "name": fn.get("name", ""), "input": args if isinstance(args, dict) else {}})
            stop_reason = "tool_use"
        return {"id": f"msg_{uuid.uuid4().hex[:8]}", "type": "message", "role": "assistant", "model": MODEL, "content": content_blocks, "stop_reason": stop_reason, "usage": {"input_tokens": result.get("prompt_eval_count", 0), "output_tokens": result.get("eval_count", 0)}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Messages failed: {e}")

@app.post("/v1/tool/execute")
async def execute_tool(request: Request):
    try:
        body = await request.json()
        tool_name = body.get("name")
        arguments = body.get("arguments", {})
        if not tool_name:
            raise HTTPException(400, "Missing 'name'")
        result = tool_executor.execute(tool_name, arguments)
        return {"tool": tool_name, "result": json.loads(result)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Tool execution failed: {e}")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str):
    return JSONResponse({"error": "Not found", "path": f"/{path}", "available": ["/", "/health", "/v1/models", "/v1/tools", "/v1/chat/completions", "/v1/messages", "/v1/tool/execute"]}, status_code=404)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 70)
    log("  Ollama Proxy with Tool Calling - GitHub Actions")
    log("=" * 70)
    log(f"🤖 Model: {MODEL} | 🌐 Port: {PROXY_PORT} | 🔌 Ollama: {OLLAMA_BASE}")
    log("=" * 70)
    try:
        import psutil
    except ImportError:
        log("📦 Installing psutil...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "-q"], check=True)
        import psutil
    _start_background_threads()
    log(f"🚀 Starting Uvicorn on port {PROXY_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="warning", timeout_keep_alive=300)
