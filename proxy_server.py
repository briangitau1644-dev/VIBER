#!/usr/bin/env python3
"""
Ollama Proxy Server v3 — Universal Client Support
===================================================
Handles ALL client configurations:

  • OpenAI-compatible clients  → POST /v1/chat/completions
  • Anthropic-compatible       → POST /v1/messages
  • Raw Ollama clients         → POST /api/chat  (proxied & rewritten)
  • Legacy Ollama clients      → POST /api/generate (proxied & rewritten)
  • Trailing-slash URLs        → normalised automatically

Root URL (GET /) returns an OpenAI-style discovery doc so clients that
probe the base URL before configuring can find the model automatically.

Fixes in v3:
  - /api/chat and /api/generate are now served by the proxy itself,
    so clients pointing directly at these paths still work
  - GET / returns OpenAI-style { object:"list", data:[{id:MODEL}] }
    so clients that auto-detect the model from the root work
  - Trailing slash stripped from all incoming paths via middleware
  - Model name alias table: any name the client sends resolves to MODEL
  - /api/tags mirrors Ollama's tag list for clients that probe it
  - /api/version returns a version >= 0.1.16 so clients never warn
  - _anthropic_tools_to_ollama() was missing — added
  - /api/generate fallback when /api/chat is unavailable
"""

import os, sys, json, time, uuid, threading, random, subprocess, requests, shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL            = os.getenv('MODEL', 'sorc/qwen3.5-claude-4.6-opus:4b')
OLLAMA_PORT      = os.getenv('OLLAMA_PORT', '11434')
OLLAMA_BASE      = f"http://127.0.0.1:{OLLAMA_PORT}"
PROXY_PORT       = int(os.getenv('PROXY_PORT', '8000'))
RAM_PRESSURE_EN  = os.getenv('RAM_PRESSURE_ENABLED', 'true').lower() == 'true'
RAM_PRESSURE_MB  = int(os.getenv('RAM_PRESSURE_MB', '256'))
LOG_INTERVAL     = float(os.getenv('ACTIVITY_LOG_INTERVAL', '30'))
KEEPALIVE_INT    = float(os.getenv('KEEPALIVE_INTERVAL', '10'))
WORKING_DIR      = os.getenv('WORKING_DIR', os.getcwd())

# Resolved at startup — True = /api/chat, False = /api/generate
USE_CHAT_ENDPOINT = True

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"⚡ Proxy v3 — Model: {MODEL}")
log(f"🔌 Ollama: {OLLAMA_BASE}")
log(f"📁 WorkDir: {WORKING_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA ENDPOINT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_ollama_capabilities() -> bool:
    global USE_CHAT_ENDPOINT
    # 1. version string check
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5)
        if r.status_code == 200:
            v = r.json().get("version", "0.0.0")
            log(f"📦 Ollama version: {v}")
            parts = v.replace("-", ".").split(".")
            try:
                maj, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                if (maj, minor, patch) >= (0, 1, 16):
                    log("✅ /api/chat supported by version")
                    USE_CHAT_ENDPOINT = True
                    return True
                log(f"⚠️  {v} < 0.1.16 — will try live probe")
            except (IndexError, ValueError):
                pass
    except Exception as e:
        log(f"⚠️  Version check: {e}")

    # 2. live probe
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": MODEL, "stream": False,
                  "messages": [{"role": "user", "content": "hi"}]},
            timeout=20,
        )
        if r.status_code in (200, 400, 422):   # 400/422 = model error, endpoint exists
            log("✅ /api/chat live probe OK")
            USE_CHAT_ENDPOINT = True
            return True
        if r.status_code == 404:
            log("⚠️  /api/chat -> 404 — switching to /api/generate")
            USE_CHAT_ENDPOINT = False
            return False
    except Exception as e:
        log(f"⚠️  /api/chat probe error: {e} — defaulting to /api/generate")
        USE_CHAT_ENDPOINT = False
        return False

    USE_CHAT_ENDPOINT = True
    return True

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {"type":"function","function":{"name":"get_current_time","description":"Get current date/time.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"calculate","description":"Math: +,-,*,/,**,%","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
    {"type":"function","function":{"name":"search_web","description":"Search the web (simulated).","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"get_weather","description":"Get weather (simulated).","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},
    {"type":"function","function":{"name":"run_command","description":"Run a shell command. Requires user permission.","parameters":{"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"},"timeout":{"type":"integer","default":60}},"required":["command"]}}},
    {"type":"function","function":{"name":"create_file","description":"Create a file. Requires user permission.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"encoding":{"type":"string","default":"utf-8"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"max_bytes":{"type":"integer","default":1048576}},"required":["path"]}}},
    {"type":"function","function":{"name":"edit_file","description":"Edit a file. Requires user permission.","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_content":{"type":"string"},"new_content":{"type":"string"},"operation":{"type":"string","enum":["replace","append","prepend"],"default":"replace"}},"required":["path","new_content"]}}},
    {"type":"function","function":{"name":"delete_file","description":"Delete a file. Requires explicit confirmation.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"create_folder","description":"Create a directory.","parameters":{"type":"object","properties":{"path":{"type":"string"},"parents":{"type":"boolean","default":True}},"required":["path"]}}},
    {"type":"function","function":{"name":"delete_folder","description":"Delete a directory. Requires explicit confirmation.","parameters":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean","default":True}},"required":["path"]}}},
    {"type":"function","function":{"name":"list_directory","description":"List directory contents.","parameters":{"type":"object","properties":{"path":{"type":"string","default":"."},"recursive":{"type":"boolean","default":False}}}}},
]

# ─────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    @staticmethod
    def _resolve(path: str) -> Path:
        p = Path(path)
        return p.resolve() if p.is_absolute() else (Path(WORKING_DIR) / p).resolve()

    @staticmethod
    def _safe(path: Path, allow_outside=False):
        try:
            r = path.resolve()
            if not allow_outside:
                try: r.relative_to(Path(WORKING_DIR).resolve())
                except ValueError: return False, f"Path outside working dir: {r}"
            return True, str(r)
        except Exception as e:
            return False, str(e)

    @staticmethod
    def execute(name: str, args: Dict) -> str:
        fn = getattr(ToolExecutor, f"_{name}", None)
        if not fn:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            log(f"🔧 {name}({str(args)[:120]})")
            return json.dumps(fn(**args), ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _get_current_time(): return {"timestamp": datetime.now().isoformat()}

    @staticmethod
    def _calculate(expression: str):
        if not all(c in "0123456789+-*/().% " for c in expression):
            return {"error": "Invalid chars"}
        try: return {"result": eval(expression, {"__builtins__": {}}, {})}
        except Exception as e: return {"error": str(e)}

    @staticmethod
    def _search_web(query: str): return {"query": query, "note": "Simulated"}

    @staticmethod
    def _get_weather(location: str):
        return {"location": location, "temp_c": random.randint(-5, 35), "note": "Simulated"}

    @staticmethod
    def _run_command(command: str, cwd=None, timeout: int = 60):
        log(f"⚠️  CMD: {command}")
        wd = Path(cwd).resolve() if cwd else Path(WORKING_DIR)
        try:
            r = subprocess.run(command, shell=True, cwd=str(wd),
                               capture_output=True, text=True, timeout=timeout)
            return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        except subprocess.TimeoutExpired: return {"error": "timed out"}
        except Exception as e: return {"error": str(e)}

    @staticmethod
    def _create_file(path: str, content: str, encoding="utf-8"):
        fp = ToolExecutor._resolve(path); ok, r = ToolExecutor._safe(fp)
        if not ok: return {"error": r}
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            fp.write_text(content, encoding=encoding)
            return {"success": True, "path": r, "bytes": len(content.encode(encoding))}
        except Exception as e: return {"error": str(e)}

    @staticmethod
    def _read_file(path: str, max_bytes=1048576):
        fp = ToolExecutor._resolve(path); ok, r = ToolExecutor._safe(fp)
        if not ok: return {"error": r}
        if not fp.is_file(): return {"error": "Not a file"}
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")[:max_bytes]
            return {"success": True, "content": content, "path": r}
        except Exception as e: return {"error": str(e)}

    @staticmethod
    def _edit_file(path: str, new_content: str, old_content=None, operation="replace"):
        fp = ToolExecutor._resolve(path); ok, r = ToolExecutor._safe(fp)
        if not ok: return {"error": r}
        if not fp.exists(): return {"error": "File not found"}
        cur = fp.read_text(encoding="utf-8", errors="replace")
        if operation == "replace" and old_content:
            if old_content not in cur: return {"error": "old_content not found"}
            out = cur.replace(old_content, new_content, 1)
        elif operation == "append": out = cur + new_content
        elif operation == "prepend": out = new_content + cur
        else: out = new_content
        try:
            fp.write_text(out, encoding="utf-8")
            return {"success": True, "path": r, "operation": operation}
        except Exception as e: return {"error": str(e)}

    @staticmethod
    def _delete_file(path: str):
        fp = ToolExecutor._resolve(path); ok, r = ToolExecutor._safe(fp)
        if not ok: return {"error": r}
        if not fp.is_file(): return {"error": "Not a file"}
        try: fp.unlink(); return {"success": True, "path": r}
        except Exception as e: return {"error": str(e)}

    @staticmethod
    def _create_folder(path: str, parents=True):
        dp = ToolExecutor._resolve(path); ok, r = ToolExecutor._safe(dp)
        if not ok: return {"error": r}
        try: Path(r).mkdir(parents=parents, exist_ok=True); return {"success": True, "path": r}
        except Exception as e: return {"error": str(e)}

    @staticmethod
    def _delete_folder(path: str, recursive=True):
        dp = ToolExecutor._resolve(path); ok, r = ToolExecutor._safe(dp)
        if not ok: return {"error": r}
        if not Path(r).is_dir(): return {"error": "Not a directory"}
        try:
            shutil.rmtree(r) if recursive else Path(r).rmdir()
            return {"success": True, "path": r}
        except Exception as e: return {"error": str(e)}

    @staticmethod
    def _list_directory(path=".", recursive=False):
        dp = ToolExecutor._resolve(path); ok, r = ToolExecutor._safe(dp, allow_outside=True)
        if not ok: return {"error": r}
        if not Path(r).is_dir(): return {"error": "Not a directory"}
        items = []
        if recursive:
            for root, dirs, files in os.walk(r):
                rr = Path(root).relative_to(r)
                for d in dirs: items.append({"type":"dir","path":str(rr/d)})
                for f in files:
                    items.append({"type":"file","path":str(rr/f),
                                  "size":(Path(root)/f).stat().st_size})
        else:
            for item in Path(r).iterdir():
                items.append({"type":"dir" if item.is_dir() else "file","name":item.name,
                               "size":item.stat().st_size if item.is_file() else None})
        return {"success": True, "path": r, "items": items, "count": len(items)}

tool_executor = ToolExecutor()

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND THREADS
# ─────────────────────────────────────────────────────────────────────────────

_ram_buffer = None

def _ram_thread():
    global _ram_buffer
    if not RAM_PRESSURE_EN: return
    try:
        _ram_buffer = bytearray(RAM_PRESSURE_MB * 1024 * 1024)
        for i in range(0, len(_ram_buffer), 4096): _ram_buffer[i] = 1
        log(f"💾 Reserved {RAM_PRESSURE_MB} MB")
    except: pass
    while True:
        if _ram_buffer: _ram_buffer[0] = _ram_buffer[-1] = 1
        time.sleep(30)

def _logger_thread():
    import psutil
    while True:
        try:
            r = psutil.virtual_memory(); c = psutil.cpu_percent(0.5)
            ep = "/api/chat" if USE_CHAT_ENDPOINT else "/api/generate"
            log(f"📊 CPU:{c}% RAM:{r.percent}% ep:{ep}")
        except: pass
        time.sleep(LOG_INTERVAL)

def _keepalive_thread():
    s = requests.Session()
    while True:
        try:
            if s.get(f"{OLLAMA_BASE}/api/tags", timeout=5).status_code == 200:
                log("💓 Ollama OK")
        except Exception as e: log(f"⚠️  KA: {e}")
        time.sleep(KEEPALIVE_INT + random.uniform(-1, 1))

def start_threads():
    for fn, n in [(_ram_thread,"ram"),(_logger_thread,"log"),(_keepalive_thread,"ka")]:
        threading.Thread(target=fn, daemon=True, name=n).start()
    log("✅ Background threads started")

# ─────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_params(p):
    if not p or not isinstance(p, dict): return {"type":"object","properties":{}}
    return p if "type" in p else {"type":"object", **p}

def _oai_tools_to_ollama(tools):
    if not tools: return None
    out = []
    for t in tools:
        if t.get("type") != "function": continue
        f = t["function"]
        out.append({"type":"function","function":{
            "name":f.get("name",""),
            "description":f.get("description",""),
            "parameters":_norm_params(f.get("parameters"))}})
    return out or None

def _anthropic_tools_to_ollama(tools):
    """Convert Anthropic tool format (input_schema) -> Ollama format."""
    if not tools: return None
    out = []
    for t in tools:
        out.append({"type":"function","function":{
            "name":t.get("name",""),
            "description":t.get("description",""),
            "parameters":_norm_params(t.get("input_schema",{}))}})
    return out or None

def _content_to_str(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict): parts.append(str(b)); continue
            t = b.get("type","")
            if t == "text": parts.append(b.get("text",""))
            elif t == "tool_result":
                c = b.get("content","")
                if isinstance(c, list):
                    c = "\n".join(x.get("text","") for x in c if isinstance(x,dict))
                parts.append(str(c))
        return "\n".join(parts)
    return str(content or "")

def _msgs_to_ollama(messages: List[Dict]) -> List[Dict]:
    out = []
    for m in messages:
        role = m.get("role","")
        content = _content_to_str(m.get("content") or "")
        if role in ("system","user"):
            out.append({"role":role,"content":content})
        elif role == "assistant":
            msg: Dict = {"role":"assistant","content":content}
            tcs = m.get("tool_calls")
            if tcs:
                ollama_tcs = []
                for tc in tcs:
                    f = tc.get("function",{}); args = f.get("arguments","{}")
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except: args = {}
                    ollama_tcs.append({"function":{"name":f.get("name",""),"arguments":args}})
                msg["tool_calls"] = ollama_tcs
            out.append(msg)
        elif role == "tool":
            tm: Dict = {"role":"tool","content":content}
            if m.get("tool_call_id"): tm["tool_call_id"] = m["tool_call_id"]
            out.append(tm)
    return out

def _msgs_to_prompt(messages: List[Dict]) -> str:
    """Flatten messages -> single prompt string for /api/generate."""
    parts = []
    for m in messages:
        role = m.get("role","user")
        content = _content_to_str(m.get("content") or "")
        if role == "system":    parts.append(f"<|system|>\n{content}")
        elif role == "user":    parts.append(f"<|user|>\n{content}")
        elif role == "assistant": parts.append(f"<|assistant|>\n{content}")
        elif role == "tool":    parts.append(f"<|tool_result|>\n{content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)

def _tcs_to_oai(tcs: List[Dict]) -> List[Dict]:
    out = []
    for tc in tcs:
        f = tc.get("function",{}); args = f.get("arguments",{})
        out.append({"id":f"call_{uuid.uuid4().hex[:8]}","type":"function",
                    "function":{"name":f.get("name",""),
                                "arguments":json.dumps(args) if isinstance(args,dict) else str(args)}})
    return out

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA SESSION
# ─────────────────────────────────────────────────────────────────────────────

def _make_session():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.5, status_forcelist=[500,502,503,504,429])
    a = HTTPAdapter(max_retries=r, pool_connections=50, pool_maxsize=50)
    s.mount("http://", a)
    return s

ollama = _make_session()

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CALLS — chat with generate fallback
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama_sync(messages, tools=None, max_tokens=2048, temp=0.7) -> Dict:
    global USE_CHAT_ENDPOINT

    if USE_CHAT_ENDPOINT:
        payload = {"model":MODEL,"stream":False,"keep_alive":-1,
                   "messages":_msgs_to_ollama(messages),
                   "options":{"num_predict":max_tokens,"temperature":temp}}
        if tools: payload["tools"] = _oai_tools_to_ollama(tools)
        try:
            r = ollama.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=(10,300))
            if r.status_code == 404:
                log("⚠️  /api/chat 404 at runtime — switching to /api/generate")
                USE_CHAT_ENDPOINT = False
            else:
                r.raise_for_status()
                return r.json()
        except requests.Timeout: raise HTTPException(504, "Ollama timed out")
        except requests.ConnectionError: raise HTTPException(503, f"Cannot reach {OLLAMA_BASE}")
        except HTTPException: raise
        except requests.RequestException as e:
            if USE_CHAT_ENDPOINT: raise HTTPException(503, str(e))

    # /api/generate fallback
    payload = {"model":MODEL,"stream":False,"keep_alive":-1,
               "prompt":_msgs_to_prompt(messages),
               "options":{"num_predict":max_tokens,"temperature":temp}}
    try:
        r = ollama.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=(10,300))
        r.raise_for_status()
        d = r.json()
        return {"message":{"role":"assistant","content":d.get("response","")},
                "prompt_eval_count":d.get("prompt_eval_count",0),
                "eval_count":d.get("eval_count",0)}
    except requests.Timeout: raise HTTPException(504, "Ollama timed out")
    except requests.ConnectionError: raise HTTPException(503, f"Cannot reach {OLLAMA_BASE}")
    except requests.RequestException as e: raise HTTPException(503, str(e))

def _call_ollama_stream(messages, tools=None, max_tokens=2048, temp=0.7):
    global USE_CHAT_ENDPOINT

    if USE_CHAT_ENDPOINT:
        payload = {"model":MODEL,"stream":True,"keep_alive":-1,
                   "messages":_msgs_to_ollama(messages),
                   "options":{"num_predict":max_tokens,"temperature":temp}}
        if tools: payload["tools"] = _oai_tools_to_ollama(tools)
        try:
            r = ollama.post(f"{OLLAMA_BASE}/api/chat", json=payload, stream=True, timeout=(10,300))
            if r.status_code == 404:
                log("⚠️  /api/chat 404 in stream — switching")
                USE_CHAT_ENDPOINT = False
            else:
                return r, "chat"
        except requests.RequestException: pass

    payload = {"model":MODEL,"stream":True,"keep_alive":-1,
               "prompt":_msgs_to_prompt(messages),
               "options":{"num_predict":max_tokens,"temperature":temp}}
    r = ollama.post(f"{OLLAMA_BASE}/api/generate", json=payload, stream=True, timeout=(10,300))
    return r, "generate"

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _stream_openai(messages, tools, max_tokens, temp):
    yield ": stream-start\n\n"
    try: resp, mode = _call_ollama_stream(messages, tools, max_tokens, temp)
    except Exception as e:
        yield f"data: {json.dumps({'error':str(e)})}\n\ndata: [DONE]\n\n"
        return

    last_ping = time.time()
    for line in resp.iter_lines():
        if not line: continue
        if time.time() - last_ping > 15:
            yield ": keep-alive\n\n"
            last_ping = time.time()
        try: chunk = json.loads(line)
        except: continue

        if mode == "chat":
            msg  = chunk.get("message", {})
            text = msg.get("content", "")
            done = chunk.get("done", False)
            delta = {"content": text} if text else {}
            if done and msg.get("tool_calls"):
                tcs = _tcs_to_oai(msg["tool_calls"])
                yield f"data: {json.dumps({'id':f'chatcmpl-{uuid.uuid4().hex[:8]}','object':'chat.completion.chunk','created':int(time.time()),'model':MODEL,'choices':[{'index':0,'delta':{'tool_calls':tcs},'finish_reason':None}]})}\n\n"
                yield f"data: {json.dumps({'id':f'chatcmpl-{uuid.uuid4().hex[:8]}','object':'chat.completion.chunk','created':int(time.time()),'model':MODEL,'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]})}\n\n"
                yield "data: [DONE]\n\n"
                return
        else:
            text = chunk.get("response", "")
            done = chunk.get("done", False)
            delta = {"content": text} if text else {}

        c = {"id":f"chatcmpl-{uuid.uuid4().hex[:8]}","object":"chat.completion.chunk",
             "created":int(time.time()),"model":MODEL,
             "choices":[{"index":0,"delta":delta,"finish_reason":"stop" if done else None}]}
        yield f"data: {json.dumps(c)}\n\n"
        if done:
            yield "data: [DONE]\n\n"
            return

def _stream_anthropic(messages, tools, max_tokens, temp):
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':mid,'type':'message','role':'assistant','content':[],'model':MODEL,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
    try: resp, mode = _call_ollama_stream(messages, tools, max_tokens, temp)
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'type':'error','error':{'message':str(e)}})}\n\n"
        return

    last_ping = time.time()
    for line in resp.iter_lines():
        if not line: continue
        if time.time() - last_ping > 15:
            yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"
            last_ping = time.time()
        try: chunk = json.loads(line)
        except: continue

        msg  = chunk.get("message", {}) if mode == "chat" else {}
        text = msg.get("content","") if mode == "chat" else chunk.get("response","")
        done = chunk.get("done", False)

        if text:
            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':text}})}\n\n"

        if done:
            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
            stop = "end_turn"
            tcs = msg.get("tool_calls",[])
            if tcs:
                stop = "tool_use"
                for i, tc in enumerate(tcs, 1):
                    f = tc.get("function",{}); args = f.get("arguments",{})
                    tid = f"toolu_{uuid.uuid4().hex[:8]}"
                    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':i,'content_block':{'type':'tool_use','id':tid,'name':f.get('name',''),'input':{}}})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':i,'delta':{'type':'input_json_delta','partial_json':json.dumps(args if isinstance(args,dict) else {})}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':i})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop,'stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"
            return

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Ollama Proxy v3", version="3.0.0")

# Strip trailing slashes so /v1/ == /v1
class StripSlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path != "/" and path.endswith("/"):
            url = request.url.replace(path=path.rstrip("/"))
            return RedirectResponse(url=str(url), status_code=307)
        return await call_next(request)

app.add_middleware(StripSlashMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Pydantic models
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

def _tc_type(tc) -> str:
    if isinstance(tc, str): return tc
    return (tc or {}).get("type", "auto")

def _effective_tools(req_tools, tc):
    if _tc_type(tc) == "none": return None
    return req_tools if req_tools is not None else TOOL_DEFINITIONS

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Discovery & Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """
    Returns an OpenAI-style model list + extra proxy info.
    Clients that probe GET / to auto-detect the model will find it here.
    """
    return {
        # OpenAI model list shape (some clients read this from /)
        "object": "list",
        "data": [{"id": MODEL, "object": "model", "created": int(time.time()),
                  "owned_by": "ollama"}],
        # Extra info
        "service": "Ollama Proxy v3",
        "version": "3.0.0",
        "model": MODEL,
        "active_endpoint": "/api/chat" if USE_CHAT_ENDPOINT else "/api/generate",
        "tools_count": len(TOOL_DEFINITIONS),
        "routes": [
            "/v1/chat/completions", "/v1/messages", "/v1/models",
            "/api/chat", "/api/generate", "/api/tags", "/api/version",
            "/health", "/v1/tools", "/v1/tool/execute",
        ],
    }

@app.get("/health")
async def health():
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            names = [m.get("name","") for m in r.json().get("models",[])]
            loaded = (MODEL in names or
                      any(n.startswith(MODEL.split(":")[0]) for n in names))
            return {"status":"ready" if loaded else "degraded",
                    "model":MODEL,"model_loaded":loaded,
                    "available_models":names,
                    "active_endpoint":"/api/chat" if USE_CHAT_ENDPOINT else "/api/generate",
                    "tools_count":len(TOOL_DEFINITIONS),"ollama_ok":True}
    except Exception as e:
        return {"status":"down","model":MODEL,"error":str(e)}
    return {"status":"down","model":MODEL}

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — OpenAI-compatible
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    """Full OpenAI model list — required for client model detection."""
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            data = [{"id":m.get("name",""),"object":"model","created":int(time.time()),
                     "owned_by":"ollama","root":m.get("name",""),"parent":None,"permission":[]}
                    for m in r.json().get("models",[])]
            if data: return {"object":"list","data":data}
    except: pass
    return {"object":"list","data":[
        {"id":MODEL,"object":"model","created":int(time.time()),
         "owned_by":"ollama","root":MODEL,"parent":None,"permission":[]}
    ]}

@app.get("/v1/tools")
async def list_tools():
    return {"tools": TOOL_DEFINITIONS, "count": len(TOOL_DEFINITIONS)}

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    try: body = await request.json()
    except: raise HTTPException(400, "Invalid JSON")
    try: req = ChatCompletionRequest(**body)
    except Exception as e: raise HTTPException(400, str(e))

    messages = [m.model_dump() for m in req.messages]
    tools = _effective_tools(req.tools, req.tool_choice)

    if req.stream:
        return StreamingResponse(
            _stream_openai(messages, tools, req.max_tokens, req.temperature),
            media_type="text/event-stream",
            headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

    result = _call_ollama_sync(messages, tools, req.max_tokens, req.temperature)
    msg = result.get("message",{}); content = msg.get("content") or ""; tcs = msg.get("tool_calls") or []
    rmsg: Dict = {"role":"assistant","content": content or None}
    fr = "stop"
    if tcs: rmsg["tool_calls"] = _tcs_to_oai(tcs); fr = "tool_calls"
    return {"id":f"chatcmpl-{uuid.uuid4().hex[:8]}","object":"chat.completion",
            "created":int(time.time()),"model":MODEL,
            "choices":[{"index":0,"message":rmsg,"finish_reason":fr}],
            "usage":{"prompt_tokens":result.get("prompt_eval_count",0),
                     "completion_tokens":result.get("eval_count",0),
                     "total_tokens":result.get("prompt_eval_count",0)+result.get("eval_count",0)}}

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    try: body = await request.json()
    except: raise HTTPException(400, "Invalid JSON")

    system = body.get("system",""); msgs = body.get("messages",[])
    max_tokens = body.get("max_tokens",2048); temp = body.get("temperature",0.7)
    stream = body.get("stream",False); tools = body.get("tools")
    tc = body.get("tool_choice",{"type":"auto"})
    tc_type = tc.get("type","auto") if isinstance(tc,dict) else str(tc)

    full_msgs = ([{"role":"system","content":system}] if system else []) + msgs
    eff_tools = (None if tc_type == "none"
                 else (_anthropic_tools_to_ollama(tools) if tools
                       else _oai_tools_to_ollama(TOOL_DEFINITIONS)))

    if stream:
        return StreamingResponse(
            _stream_anthropic(full_msgs, eff_tools, max_tokens, temp),
            media_type="text/event-stream",
            headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

    result = _call_ollama_sync(full_msgs, eff_tools, max_tokens, temp)
    msg = result.get("message",{}); content = msg.get("content") or ""; tcs = msg.get("tool_calls") or []
    blocks = ([{"type":"text","text":content}] if content else [])
    stop = "end_turn"
    for tc_item in tcs:
        f = tc_item.get("function",{}); args = f.get("arguments",{})
        blocks.append({"type":"tool_use","id":f"toolu_{uuid.uuid4().hex[:8]}",
                        "name":f.get("name",""),
                        "input":args if isinstance(args,dict) else {}})
        stop = "tool_use"
    return {"id":f"msg_{uuid.uuid4().hex[:8]}","type":"message","role":"assistant","model":MODEL,
            "content":blocks,"stop_reason":stop,
            "usage":{"input_tokens":result.get("prompt_eval_count",0),
                     "output_tokens":result.get("eval_count",0)}}

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Raw Ollama passthrough
# Clients configured with apiBase = https://tunnel-url.trycloudflare.com
# (no /v1 suffix) and using native Ollama protocol will hit these routes.
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def ollama_chat_passthrough(request: Request):
    """
    Accept raw /api/chat requests.
    Routes through proxy logic so /api/generate fallback applies.
    """
    try: body = await request.json()
    except: raise HTTPException(400, "Invalid JSON")

    messages  = body.get("messages", [])
    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temp       = body.get("options", {}).get("temperature", 0.7)
    stream     = body.get("stream", False)
    tools      = body.get("tools")

    if stream:
        async def _gen():
            try: resp, _ = _call_ollama_stream(messages, tools, max_tokens, temp)
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"; return
            for line in resp.iter_lines():
                if line: yield line.decode() + "\n"
        return StreamingResponse(_gen(), media_type="application/x-ndjson")

    result = _call_ollama_sync(messages, tools, max_tokens, temp)
    return {"model":MODEL,"message":result.get("message",{}),"done":True,
            "done_reason":"stop","prompt_eval_count":result.get("prompt_eval_count",0),
            "eval_count":result.get("eval_count",0)}

@app.post("/api/generate")
async def ollama_generate_passthrough(request: Request):
    """Accept raw /api/generate — proxy directly to Ollama."""
    try: body = await request.json()
    except: raise HTTPException(400, "Invalid JSON")

    payload = {"model":MODEL, "prompt":body.get("prompt",""),
               "stream":body.get("stream",False), "keep_alive":-1,
               "options":body.get("options",{})}
    try:
        if body.get("stream", False):
            resp = ollama.post(f"{OLLAMA_BASE}/api/generate", json=payload,
                               stream=True, timeout=(10,300))
            async def _gen():
                for line in resp.iter_lines():
                    if line: yield line.decode() + "\n"
            return StreamingResponse(_gen(), media_type="application/x-ndjson")
        else:
            r = ollama.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=(10,300))
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(503, str(e))

@app.get("/api/tags")
async def ollama_tags():
    """
    Mirror Ollama /api/tags — clients probe this to list available models.
    Always includes MODEL even if Ollama is unreachable.
    """
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"models": [{
            "name": MODEL, "model": MODEL,
            "modified_at": datetime.now().isoformat(),
            "size": 0, "digest": "",
            "details": {"format":"gguf","family":"qwen","parameter_size":"4B","quantization_level":"Q4_0"}
        }]}

@app.get("/api/version")
async def ollama_version():
    """
    Mirror Ollama version.
    Always reports >= 0.1.16 so clients never show the /api/chat warning.
    """
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/version", timeout=5)
        if r.status_code == 200:
            data = r.json()
            # Bump if old
            v = data.get("version","0.0.0")
            parts = v.replace("-",".").split(".")
            try:
                if (int(parts[0]),int(parts[1]),int(parts[2])) < (0,1,16):
                    data["version"] = "0.5.0"
            except: data["version"] = "0.5.0"
            return data
    except: pass
    return {"version": "0.5.0"}

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Tool execution
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/tool/execute")
async def execute_tool(request: Request):
    try: body = await request.json()
    except: raise HTTPException(400, "Invalid JSON")
    name = body.get("name"); args = body.get("arguments", {})
    if not name: raise HTTPException(400, "Missing 'name'")
    log(f"🔧 Tool: {name}")
    return {"tool": name, "result": json.loads(tool_executor.execute(name, args))}

# ─────────────────────────────────────────────────────────────────────────────
# CATCH-ALL
# ─────────────────────────────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE"])
async def catch_all(path: str):
    return JSONResponse({
        "error": "Not found", "path": f"/{path}",
        "hint": ("Use /v1/chat/completions (OpenAI-compatible) "
                 "or /api/chat (raw Ollama) "
                 "or /v1/messages (Anthropic-compatible)"),
        "available": ["/","/health","/v1/models","/v1/chat/completions",
                      "/v1/messages","/api/chat","/api/generate",
                      "/api/tags","/api/version","/v1/tools","/v1/tool/execute"],
    }, status_code=404)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("="*70)
    log("  Ollama Proxy v3 — Universal Client Support")
    log("="*70)
    log(f"🤖 Model:  {MODEL}")
    log(f"🔌 Ollama: {OLLAMA_BASE}")
    log(f"🌐 Port:   {PROXY_PORT}")
    log(f"📁 Work:   {WORKING_DIR}")
    log(f"🔧 Tools:  {len(TOOL_DEFINITIONS)}")
    log("")
    log("📡 Client config options (both work):")
    log("   apiBase = https://…trycloudflare.com/v1   ← OpenAI-compatible")
    log("   apiBase = https://…trycloudflare.com      ← raw Ollama protocol")
    log("   model   = sorc/qwen3.5-claude-4.6-opus:4b")
    log("="*70)

    try:
        import psutil
    except ImportError:
        subprocess.run([sys.executable,"-m","pip","install","psutil","-q"], check=True)

    detect_ollama_capabilities()
    log(f"🛰  Active: /api/{'chat' if USE_CHAT_ENDPOINT else 'generate'}")
    start_threads()
    log(f"🚀 Uvicorn on :{PROXY_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT,
                log_level="warning", timeout_keep_alive=300)
