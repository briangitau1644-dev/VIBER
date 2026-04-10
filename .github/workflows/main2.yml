#!/usr/bin/env python3
"""
Ollama Proxy Server with Full Tool Calling Support + System Operations
GitHub Actions Edition - CPU Optimized
Supports: OpenAI /v1/chat/completions + Anthropic /v1/messages
System Tools: run_command, file operations, folder management
⚠️  Permission handling is delegated to the client application

FIXES:
- Added /api/generate fallback when /api/chat is unavailable
- Added missing _anthropic_tools_to_ollama() function
- Improved model detection for clients (model ID normalisation)
- Better Ollama version detection and endpoint selection
"""

import os, sys, json, time, uuid, threading, random, subprocess, requests, shutil
from datetime import datetime
from pathlib import Path
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
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
OLLAMA_BASE = f"http://127.0.0.1:{OLLAMA_PORT}".rstrip('/')
PROXY_PORT = int(os.getenv('PROXY_PORT', '8000'))
RAM_PRESSURE_ENABLED = os.getenv('RAM_PRESSURE_ENABLED', 'true').lower() == 'true'
RAM_PRESSURE_MB = int(os.getenv('RAM_PRESSURE_MB', '256'))
ACTIVITY_LOG_INTERVAL = float(os.getenv('ACTIVITY_LOG_INTERVAL', '30'))
KEEPALIVE_INTERVAL = float(os.getenv('KEEPALIVE_INTERVAL', '10'))
WORKING_DIR = os.getenv('WORKING_DIR', os.getcwd())

# Runtime flag: set to True once we confirm /api/chat works
USE_CHAT_ENDPOINT = True   # will be re-evaluated at startup

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"⚡ Proxy Starting - Model: {MODEL}")
log(f"🔌 Ollama Base: {OLLAMA_BASE}")
log(f"📁 Working Directory: {WORKING_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA ENDPOINT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_ollama_capabilities() -> bool:
    """
    Probe Ollama to decide whether /api/chat is available.
    Falls back to /api/generate (legacy) if not.
    Returns True if /api/chat is usable.
    """
    global USE_CHAT_ENDPOINT
    # First check version string
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5)
        if resp.status_code == 200:
            version_str = resp.json().get("version", "0.0.0")
            log(f"📦 Ollama version: {version_str}")
            parts = version_str.replace("-", ".").split(".")
            try:
                major = int(parts[0])
                minor = int(parts[1])
                patch = int(parts[2])
                if (major, minor, patch) >= (0, 1, 16):
                    log("✅ Ollama >= 0.1.16: /api/chat supported")
                    USE_CHAT_ENDPOINT = True
                    return True
                else:
                    log(f"⚠️  Ollama {version_str} < 0.1.16: falling back to /api/generate")
                    USE_CHAT_ENDPOINT = False
                    return False
            except (IndexError, ValueError):
                pass
    except Exception as e:
        log(f"⚠️  Version check failed: {e}")

    # Live-probe /api/chat with a minimal no-op request
    try:
        probe = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": MODEL, "stream": False, "messages": [{"role": "user", "content": "hi"}]},
            timeout=15,
        )
        if probe.status_code == 200:
            log("✅ /api/chat probe succeeded")
            USE_CHAT_ENDPOINT = True
            return True
        elif probe.status_code == 404:
            log("⚠️  /api/chat → 404, switching to /api/generate")
            USE_CHAT_ENDPOINT = False
            return False
        else:
            log(f"⚠️  /api/chat probe returned {probe.status_code}, assuming supported")
            USE_CHAT_ENDPOINT = True
            return True
    except Exception as e:
        log(f"⚠️  /api/chat probe error: {e}. Defaulting to /api/generate for safety")
        USE_CHAT_ENDPOINT = False
        return False

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": "get_current_time", "description": "Get current date/time in ISO format.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "calculate", "description": "Perform math: +, -, *, /, **, %, //", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search for information (simulated).", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for location (simulated).", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Execute a shell command. ⚠️ Client must request user permission before calling.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string", "default": None},
            "timeout": {"type": "integer", "default": 60}
        }, "required": ["command"]}
    }},
    {"type": "function", "function": {
        "name": "create_file",
        "description": "Create a new file with content. ⚠️ Client must request user permission.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "encoding": {"type": "string", "default": "utf-8"}
        }, "required": ["path", "content"]}
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read contents of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer", "default": 1048576}
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Edit a file by replacing content. ⚠️ Client must request user permission.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_content": {"type": "string", "default": None},
            "new_content": {"type": "string"},
            "operation": {"type": "string", "enum": ["replace", "append", "prepend"], "default": "replace"}
        }, "required": ["path", "new_content"]}
    }},
    {"type": "function", "function": {
        "name": "delete_file",
        "description": "Delete a file. ⚠️ Client must confirm with user.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "create_folder",
        "description": "Create a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "parents": {"type": "boolean", "default": True}
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "delete_folder",
        "description": "Delete a directory. ⚠️ Client must confirm with user.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "recursive": {"type": "boolean", "default": True}
        }, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and folders in a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "default": "."},
            "recursive": {"type": "boolean", "default": False}
        }}
    }},
]

# ─────────────────────────────────────────────────────────────────────────────
# TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (Path(WORKING_DIR) / p).resolve()

    @staticmethod
    def _safe_path(path: Path, allow_outside: bool = False):
        try:
            resolved = path.resolve()
            if not allow_outside:
                work = Path(WORKING_DIR).resolve()
                try:
                    resolved.relative_to(work)
                except ValueError:
                    return False, f"Path outside working directory: {resolved}"
            return True, str(resolved)
        except Exception as e:
            return False, f"Path error: {e}"

    @staticmethod
    def execute(name: str, arguments: Dict[str, Any]) -> str:
        try:
            fn = getattr(ToolExecutor, f"_{name}", None)
            if fn is None:
                return json.dumps({"error": f"Unknown tool: {name}"})
            log(f"🔧 {name}({json.dumps(arguments, default=str)[:150]})")
            return json.dumps(fn(**arguments), ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _get_current_time(): return {"timestamp": datetime.now().isoformat()}
    @staticmethod
    def _calculate(expression: str):
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters"}
        try:
            return {"result": eval(expression, {"__builtins__": {}}, {})}
        except Exception as e:
            return {"error": str(e)}
    @staticmethod
    def _search_web(query: str): return {"query": query, "note": "Simulated"}
    @staticmethod
    def _get_weather(location: str): return {"location": location, "temperature_c": random.randint(-10, 35), "note": "Simulated"}

    @staticmethod
    def _run_command(command: str, cwd=None, timeout: int = 60):
        log(f"⚠️  COMMAND: {command}")
        try:
            wd = Path(cwd).resolve() if cwd else Path(WORKING_DIR)
            r = subprocess.run(command, shell=True, cwd=str(wd), capture_output=True, text=True, timeout=timeout, env={**os.environ})
            return {"command": command, "exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
        except subprocess.TimeoutExpired as e:
            return {"error": "Timed out", "command": command}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _create_file(path: str, content: str, encoding: str = "utf-8"):
        fp = ToolExecutor._resolve_path(path)
        ok, resolved = ToolExecutor._safe_path(fp)
        if not ok: return {"error": resolved}
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved, 'w', encoding=encoding) as f: f.write(content)
            return {"success": True, "path": resolved, "bytes_written": len(content.encode(encoding))}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _read_file(path: str, max_bytes: int = 1048576):
        fp = ToolExecutor._resolve_path(path)
        ok, resolved = ToolExecutor._safe_path(fp)
        if not ok: return {"error": resolved}
        try:
            if not Path(resolved).is_file(): return {"error": "Not a file"}
            with open(resolved, 'r', encoding='utf-8', errors='replace') as f: content = f.read(max_bytes)
            return {"success": True, "path": resolved, "content": content, "truncated": len(content) >= max_bytes}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _edit_file(path: str, new_content: str, old_content=None, operation: str = "replace"):
        fp = ToolExecutor._resolve_path(path)
        ok, resolved = ToolExecutor._safe_path(fp)
        if not ok: return {"error": resolved}
        try:
            if not Path(resolved).exists(): return {"error": "File not found"}
            with open(resolved, 'r', encoding='utf-8', errors='replace') as f: current = f.read()
            if operation == "replace" and old_content:
                if old_content not in current: return {"error": "old_content not found"}
                updated = current.replace(old_content, new_content, 1)
            elif operation == "append": updated = current + new_content
            elif operation == "prepend": updated = new_content + current
            else: updated = new_content
            with open(resolved, 'w', encoding='utf-8') as f: f.write(updated)
            return {"success": True, "path": resolved, "operation": operation}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _delete_file(path: str):
        fp = ToolExecutor._resolve_path(path)
        ok, resolved = ToolExecutor._safe_path(fp)
        if not ok: return {"error": resolved}
        try:
            if not Path(resolved).is_file(): return {"error": "Not a file"}
            Path(resolved).unlink()
            return {"success": True, "path": resolved}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _create_folder(path: str, parents: bool = True):
        dp = ToolExecutor._resolve_path(path)
        ok, resolved = ToolExecutor._safe_path(dp)
        if not ok: return {"error": resolved}
        try:
            Path(resolved).mkdir(parents=parents, exist_ok=True)
            return {"success": True, "path": resolved}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _delete_folder(path: str, recursive: bool = True):
        dp = ToolExecutor._resolve_path(path)
        ok, resolved = ToolExecutor._safe_path(dp)
        if not ok: return {"error": resolved}
        try:
            if not Path(resolved).is_dir(): return {"error": "Not a directory"}
            shutil.rmtree(resolved) if recursive else Path(resolved).rmdir()
            return {"success": True, "path": resolved}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _list_directory(path: str = ".", recursive: bool = False):
        dp = ToolExecutor._resolve_path(path)
        ok, resolved = ToolExecutor._safe_path(dp, allow_outside=True)
        if not ok: return {"error": resolved}
        try:
            if not Path(resolved).is_dir(): return {"error": "Not a directory"}
            items = []
            if recursive:
                for root, dirs, files in os.walk(resolved):
                    rr = Path(root).relative_to(resolved)
                    for d in dirs: items.append({"type": "directory", "path": str(rr / d)})
                    for f in files: items.append({"type": "file", "path": str(rr / f), "size": (Path(root) / f).stat().st_size})
            else:
                for item in Path(resolved).iterdir():
                    items.append({"type": "directory" if item.is_dir() else "file", "name": item.name, "size": item.stat().st_size if item.is_file() else None})
            return {"success": True, "path": resolved, "items": items, "count": len(items)}
        except Exception as e:
            return {"error": str(e)}

tool_executor = ToolExecutor()

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE KEEP-ALIVE
# ─────────────────────────────────────────────────────────────────────────────

_ram_buffer = None

def _ram_pressure_thread():
    global _ram_buffer
    if not RAM_PRESSURE_ENABLED: return
    try:
        _ram_buffer = bytearray(RAM_PRESSURE_MB * 1024 * 1024)
        for i in range(0, len(_ram_buffer), 4096): _ram_buffer[i] = 1
        log(f"💾 Reserved {RAM_PRESSURE_MB} MB RAM")
    except: pass
    while True:
        if _ram_buffer: _ram_buffer[0] = _ram_buffer[-1] = 1
        time.sleep(30)

def _activity_logger():
    import psutil
    while True:
        try:
            ram = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.5)
            log(f"📊 CPU:{cpu}% RAM:{ram.percent}% endpoint:/api/{'chat' if USE_CHAT_ENDPOINT else 'generate'}")
        except: pass
        time.sleep(ACTIVITY_LOG_INTERVAL)

def _ollama_keepalive():
    session = requests.Session()
    while True:
        try:
            resp = session.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            if resp.status_code == 200: log("💓 Ollama OK")
        except Exception as e: log(f"⚠️ Keepalive: {e}")
        time.sleep(KEEPALIVE_INTERVAL + random.uniform(-1, 1))

def _start_background_threads():
    for fn, name in [(_ram_pressure_thread, "ram"), (_activity_logger, "logger"), (_ollama_keepalive, "keepalive")]:
        threading.Thread(target=fn, daemon=True, name=name).start()
    log("✅ Background threads started")

# ─────────────────────────────────────────────────────────────────────────────
# FORMAT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_tool_params(params):
    if not params or not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    return params if "type" in params else {**params, "type": "object"}

def _oai_tools_to_ollama(tools):
    if tools is None: return None
    result = []
    for t in tools:
        if t.get("type") != "function": continue
        fn = t.get("function", {})
        result.append({"type": "function", "function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": _normalize_tool_params(fn.get("parameters"))
        }})
    return result or None

def _anthropic_tools_to_ollama(tools):
    """Convert Anthropic-format tools to Ollama format."""
    if tools is None: return None
    result = []
    for t in tools:
        name = t.get("name", "")
        description = t.get("description", "")
        input_schema = t.get("input_schema", {})
        result.append({"type": "function", "function": {
            "name": name,
            "description": description,
            "parameters": _normalize_tool_params(input_schema)
        }})
    return result or None

def _oai_messages_to_ollama(messages):
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text": parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        c = block.get("content", "")
                        if isinstance(c, list): c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                        parts.append(str(c))
                else: parts.append(str(block))
            content = "\n".join(parts)
        if role in ("system", "user"):
            result.append({"role": role, "content": content})
        elif role == "assistant":
            oai_msg = {"role": "assistant", "content": content}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                ollama_tcs = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except: args = {}
                    ollama_tcs.append({"function": {"name": fn.get("name", ""), "arguments": args}})
                oai_msg["tool_calls"] = ollama_tcs
            result.append(oai_msg)
        elif role == "tool":
            tool_msg = {"role": "tool", "content": str(content)}
            if msg.get("tool_call_id"): tool_msg["tool_call_id"] = msg["tool_call_id"]
            result.append(tool_msg)
    return result

def _messages_to_generate_prompt(messages: List[Dict]) -> str:
    """Flatten chat messages into a single prompt for /api/generate (legacy fallback)."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        elif role == "tool":
            parts.append(f"<|tool_result|>\n{content}")
    parts.append("<|assistant|>")
    return "\n".join(parts)

def _ollama_tool_calls_to_oai(ollama_tcs):
    result = []
    for tc in ollama_tcs:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
        result.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": fn.get("name", ""), "arguments": args_str}})
    return result

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA SESSION
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

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA COMMUNICATION — /api/chat with /api/generate fallback
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_chat_sync(messages, tools=None, max_tokens=2048, temperature=0.7):
    global USE_CHAT_ENDPOINT

    if USE_CHAT_ENDPOINT:
        payload = {
            "model": MODEL, "stream": False, "keep_alive": -1,
            "messages": _oai_messages_to_ollama(messages),
            "options": {"num_predict": max_tokens, "temperature": temperature}
        }
        if tools is not None:
            payload["tools"] = _oai_tools_to_ollama(tools)

        try:
            resp = ollama_session.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=(10, 300))
            if resp.status_code == 404:
                log("⚠️  /api/chat → 404 at runtime. Switching to /api/generate fallback.")
                USE_CHAT_ENDPOINT = False
                # fall through to generate below
            else:
                resp.raise_for_status()
                return resp.json()
        except requests.exceptions.Timeout:
            raise HTTPException(504, "Ollama timed out")
        except requests.exceptions.ConnectionError:
            raise HTTPException(503, f"Cannot connect to Ollama at {OLLAMA_BASE}")
        except HTTPException:
            raise
        except requests.exceptions.RequestException as e:
            if not USE_CHAT_ENDPOINT:
                pass  # fall through
            else:
                raise HTTPException(503, f"Ollama error: {e}")

    # ── /api/generate fallback (no native tool calling) ──
    prompt = _messages_to_generate_prompt(messages)
    payload = {
        "model": MODEL, "stream": False, "keep_alive": -1,
        "prompt": prompt,
        "options": {"num_predict": max_tokens, "temperature": temperature}
    }
    try:
        resp = ollama_session.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=(10, 300))
        resp.raise_for_status()
        data = resp.json()
        # Normalise to /api/chat shape
        return {"message": {"role": "assistant", "content": data.get("response", "")},
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0)}
    except requests.exceptions.Timeout:
        raise HTTPException(504, "Ollama timed out")
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, f"Cannot connect to Ollama at {OLLAMA_BASE}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(503, f"Ollama /api/generate error: {e}")


def _ollama_chat_stream(messages, tools=None, max_tokens=2048, temperature=0.7):
    global USE_CHAT_ENDPOINT

    if USE_CHAT_ENDPOINT:
        payload = {
            "model": MODEL, "stream": True, "keep_alive": -1,
            "messages": _oai_messages_to_ollama(messages),
            "options": {"num_predict": max_tokens, "temperature": temperature}
        }
        if tools is not None:
            payload["tools"] = _oai_tools_to_ollama(tools)
        try:
            resp = ollama_session.post(f"{OLLAMA_BASE}/api/chat", json=payload, stream=True, timeout=(10, 300))
            if resp.status_code == 404:
                log("⚠️  /api/chat → 404 in stream. Switching to /api/generate.")
                USE_CHAT_ENDPOINT = False
            else:
                return resp
        except requests.exceptions.RequestException as e:
            raise

    # Fallback stream via /api/generate
    prompt = _messages_to_generate_prompt(messages)
    payload = {
        "model": MODEL, "stream": True, "keep_alive": -1,
        "prompt": prompt,
        "options": {"num_predict": max_tokens, "temperature": temperature}
    }
    return ollama_session.post(f"{OLLAMA_BASE}/api/generate", json=payload, stream=True, timeout=(10, 300))

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _generate_openai_stream(messages, tools, max_tokens, temperature):
    yield ": stream-start\n\n"
    try:
        resp = _ollama_chat_stream(messages, tools, max_tokens, temperature)
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    last_ping = time.time()
    # detect if we got a /api/generate stream (has "response" key) or /api/chat stream
    for line in resp.iter_lines():
        if not line: continue
        if time.time() - last_ping > 15:
            yield ": keep-alive\n\n"
            last_ping = time.time()
        try:
            chunk = json.loads(line)
        except: continue

        # /api/chat chunk shape
        if "message" in chunk:
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            done = chunk.get("done", False)
            delta = {"content": content} if content else {}
            if done and msg.get("tool_calls"):
                oai_tcs = _ollama_tool_calls_to_oai(msg["tool_calls"])
                yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': MODEL, 'choices': [{'index': 0, 'delta': {'tool_calls': oai_tcs}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex[:8]}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': MODEL, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
                yield "data: [DONE]\n\n"
                return
            oai_chunk = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if done else None}]}
            yield f"data: {json.dumps(oai_chunk)}\n\n"
            if done:
                yield "data: [DONE]\n\n"
                return
        # /api/generate chunk shape
        elif "response" in chunk:
            content = chunk.get("response", "")
            done = chunk.get("done", False)
            delta = {"content": content} if content else {}
            oai_chunk = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if done else None}]}
            yield f"data: {json.dumps(oai_chunk)}\n\n"
            if done:
                yield "data: [DONE]\n\n"
                return

def _generate_anthropic_stream(messages, tools, max_tokens, temperature):
    msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': MODEL, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    try:
        resp = _ollama_chat_stream(messages, tools, max_tokens, temperature)
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"
        return

    last_ping = time.time()
    for line in resp.iter_lines():
        if not line: continue
        if time.time() - last_ping > 15:
            yield "event: ping\ndata: {\"type\":\"ping\"}\n\n"
            last_ping = time.time()
        try: chunk = json.loads(line)
        except: continue

        # Normalise both chunk shapes
        if "message" in chunk:
            msg = chunk.get("message", {})
            content = msg.get("content", "")
            done = chunk.get("done", False)
        else:
            content = chunk.get("response", "")
            done = chunk.get("done", False)
            msg = {}

        if content:
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"

        if done:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            stop_reason = "end_turn"
            tcs = msg.get("tool_calls", [])
            if tcs:
                stop_reason = "tool_use"
                for idx, tc in enumerate(tcs, 1):
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    tu_id = f"toolu_{uuid.uuid4().hex[:8]}"
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'tool_use', 'id': tu_id, 'name': fn.get('name', ''), 'input': {}}})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(args if isinstance(args, dict) else {})}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            return

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Ollama Proxy with Tool Calling", version="2.0.0")
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
    return {
        "service": "Ollama Proxy with Tool Calling + System Tools",
        "version": "2.0.0",
        "model": MODEL,
        "ollama_base": OLLAMA_BASE,
        "working_dir": WORKING_DIR,
        "active_endpoint": "/api/chat" if USE_CHAT_ENDPOINT else "/api/generate (fallback)",
        "tools_available": [t["function"]["name"] for t in TOOL_DEFINITIONS]
    }

@app.get("/health")
async def health():
    try:
        resp = ollama_session.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if resp.status_code == 200:
            models_raw = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models_raw]
            # Clients may omit the tag; match on base name too
            model_loaded = MODEL in model_names or any(n.startswith(MODEL.split(":")[0]) for n in model_names)
            return {
                "status": "ready" if model_loaded else "degraded",
                "model": MODEL,
                "model_loaded": model_loaded,
                "available_models": model_names,
                "ollama_ok": True,
                "ollama_base": OLLAMA_BASE,
                "active_endpoint": "/api/chat" if USE_CHAT_ENDPOINT else "/api/generate",
                "tools_count": len(TOOL_DEFINITIONS)
            }
    except Exception as e:
        return {"status": "down", "model": MODEL, "error": str(e)}
    return {"status": "down", "model": MODEL}

@app.get("/v1/models")
async def list_models():
    """Return models in OpenAI-compatible format so clients can detect the model."""
    try:
        resp = ollama_session.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = []
            for m in resp.json().get("models", []):
                name = m.get("name", "")
                data.append({
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama",
                    "permission": [],
                    "root": name,
                    "parent": None
                })
            if not data:
                data = [{"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"}]
            return {"object": "list", "data": data}
    except: pass
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"}]}

@app.get("/v1/tools")
async def list_tools():
    return {"tools": TOOL_DEFINITIONS, "count": len(TOOL_DEFINITIONS)}

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    try: body = await request.json()
    except: raise HTTPException(400, "Invalid JSON")
    try: req = ChatCompletionRequest(**body)
    except Exception as e: raise HTTPException(400, f"Validation failed: {e}")

    messages = [m.model_dump() for m in req.messages]
    tc = req.tool_choice
    tc_type = tc if isinstance(tc, str) else (tc or {}).get("type", "auto")
    effective_tools = None if tc_type == "none" else (req.tools if req.tools is not None else TOOL_DEFINITIONS)

    if req.stream:
        return StreamingResponse(
            _generate_openai_stream(messages, effective_tools, req.max_tokens, req.temperature),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
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
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [{"index": 0, "message": response_msg, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            }
        }
    except HTTPException: raise
    except Exception as e:
        log(f"❌ Chat failed: {e}")
        raise HTTPException(503, f"Chat failed: {e}")

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    try: body = await request.json()
    except: raise HTTPException(400, "Invalid JSON")

    system = body.get("system", "")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 2048)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)
    tools = body.get("tools")
    tool_choice = body.get("tool_choice", {"type": "auto"})
    tc_type = tool_choice.get("type", "auto") if isinstance(tool_choice, dict) else str(tool_choice)

    full_messages = ([{"role": "system", "content": system}] if system else []) + messages
    effective_tools = None if tc_type == "none" else (_anthropic_tools_to_ollama(tools) if tools else _oai_tools_to_ollama(TOOL_DEFINITIONS))

    if stream:
        return StreamingResponse(
            _generate_anthropic_stream(full_messages, effective_tools, max_tokens, temperature),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
    try:
        result = _ollama_chat_sync(full_messages, effective_tools, max_tokens, temperature)
        msg = result.get("message", {})
        content = msg.get("content") or ""
        ollama_tcs = msg.get("tool_calls") or []
        content_blocks = ([{"type": "text", "text": content}] if content else [])
        stop_reason = "end_turn"
        for tc in ollama_tcs:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            content_blocks.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:8]}", "name": fn.get("name", ""), "input": args if isinstance(args, dict) else {}})
            stop_reason = "tool_use"
        return {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "type": "message", "role": "assistant", "model": MODEL,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": result.get("prompt_eval_count", 0), "output_tokens": result.get("eval_count", 0)}
        }
    except HTTPException: raise
    except Exception as e: raise HTTPException(503, f"Messages failed: {e}")

@app.post("/v1/tool/execute")
async def execute_tool(request: Request):
    try:
        body = await request.json()
        tool_name = body.get("name")
        arguments = body.get("arguments", {})
        if not tool_name: raise HTTPException(400, "Missing 'name' field")
        log(f"🔧 Tool execution request: {tool_name}")
        result = tool_executor.execute(tool_name, arguments)
        result_obj = json.loads(result)
        return {"tool": tool_name, "result": result_obj}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Tool execution failed: {e}")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str):
    return JSONResponse({
        "error": "Not found", "path": f"/{path}",
        "available": ["/", "/health", "/v1/models", "/v1/tools", "/v1/chat/completions", "/v1/messages", "/v1/tool/execute"]
    }, status_code=404)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 70)
    log("  Ollama Proxy v2 — Tool Calling + System Tools [AGENTIC MODE]")
    log("=" * 70)
    log(f"🤖 Model:       {MODEL}")
    log(f"🔌 Ollama:      {OLLAMA_BASE}")
    log(f"🌐 Proxy Port:  {PROXY_PORT}")
    log(f"📁 Working Dir: {WORKING_DIR}")
    log(f"🔧 Tools:       {len(TOOL_DEFINITIONS)}")
    log("=" * 70)

    try:
        import psutil
    except ImportError:
        log("📦 Installing psutil...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "-q"], check=True)
        import psutil

    detect_ollama_capabilities()
    log(f"🛰  Active endpoint: /api/{'chat' if USE_CHAT_ENDPOINT else 'generate'}")

    _start_background_threads()
    log(f"🚀 Starting Uvicorn on port {PROXY_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="warning", timeout_keep_alive=300)
