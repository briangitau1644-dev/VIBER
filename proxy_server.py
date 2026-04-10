#!/usr/bin/env python3
"""
Ollama Proxy Server v4.1 — gemma4:e4b — RAM-Resident Mode
=========================================================
Fixes applied:
  • Strict URL normalization middleware (strips invalid schemes/ports)
  • Backend OLLAMA_BASE always uses http://127.0.0.1:11434 (never client input)
  • Request validation rejects malformed apiBase early with helpful error
  • Cloudflare tunnel health endpoint returns full config for debugging
  • All proxy→Ollama requests use explicit, hardcoded backend URL
"""

import os, sys, json, time, uuid, threading, random, subprocess, shutil, logging, re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Backend is ALWAYS localhost:11434, never client-controlled
# ─────────────────────────────────────────────────────────────────────────────

MODEL           = os.getenv("MODEL", "gemma4:e4b")
OLLAMA_PORT     = os.getenv("OLLAMA_PORT", "11434")
# ⚠️ CRITICAL: Backend URL is hardcoded to loopback — never trust client input
OLLAMA_BASE     = f"http://127.0.0.1:{OLLAMA_PORT}"
PROXY_PORT      = int(os.getenv("PROXY_PORT", "8000"))
RAM_PRESSURE_EN = os.getenv("RAM_PRESSURE_ENABLED", "true").lower() == "true"
RAM_PRESSURE_MB = int(os.getenv("RAM_PRESSURE_MB", "512"))
LOG_INTERVAL    = float(os.getenv("ACTIVITY_LOG_INTERVAL", "30"))
KEEPALIVE_INT   = float(os.getenv("KEEPALIVE_INTERVAL", "8"))
WORKING_DIR     = os.getenv("WORKING_DIR", os.getcwd())

OLLAMA_KEEP_ALIVE = -1  # Keep model in RAM forever

MODEL_ALIASES = {
    MODEL, MODEL.split(":")[0],
    "gemma4", "gemma4:e4b", "gemma", "default", "local",
    "sorc/qwen3.5-claude-4.6-opus:4b", "qwen3.5",
}

USE_CHAT_ENDPOINT = True  # Default to /api/chat; auto-detect fallback

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("proxy")

def log(msg: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    getattr(logger, level, logger.info)(msg)

log(f"⚡ Proxy v4.1 — Model: {MODEL}")
log(f"🔌 Backend Ollama: {OLLAMA_BASE} (HARDCODED LOOPBACK)")
log(f"📁 WorkDir: {WORKING_DIR}")
log(f"🧠 RAM Keepalive: every {KEEPALIVE_INT}s (keep_alive={OLLAMA_KEEP_ALIVE})")

# ─────────────────────────────────────────────────────────────────────────────
# URL VALIDATION MIDDLEWARE — Reject malformed client URLs early
# ─────────────────────────────────────────────────────────────────────────────

class URLValidationMiddleware(BaseHTTPMiddleware):
    """
    Intercept requests and validate that the client isn't sending malformed URLs.
    Common issue: clients send 'apibase://:80' which is not a valid URL scheme.
    """
    async def dispatch(self, request: Request, call_next):
        # Check for obviously malformed URLs in headers or query params
        raw_url = str(request.url)
        
        # Detect invalid schemes like 'apibase://'
        if re.search(r'[a-zA-Z]+://:\d+', raw_url) or raw_url.startswith('apibase://'):
            log(f"❌ Rejected malformed URL from client: {raw_url[:200]}", "error")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid apiBase configuration",
                    "detail": "Client sent malformed URL. Expected: https://<tunnel-url> or https://<tunnel-url>/v1",
                    "received": raw_url[:100],
                    "fix": "Set apiBase to the full Cloudflare tunnel URL (e.g., https://xyz.trycloudflare.com/v1)",
                }
            )
        
        # Normalize: ensure no trailing slash on internal routing
        path = request.url.path
        if path != "/" and path.endswith("/"):
            # Redirect to clean URL (307 preserves method/body)
            clean_url = request.url.replace(path=path.rstrip("/"))
            return RedirectResponse(url=str(clean_url), status_code=307)
        
        return await call_next(request)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION — TCP keepalives + retry for backend Ollama calls
# ─────────────────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PUT", "DELETE"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=32,
        pool_maxsize=64,
        pool_block=False,
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"Connection": "keep-alive"})
    return s

ollama = _make_session()

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {"type":"function","function":{"name":"get_current_time","description":"Return current ISO datetime.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"calculate","description":"Evaluate a safe math expression: +,-,*,/,**,%","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Math expression"}},"required":["expression"]}}},
    {"type":"function","function":{"name":"search_web","description":"Search the web (simulated).","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"get_weather","description":"Get weather for a location (simulated).","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},
    {"type":"function","function":{"name":"run_command","description":"Run a shell command in the working directory.","parameters":{"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"},"timeout":{"type":"integer","default":60}},"required":["command"]}}},
    {"type":"function","function":{"name":"create_file","description":"Create or overwrite a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"encoding":{"type":"string","default":"utf-8"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a file's contents.","parameters":{"type":"object","properties":{"path":{"type":"string"},"max_bytes":{"type":"integer","default":1048576}},"required":["path"]}}},
    {"type":"function","function":{"name":"edit_file","description":"Edit an existing file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_content":{"type":"string"},"new_content":{"type":"string"},"operation":{"type":"string","enum":["replace","append","prepend"],"default":"replace"}},"required":["path","new_content"]}}},
    {"type":"function","function":{"name":"delete_file","description":"Delete a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"create_folder","description":"Create a directory.","parameters":{"type":"object","properties":{"path":{"type":"string"},"parents":{"type":"boolean","default":True}},"required":["path"]}}},
    {"type":"function","function":{"name":"delete_folder","description":"Delete a directory.","parameters":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean","default":True}},"required":["path"]}}},
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
    def _safe(path: Path, allow_outside: bool = False):
        try:
            r = path.resolve()
            if not allow_outside:
                try:
                    r.relative_to(Path(WORKING_DIR).resolve())
                except ValueError:
                    return False, f"Path outside working dir: {r}"
            return True, str(r)
        except Exception as e:
            return False, str(e)

    @staticmethod
    def execute(name: str, args: Dict) -> str:
        fn = getattr(ToolExecutor, f"_{name}", None)
        if not fn:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            log(f"🔧 TOOL {name}({str(args)[:160]})")
            result = fn(**args)
            log(f"   ↳ {str(result)[:120]}")
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            err = {"error": f"{type(e).__name__}: {e}"}
            log(f"   ↳ ERROR: {err}", "error")
            return json.dumps(err)

    @staticmethod
    def _get_current_time():
        return {"timestamp": datetime.now().isoformat(), "timezone": "UTC"}

    @staticmethod
    def _calculate(expression: str):
        SAFE = set("0123456789+-*/().% ")
        if not all(c in SAFE for c in expression):
            return {"error": "Invalid characters in expression"}
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _search_web(query: str):
        return {"query": query, "results": [], "note": "Simulated — real search not available"}

    @staticmethod
    def _get_weather(location: str):
        return {
            "location": location,
            "temp_c": random.randint(-5, 38),
            "temp_f": None,
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Windy"]),
            "note": "Simulated data",
        }

    @staticmethod
    def _run_command(command: str, cwd: str = None, timeout: int = 60):
        log(f"⚠️  Shell CMD: {command}")
        wd = Path(cwd).resolve() if cwd else Path(WORKING_DIR)
        try:
            result = subprocess.run(
                command, shell=True, cwd=str(wd),
                capture_output=True, text=True, timeout=timeout,
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout[:8192],
                "stderr": result.stderr[:2048],
                "command": command,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Timed out after {timeout}s", "command": command}
        except Exception as e:
            return {"error": str(e), "command": command}

    @staticmethod
    def _create_file(path: str, content: str, encoding: str = "utf-8"):
        fp = ToolExecutor._resolve(path)
        ok, r = ToolExecutor._safe(fp)
        if not ok:
            return {"error": r}
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            fp.write_text(content, encoding=encoding)
            return {"success": True, "path": r, "bytes": len(content.encode(encoding))}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _read_file(path: str, max_bytes: int = 1048576):
        fp = ToolExecutor._resolve(path)
        ok, r = ToolExecutor._safe(fp)
        if not ok:
            return {"error": r}
        if not fp.is_file():
            return {"error": f"Not a file: {r}"}
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            truncated = len(content.encode()) > max_bytes
            return {
                "success": True, "path": r,
                "content": content[:max_bytes],
                "truncated": truncated,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _edit_file(path: str, new_content: str, old_content: str = None, operation: str = "replace"):
        fp = ToolExecutor._resolve(path)
        ok, r = ToolExecutor._safe(fp)
        if not ok:
            return {"error": r}
        if not fp.exists():
            return {"error": f"File not found: {r}"}
        cur = fp.read_text(encoding="utf-8", errors="replace")
        if operation == "replace" and old_content:
            if old_content not in cur:
                return {"error": "old_content not found in file"}
            out = cur.replace(old_content, new_content, 1)
        elif operation == "append":
            out = cur + new_content
        elif operation == "prepend":
            out = new_content + cur
        else:
            out = new_content
        try:
            fp.write_text(out, encoding="utf-8")
            return {"success": True, "path": r, "operation": operation}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _delete_file(path: str):
        fp = ToolExecutor._resolve(path)
        ok, r = ToolExecutor._safe(fp)
        if not ok:
            return {"error": r}
        if not fp.is_file():
            return {"error": f"Not a file: {r}"}
        try:
            fp.unlink()
            return {"success": True, "path": r}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _create_folder(path: str, parents: bool = True):
        dp = ToolExecutor._resolve(path)
        ok, r = ToolExecutor._safe(dp)
        if not ok:
            return {"error": r}
        try:
            Path(r).mkdir(parents=parents, exist_ok=True)
            return {"success": True, "path": r}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _delete_folder(path: str, recursive: bool = True):
        dp = ToolExecutor._resolve(path)
        ok, r = ToolExecutor._safe(dp)
        if not ok:
            return {"error": r}
        if not Path(r).is_dir():
            return {"error": f"Not a directory: {r}"}
        try:
            shutil.rmtree(r) if recursive else Path(r).rmdir()
            return {"success": True, "path": r}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _list_directory(path: str = ".", recursive: bool = False):
        dp = ToolExecutor._resolve(path)
        ok, r = ToolExecutor._safe(dp, allow_outside=True)
        if not ok:
            return {"error": r}
        if not Path(r).is_dir():
            return {"error": f"Not a directory: {r}"}
        items = []
        try:
            if recursive:
                for root, dirs, files in os.walk(r):
                    rr = Path(root).relative_to(r)
                    for d in dirs:
                        items.append({"type": "dir", "path": str(rr / d)})
                    for f in files:
                        fpath = Path(root) / f
                        items.append({"type": "file", "path": str(rr / f),
                                      "size": fpath.stat().st_size})
            else:
                for item in sorted(Path(r).iterdir()):
                    items.append({
                        "type": "dir" if item.is_dir() else "file",
                        "name": item.name,
                        "size": item.stat().st_size if item.is_file() else None,
                    })
            return {"success": True, "path": r, "items": items, "count": len(items)}
        except Exception as e:
            return {"error": str(e)}

tool_executor = ToolExecutor()

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CAPABILITY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_ollama_capabilities() -> bool:
    global USE_CHAT_ENDPOINT
    log("🔍 Detecting Ollama capabilities...")
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/version", timeout=10)
        if r.status_code == 200:
            v = r.json().get("version", "0.0.0")
            log(f"📦 Ollama version: {v}")
            parts = v.replace("-", ".").split(".")
            try:
                maj, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                if (maj, minor, patch) >= (0, 1, 16):
                    log("✅ /api/chat confirmed by version number")
                    USE_CHAT_ENDPOINT = True
                    return True
            except (IndexError, ValueError):
                pass
    except Exception as e:
        log(f"⚠️  Version check failed: {e}", "warning")
    try:
        r = ollama.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": MODEL, "stream": False,
                  "messages": [{"role": "user", "content": "hi"}],
                  "keep_alive": OLLAMA_KEEP_ALIVE,
                  "options": {"num_predict": 1}},
            timeout=30,
        )
        if r.status_code in (200, 400, 422):
            log("✅ /api/chat live probe succeeded")
            USE_CHAT_ENDPOINT = True
            return True
        if r.status_code == 404:
            log("⚠️  /api/chat returned 404 — using /api/generate", "warning")
            USE_CHAT_ENDPOINT = False
            return False
    except Exception as e:
        log(f"⚠️  /api/chat probe error: {e} — defaulting to /api/generate", "warning")
        USE_CHAT_ENDPOINT = False
        return False
    USE_CHAT_ENDPOINT = True
    return True


def ensure_model_in_ram():
    log(f"🧠 Ensuring {MODEL} is loaded in RAM (keep_alive=-1)...")
    endpoint = "/api/chat" if USE_CHAT_ENDPOINT else "/api/generate"
    try:
        if USE_CHAT_ENDPOINT:
            payload = {
                "model": MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": 1},
            }
        else:
            payload = {
                "model": MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
                "prompt": "hi", "options": {"num_predict": 1},
            }
        r = ollama.post(f"{OLLAMA_BASE}{endpoint}", json=payload, timeout=300)
        if r.status_code == 200:
            log(f"✅ {MODEL} confirmed in RAM")
            return True
        else:
            log(f"⚠️  RAM load response: {r.status_code} {r.text[:200]}", "warning")
            return False
    except Exception as e:
        log(f"⚠️  RAM load error: {e}", "warning")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND THREADS
# ─────────────────────────────────────────────────────────────────────────────

_ram_buffer = None

def _ram_pressure_thread():
    global _ram_buffer
    if not RAM_PRESSURE_EN:
        return
    try:
        _ram_buffer = bytearray(RAM_PRESSURE_MB * 1024 * 1024)
        for i in range(0, len(_ram_buffer), 4096):
            _ram_buffer[i] = 1
        log(f"💾 Locked {RAM_PRESSURE_MB} MB in RAM buffer")
    except MemoryError:
        log("⚠️  RAM buffer allocation failed (MemoryError)", "warning")
    while True:
        if _ram_buffer:
            _ram_buffer[0] = _ram_buffer[-1] = 1
        time.sleep(30)

def _activity_logger_thread():
    import psutil
    while True:
        try:
            vm = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.5)
            ep = "/api/chat" if USE_CHAT_ENDPOINT else "/api/generate"
            log(f"📊 CPU:{cpu:.1f}%  RAM:{vm.used//1024//1024}MB/{vm.total//1024//1024}MB ({vm.percent}%)  ep:{ep}")
        except Exception as e:
            log(f"⚠️  Logger error: {e}", "warning")
        time.sleep(LOG_INTERVAL)

def _model_keepalive_thread():
    s = _make_session()
    consecutive_failures = 0
    while True:
        try:
            r = s.get(f"{OLLAMA_BASE}/api/tags", timeout=8)
            if r.status_code == 200:
                consecutive_failures = 0
                log("💓 Ollama alive")
            else:
                consecutive_failures += 1
                log(f"⚠️  /api/tags returned {r.status_code}", "warning")
            try:
                ps_r = s.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
                if ps_r.status_code == 200:
                    ps_data = ps_r.json()
                    loaded = [m.get("name","") for m in ps_data.get("models", [])]
                    model_loaded = any(MODEL in n or n.split(":")[0] in MODEL for n in loaded)
                    if not model_loaded:
                        log(f"⚠️  {MODEL} not in RAM (loaded: {loaded}) — triggering reload...", "warning")
                        ensure_model_in_ram()
                    else:
                        log(f"🧠 {MODEL} confirmed in RAM")
            except Exception:
                pass
        except requests.ConnectionError as e:
            consecutive_failures += 1
            log(f"⚠️  Keepalive connection error: {e}", "warning")
        except Exception as e:
            consecutive_failures += 1
            log(f"⚠️  Keepalive error: {e}", "warning")
        if consecutive_failures >= 5:
            log("❌ 5 consecutive keepalive failures — Ollama may be down!", "error")
            consecutive_failures = 0
        jitter = random.uniform(-1, 1)
        time.sleep(max(1, KEEPALIVE_INT + jitter))

def start_background_threads():
    threads = [
        (_ram_pressure_thread, "ram-pressure"),
        (_activity_logger_thread, "activity-log"),
        (_model_keepalive_thread, "model-keepalive"),
    ]
    for fn, name in threads:
        t = threading.Thread(target=fn, daemon=True, name=name)
        t.start()
        log(f"🧵 Started thread: {name}")
    log("✅ All background threads running")

# ─────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_params(p) -> Dict:
    if not p or not isinstance(p, dict):
        return {"type": "object", "properties": {}}
    return p if "type" in p else {"type": "object", **p}

def _oai_tools_to_ollama(tools: List[Dict]) -> Optional[List[Dict]]:
    if not tools:
        return None
    out = []
    for t in tools:
        if t.get("type") != "function":
            continue
        f = t["function"]
        out.append({"type": "function", "function": {
            "name": f.get("name", ""),
            "description": f.get("description", ""),
            "parameters": _norm_params(f.get("parameters")),
        }})
    return out or None

def _anthropic_tools_to_ollama(tools: List[Dict]) -> Optional[List[Dict]]:
    if not tools:
        return None
    out = []
    for t in tools:
        out.append({"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": _norm_params(t.get("input_schema", {})),
        }})
    return out or None

def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                parts.append(str(b))
                continue
            t = b.get("type", "")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_result":
                c = b.get("content", "")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                parts.append(str(c))
        return "\n".join(parts)
    return str(content or "")

def _msgs_to_ollama(messages: List[Dict]) -> List[Dict]:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = _content_to_str(m.get("content") or "")
        if role in ("system", "user"):
            out.append({"role": role, "content": content})
        elif role == "assistant":
            msg: Dict = {"role": "assistant", "content": content}
            tcs = m.get("tool_calls")
            if tcs:
                ollama_tcs = []
                for tc in tcs:
                    f = tc.get("function", {})
                    args = f.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    ollama_tcs.append({"function": {"name": f.get("name", ""), "arguments": args}})
                msg["tool_calls"] = ollama_tcs
            out.append(msg)
        elif role == "tool":
            tm: Dict = {"role": "tool", "content": content}
            if m.get("tool_call_id"):
                tm["tool_call_id"] = m["tool_call_id"]
            out.append(tm)
    return out

def _msgs_to_prompt(messages: List[Dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = _content_to_str(m.get("content") or "")
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

def _tcs_to_oai(tcs: List[Dict]) -> List[Dict]:
    out = []
    for tc in tcs:
        f = tc.get("function", {})
        args = f.get("arguments", {})
        out.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": f.get("name", ""),
                "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
            },
        })
    return out

def _resolve_model(name: Optional[str]) -> str:
    if not name:
        return MODEL
    if name in MODEL_ALIASES:
        return MODEL
    for alias in MODEL_ALIASES:
        if alias and name in alias:
            return MODEL
    return MODEL

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CALLS — Backend URL is HARDCODED, never from client
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama_sync(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 2048,
    temp: float = 0.7,
) -> Dict:
    global USE_CHAT_ENDPOINT
    start = time.time()
    if USE_CHAT_ENDPOINT:
        payload: Dict = {
            "model": MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": _msgs_to_ollama(messages),
            "options": {"num_predict": max_tokens, "temperature": temp},
        }
        if tools:
            payload["tools"] = _oai_tools_to_ollama(tools)
        for attempt in range(1, 4):
            try:
                log(f"➡️  /api/chat attempt {attempt} ({len(messages)} msgs)")
                # ⚠️ CRITICAL: Use hardcoded OLLAMA_BASE, never client input
                r = ollama.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=(15, 600))
                if r.status_code == 404:
                    log("⚠️  /api/chat 404 — switching to /api/generate", "warning")
                    USE_CHAT_ENDPOINT = False
                    break
                r.raise_for_status()
                elapsed = time.time() - start
                log(f"✅ /api/chat done in {elapsed:.1f}s")
                return r.json()
            except requests.Timeout:
                log(f"⚠️  Attempt {attempt}: timeout", "warning")
                if attempt == 3:
                    raise HTTPException(504, "Ollama timed out after 3 attempts")
                time.sleep(2 ** attempt)
            except requests.ConnectionError as e:
                log(f"⚠️  Attempt {attempt}: connection error: {e}", "warning")
                if attempt == 3:
                    raise HTTPException(503, f"Cannot connect to Ollama: {e}")
                time.sleep(2 ** attempt)
            except HTTPException:
                raise
            except requests.RequestException as e:
                log(f"⚠️  Attempt {attempt}: {e}", "warning")
                if attempt == 3:
                    raise HTTPException(503, str(e))
                time.sleep(2 ** attempt)
    # Fallback to /api/generate
    log("📡 Using /api/generate fallback")
    payload_gen = {
        "model": MODEL, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE,
        "prompt": _msgs_to_prompt(messages),
        "options": {"num_predict": max_tokens, "temperature": temp},
    }
    for attempt in range(1, 4):
        try:
            r = ollama.post(f"{OLLAMA_BASE}/api/generate", json=payload_gen, timeout=(15, 600))
            r.raise_for_status()
            d = r.json()
            return {
                "message": {"role": "assistant", "content": d.get("response", "")},
                "prompt_eval_count": d.get("prompt_eval_count", 0),
                "eval_count": d.get("eval_count", 0),
            }
        except requests.Timeout:
            if attempt == 3:
                raise HTTPException(504, "Ollama /api/generate timed out")
            time.sleep(2 ** attempt)
        except requests.ConnectionError as e:
            if attempt == 3:
                raise HTTPException(503, f"Cannot connect to Ollama: {e}")
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            raise HTTPException(503, str(e))
    raise HTTPException(503, "All Ollama attempts failed")

def _call_ollama_stream(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 2048,
    temp: float = 0.7,
):
    global USE_CHAT_ENDPOINT
    if USE_CHAT_ENDPOINT:
        payload = {
            "model": MODEL, "stream": True, "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": _msgs_to_ollama(messages),
            "options": {"num_predict": max_tokens, "temperature": temp},
        }
        if tools:
            payload["tools"] = _oai_tools_to_ollama(tools)
        try:
            r = ollama.post(f"{OLLAMA_BASE}/api/chat", json=payload, stream=True, timeout=(15, 600))
            if r.status_code == 404:
                log("⚠️  /api/chat 404 in stream — switching to /api/generate", "warning")
                USE_CHAT_ENDPOINT = False
            else:
                return r, "chat"
        except requests.RequestException as e:
            log(f"⚠️  Stream /api/chat failed: {e} — trying /api/generate", "warning")
    # Fallback
    payload_gen = {
        "model": MODEL, "stream": True, "keep_alive": OLLAMA_KEEP_ALIVE,
        "prompt": _msgs_to_prompt(messages),
        "options": {"num_predict": max_tokens, "temperature": temp},
    }
    r = ollama.post(f"{OLLAMA_BASE}/api/generate", json=payload_gen, stream=True, timeout=(15, 600))
    return r, "generate"

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _stream_openai(messages, tools, max_tokens, temp):
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    yield ": stream-start\n\n"
    try:
        resp, mode = _call_ollama_stream(messages, tools, max_tokens, temp)
    except HTTPException as e:
        yield f"data: {json.dumps({'error': e.detail})}\n\ndata: [DONE]\n\n"
        return
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\ndata: [DONE]\n\n"
        return
    last_ping = time.time()
    for raw in resp.iter_lines():
        if not raw:
            continue
        if time.time() - last_ping > 12:
            yield ": keep-alive\n\n"
            last_ping = time.time()
        try:
            chunk = json.loads(raw)
        except Exception:
            continue
        if mode == "chat":
            msg = chunk.get("message", {})
            text = msg.get("content", "")
            done = chunk.get("done", False)
            tcs = msg.get("tool_calls", []) if done else []
            if tcs:
                oai_tcs = _tcs_to_oai(tcs)
                yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':int(time.time()),'model':MODEL,'choices':[{'index':0,'delta':{'tool_calls':oai_tcs},'finish_reason':None}]})}\n\n"
                yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':int(time.time()),'model':MODEL,'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls'}]})}\n\n"
                yield "data: [DONE]\n\n"
                return
            delta = {"content": text} if text else {}
        else:
            text = chunk.get("response", "")
            done = chunk.get("done", False)
            delta = {"content": text} if text else {}
        payload = {
            "id": cid, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if done else None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        if done:
            yield "data: [DONE]\n\n"
            return

def _stream_anthropic(messages, tools, max_tokens, temp):
    mid = f"msg_{uuid.uuid4().hex[:8]}"
    yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':mid,'type':'message','role':'assistant','content':[],'model':MODEL,'stop_reason':None,'usage':{'input_tokens':0,'output_tokens':0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"
    try:
        resp, mode = _call_ollama_stream(messages, tools, max_tokens, temp)
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'type':'error','error':{'message':str(e)}})}\n\n"
        return
    last_ping = time.time()
    for raw in resp.iter_lines():
        if not raw:
            continue
        if time.time() - last_ping > 12:
            yield 'event: ping\ndata: {"type":"ping"}\n\n'
            last_ping = time.time()
        try:
            chunk = json.loads(raw)
        except Exception:
            continue
        msg = chunk.get("message", {}) if mode == "chat" else {}
        text = msg.get("content", "") if mode == "chat" else chunk.get("response", "")
        done = chunk.get("done", False)
        if text:
            yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':text}})}\n\n"
        if done:
            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"
            stop = "end_turn"
            tcs = msg.get("tool_calls", [])
            for i, tc in enumerate(tcs, 1):
                f = tc.get("function", {})
                args = f.get("arguments", {})
                tid = f"toolu_{uuid.uuid4().hex[:8]}"
                stop = "tool_use"
                yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':i,'content_block':{'type':'tool_use','id':tid,'name':f.get('name',''),'input':{}}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':i,'delta':{'type':'input_json_delta','partial_json':json.dumps(args if isinstance(args,dict) else {})}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':i})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop,'stop_sequence':None},'usage':{'output_tokens':0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"
            return

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("🚀 FastAPI startup — detecting Ollama capabilities...")
    detect_ollama_capabilities()
    log(f"🛰  Active endpoint: /api/{'chat' if USE_CHAT_ENDPOINT else 'generate'}")
    ensure_model_in_ram()
    start_background_threads()
    yield
    log("🛑 FastAPI shutdown")

app = FastAPI(title="Ollama Proxy v4.1 — gemma4:e4b", version="4.1.0", lifespan=lifespan)

# Add middleware in correct order: URL validation FIRST, then CORS, then strip-slash
app.add_middleware(URLValidationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Strip trailing slashes AFTER validation
class StripSlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path != "/" and path.endswith("/"):
            url = request.url.replace(path=path.rstrip("/"))
            return RedirectResponse(url=str(url), status_code=307)
        return await call_next(request)
app.add_middleware(StripSlashMiddleware)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict], None] = ""
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = "auto"

def _tc_type(tc) -> str:
    if isinstance(tc, str):
        return tc
    return (tc or {}).get("type", "auto")

def _effective_tools(req_tools, tc):
    if _tc_type(tc) == "none":
        return None
    return req_tools if req_tools is not None else TOOL_DEFINITIONS

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Discovery & Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "object": "list",
        "data": [{"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"}],
        "service": "Ollama Proxy v4.1",
        "version": "4.1.0",
        "model": MODEL,
        "active_endpoint": "/api/chat" if USE_CHAT_ENDPOINT else "/api/generate",
        "ram_keepalive": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "tools_count": len(TOOL_DEFINITIONS),
        "backend_ollama": OLLAMA_BASE,
        "routes": ["/v1/chat/completions", "/v1/messages", "/v1/models", "/api/chat", "/api/generate", "/api/tags", "/api/version", "/api/ps", "/health", "/v1/tools", "/v1/tool/execute"],
    }

@app.get("/health")
async def health():
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            names = [m.get("name", "") for m in r.json().get("models", [])]
            loaded = (MODEL in names or any(n.split(":")[0] == MODEL.split(":")[0] for n in names))
            ram_loaded = None
            try:
                ps_r = ollama.get(f"{OLLAMA_BASE}/api/ps", timeout=3)
                if ps_r.status_code == 200:
                    ps_models = [m.get("name","") for m in ps_r.json().get("models", [])]
                    ram_loaded = any(MODEL in n or n.split(":")[0] == MODEL.split(":")[0] for n in ps_models)
            except Exception:
                ram_loaded = None
            return {
                "status": "ready" if loaded else "degraded",
                "model": MODEL,
                "model_loaded": loaded,
                "model_in_ram": ram_loaded,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "available_models": names,
                "active_endpoint": "/api/chat" if USE_CHAT_ENDPOINT else "/api/generate",
                "tools_count": len(TOOL_DEFINITIONS),
                "ollama_ok": True,
                "proxy_version": "4.1.0",
                "backend_url": OLLAMA_BASE,
            }
    except Exception as e:
        return {"status": "down", "model": MODEL, "error": str(e), "ollama_ok": False, "backend_url": OLLAMA_BASE}
    return {"status": "down", "model": MODEL, "ollama_ok": False, "backend_url": OLLAMA_BASE}

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — OpenAI-compatible
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            data = [{"id": m.get("name",""), "object": "model", "created": int(time.time()), "owned_by": "ollama", "root": m.get("name",""), "parent": None, "permission": []} for m in r.json().get("models", [])]
            if data:
                return {"object": "list", "data": data}
    except Exception:
        pass
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama", "root": MODEL, "parent": None, "permission": []}]}

@app.get("/v1/tools")
async def list_tools():
    return {"tools": TOOL_DEFINITIONS, "count": len(TOOL_DEFINITIONS)}

@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    try:
        req = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(400, f"Request validation error: {e}")
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    tools = _effective_tools(req.tools, req.tool_choice)
    log(f"📨 /v1/chat/completions stream={req.stream} msgs={len(messages)}")
    if req.stream:
        return StreamingResponse(_stream_openai(messages, tools, req.max_tokens, req.temperature), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering":"no", "Transfer-Encoding":"chunked"})
    result = _call_ollama_sync(messages, tools, req.max_tokens, req.temperature)
    msg = result.get("message", {})
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    rmsg: Dict = {"role": "assistant", "content": content or None}
    fr = "stop"
    if tcs:
        rmsg["tool_calls"] = _tcs_to_oai(tcs)
        fr = "tool_calls"
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "message": rmsg, "finish_reason": fr}], "usage": {"prompt_tokens": result.get("prompt_eval_count", 0), "completion_tokens": result.get("eval_count", 0), "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)}}

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    system = body.get("system", "")
    msgs = body.get("messages", [])
    max_tokens = body.get("max_tokens", 2048)
    temp = body.get("temperature", 0.7)
    stream = body.get("stream", False)
    tools = body.get("tools")
    tc = body.get("tool_choice", {"type": "auto"})
    tc_type = tc.get("type", "auto") if isinstance(tc, dict) else str(tc)
    full_msgs = ([{"role": "system", "content": system}] if system else []) + msgs
    eff_tools = None if tc_type == "none" else (_anthropic_tools_to_ollama(tools) if tools else _oai_tools_to_ollama(TOOL_DEFINITIONS))
    log(f"📨 /v1/messages stream={stream} msgs={len(full_msgs)}")
    if stream:
        return StreamingResponse(_stream_anthropic(full_msgs, eff_tools, max_tokens, temp), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering":"no"})
    result = _call_ollama_sync(full_msgs, eff_tools, max_tokens, temp)
    msg = result.get("message", {})
    content = msg.get("content") or ""
    tcs = msg.get("tool_calls") or []
    blocks = [{"type": "text", "text": content}] if content else []
    stop = "end_turn"
    for tc_item in tcs:
        f = tc_item.get("function", {})
        args = f.get("arguments", {})
        blocks.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:8]}", "name": f.get("name", ""), "input": args if isinstance(args, dict) else {}})
        stop = "tool_use"
    return {"id": f"msg_{uuid.uuid4().hex[:8]}", "type": "message", "role": "assistant", "model": MODEL, "content": blocks, "stop_reason": stop, "usage": {"input_tokens": result.get("prompt_eval_count", 0), "output_tokens": result.get("eval_count", 0)}}

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Raw Ollama passthrough
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def ollama_chat_passthrough(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    messages = body.get("messages", [])
    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temp = body.get("options", {}).get("temperature", 0.7)
    stream = body.get("stream", False)
    tools = body.get("tools")
    log(f"📨 /api/chat stream={stream} msgs={len(messages)}")
    if stream:
        async def _gen():
            try:
                resp, _ = _call_ollama_stream(messages, tools, max_tokens, temp)
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"
                return
            for line in resp.iter_lines():
                if line:
                    yield (line.decode() if isinstance(line, bytes) else line) + "\n"
        return StreamingResponse(_gen(), media_type="application/x-ndjson")
    result = _call_ollama_sync(messages, tools, max_tokens, temp)
    return {"model": MODEL, "message": result.get("message", {}), "done": True, "done_reason": "stop", "prompt_eval_count": result.get("prompt_eval_count", 0), "eval_count": result.get("eval_count", 0)}

@app.post("/api/generate")
async def ollama_generate_passthrough(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    log(f"📨 /api/generate stream={body.get('stream', False)}")
    payload = {"model": MODEL, "prompt": body.get("prompt", ""), "stream": body.get("stream", False), "keep_alive": OLLAMA_KEEP_ALIVE, "options": body.get("options", {})}
    try:
        if body.get("stream", False):
            resp = ollama.post(f"{OLLAMA_BASE}/api/generate", json=payload, stream=True, timeout=(15, 600))
            async def _gen():
                for line in resp.iter_lines():
                    if line:
                        yield (line.decode() if isinstance(line, bytes) else line) + "\n"
            return StreamingResponse(_gen(), media_type="application/x-ndjson")
        else:
            r = ollama.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=(15, 600))
            r.raise_for_status()
            return r.json()
    except requests.Timeout:
        raise HTTPException(504, "Ollama /api/generate timed out")
    except requests.ConnectionError as e:
        raise HTTPException(503, f"Cannot connect to Ollama: {e}")
    except Exception as e:
        raise HTTPException(503, str(e))

@app.get("/api/tags")
async def ollama_tags():
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"models": [{"name": MODEL, "model": MODEL, "modified_at": datetime.now().isoformat(), "size": 0, "digest": "", "details": {"format": "gguf", "family": "gemma", "parameter_size": "e4b", "quantization_level": "Q4_0"}}]}

@app.get("/api/version")
async def ollama_version():
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/version", timeout=5)
        if r.status_code == 200:
            data = r.json()
            v = data.get("version", "0.0.0")
            parts = v.replace("-", ".").split(".")
            try:
                if (int(parts[0]), int(parts[1]), int(parts[2])) < (0, 1, 16):
                    data["version"] = "0.5.0"
            except Exception:
                data["version"] = "0.5.0"
            return data
    except Exception:
        pass
    return {"version": "0.5.0"}

@app.get("/api/ps")
async def ollama_ps():
    try:
        r = ollama.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"models": []}

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Tool execution & RAM control
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/tool/execute")
async def execute_tool(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")
    name = body.get("name")
    args = body.get("arguments", {})
    if not name:
        raise HTTPException(400, "Missing 'name' field")
    log(f"🔧 Direct tool call: {name}")
    return {"tool": name, "result": json.loads(tool_executor.execute(name, args))}

@app.post("/v1/model/reload")
async def reload_model_into_ram():
    log("🔄 Manual RAM reload requested")
    ok = ensure_model_in_ram()
    return {"success": ok, "model": MODEL, "keep_alive": OLLAMA_KEEP_ALIVE}

# ─────────────────────────────────────────────────────────────────────────────
# CATCH-ALL
# ─────────────────────────────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def catch_all(path: str, request: Request):
    log(f"⚠️  404 {request.method} /{path}", "warning")
    return JSONResponse({
        "error": "Endpoint not found",
        "path": f"/{path}",
        "method": request.method,
        "hint": "Use /v1/chat/completions (OpenAI), /v1/messages (Anthropic), or /api/chat (raw Ollama)",
        "available": ["/", "/health", "/v1/models", "/v1/tools", "/v1/chat/completions", "/v1/messages", "/v1/tool/execute", "/v1/model/reload", "/api/chat", "/api/generate", "/api/tags", "/api/version", "/api/ps"],
        "client_config_help": {
            "OpenAI-compatible": f"apiBase: '{request.url.scheme}://{request.url.netloc}/v1'",
            "Raw Ollama": f"apiBase: '{request.url.scheme}://{request.url.netloc}'",
            "Anthropic": f"apiBase: '{request.url.scheme}://{request.url.netloc}/v1/messages'",
            "model": MODEL,
        }
    }, status_code=404)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72, flush=True)
    print("  Ollama Proxy v4.1 — gemma4:e4b — RAM-Resident Mode", flush=True)
    print("=" * 72, flush=True)
    print(f"  Model:       {MODEL}", flush=True)
    print(f"  Backend:     {OLLAMA_BASE} (HARDCODED — never client input)", flush=True)
    print(f"  Proxy Port:  {PROXY_PORT}", flush=True)
    print(f"  WorkDir:     {WORKING_DIR}", flush=True)
    print(f"  Tools:       {len(TOOL_DEFINITIONS)}", flush=True)
    print(f"  KeepAlive:   {OLLAMA_KEEP_ALIVE} (model always in RAM)", flush=True)
    print(f"  RAM Pressure:{RAM_PRESSURE_MB} MB", flush=True)
    print("", flush=True)
    print("  ✅ CLIENT CONFIGURATION (copy-paste these):", flush=True)
    print(f"    OpenAI:     apiBase = https://<tunnel>.trycloudflare.com/v1", flush=True)
    print(f"    Raw Ollama: apiBase = https://<tunnel>.trycloudflare.com", flush=True)
    print(f"    Anthropic:  apiBase = https://<tunnel>.trycloudflare.com/v1/messages", flush=True)
    print(f"    model:      {MODEL}", flush=True)
    print(f"    ⚠️  NEVER use 'apibase://:80' or omit the https:// prefix", flush=True)
    print("=" * 72, flush=True)
    try:
        import psutil
    except ImportError:
        print("Installing psutil...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "-q"], check=True)
        import psutil
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="warning", timeout_keep_alive=600, h11_max_incomplete_event_size=16 * 1024 * 1024, access_log=False)
