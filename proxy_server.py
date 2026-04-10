#!/usr/bin/env python3
"""
Ollama Proxy Server with Full Tool Calling Support + System Operations
GitHub Actions Edition - CPU Optimized
Supports: OpenAI /v1/chat/completions + Anthropic /v1/messages
System Tools: run_command, file operations, folder management
⚠️  Permission handling is delegated to the client application
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

# ⚠️  Working directory for file operations (client should validate paths)
WORKING_DIR = os.getenv('WORKING_DIR', os.getcwd())

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"⚡ Proxy Starting - Model: {MODEL}")
log(f"🔌 Ollama Base: {OLLAMA_BASE}")
log(f"📁 Working Directory: {WORKING_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA VERSION CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_version() -> bool:
    """Check if Ollama version supports /api/chat (requires >= 0.1.16)"""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5)
        if resp.status_code == 200:
            version = resp.json().get("version", "0.0.0")
            log(f"📦 Ollama version detected: {version}")
            parts = version.replace("-", ".").split(".")
            try:
                major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
                if major > 0 or (major == 0 and minor > 1) or (major == 0 and minor == 1 and patch >= 16):
                    log("✅ Ollama version supports /api/chat")
                    return True
            except (IndexError, ValueError):
                pass
            log(f"⚠️ Ollama {version} may not support /api/chat - please upgrade to >= 0.1.16")
            return False
    except Exception as e:
        log(f"⚠️ Could not check Ollama version: {e}")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 SYSTEM TOOL DEFINITIONS (For Agentic Mode)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    # Time & Math (original)
    {"type": "function", "function": {"name": "get_current_time", "description": "Get current date/time in ISO format.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "calculate", "description": "Perform math: +, -, *, /, **, %, //", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    
    # Web & Weather (simulated)
    {"type": "function", "function": {"name": "search_web", "description": "Search for information (simulated).", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get weather for location (simulated).", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    
    # 🖥️  TERMINAL COMMAND EXECUTION
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Execute a shell command in the terminal. ⚠️  Client must request user permission before calling. Returns stdout, stderr, exit_code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "cwd": {"type": "string", "description": "Working directory (optional, defaults to project root)", "default": None},
                "timeout": {"type": "integer", "description": "Command timeout in seconds", "default": 60}
            },
            "required": ["command"]
        }
    }},
    
    # 📁 FILE OPERATIONS
    {"type": "function", "function": {
        "name": "create_file",
        "description": "Create a new file with content. ⚠️  Client must request user permission before calling.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative to working directory or absolute)"},
                "content": {"type": "string", "description": "Content to write to the file"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"}
            },
            "required": ["path", "content"]
        }
    }},
    
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read contents of a file. ⚠️  Client should validate path access.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "max_bytes": {"type": "integer", "description": "Maximum bytes to read (prevent huge files)", "default": 1048576}
            },
            "required": ["path"]
        }
    }},
    
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Edit a file by replacing content or applying changes. ⚠️  Client must request user permission.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_content": {"type": "string", "description": "Content to find and replace (optional for full replace)", "default": None},
                "new_content": {"type": "string", "description": "New content to write"},
                "operation": {"type": "string", "enum": ["replace", "append", "prepend"], "description": "Edit operation type", "default": "replace"}
            },
            "required": ["path", "new_content"]
        }
    }},
    
    {"type": "function", "function": {
        "name": "delete_file",
        "description": "Delete a file. ⚠️  Client must request explicit user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to delete"}
            },
            "required": ["path"]
        }
    }},
    
    # 📂 FOLDER OPERATIONS
    {"type": "function", "function": {
        "name": "create_folder",
        "description": "Create a new directory. ⚠️  Client must request user permission.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to create"},
                "parents": {"type": "boolean", "description": "Create parent directories if needed", "default": True}
            },
            "required": ["path"]
        }
    }},
    
    {"type": "function", "function": {
        "name": "delete_folder",
        "description": "Delete a directory and its contents. ⚠️  Client must request explicit user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to delete"},
                "recursive": {"type": "boolean", "description": "Delete contents recursively", "default": True}
            },
            "required": ["path"]
        }
    }},
    
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and folders in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list", "default": "."},
                "recursive": {"type": "boolean", "description": "List recursively", "default": False}
            }
        }
    }},
]

# ─────────────────────────────────────────────────────────────────────────────
# 🔧 SYSTEM TOOL EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    @staticmethod
    def _resolve_path(path: str) -> Path:
        """Resolve path relative to WORKING_DIR or as absolute"""
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (Path(WORKING_DIR) / p).resolve()
    
    @staticmethod
    def _safe_path(path: Path, allow_outside: bool = False) -> tuple[bool, str]:
        """Check if path is safe to operate on. Returns (is_safe, message)"""
        try:
            resolved = path.resolve()
            if not allow_outside:
                work = Path(WORKING_DIR).resolve()
                try:
                    resolved.relative_to(work)
                except ValueError:
                    return False, f"Path outside working directory: {resolved} (work: {work})"
            return True, str(resolved)
        except Exception as e:
            return False, f"Path resolution error: {e}"
    
    @staticmethod
    def execute(name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return JSON result. ⚠️  Permission checks delegated to client."""
        try:
            fn = getattr(ToolExecutor, f"_{name}", None)
            if fn is None:
                return json.dumps({"error": f"Unknown tool: {name}", "available": [t["function"]["name"] for t in TOOL_DEFINITIONS]})
            
            log(f"🔧 Executing tool: {name} with args: {json.dumps(arguments, default=str)[:200]}...")
            result = fn(**arguments)
            return json.dumps(result, ensure_ascii=False, default=str)
        except PermissionError as e:
            return json.dumps({"error": "Permission denied", "details": str(e), "note": "Client should handle permission prompts"})
        except FileNotFoundError as e:
            return json.dumps({"error": "File/directory not found", "details": str(e)})
        except Exception as e:
            log(f"❌ Tool execution error: {type(e).__name__}: {e}")
            return json.dumps({"error": f"{type(e).__name__}: {e}", "traceback": str(e)})
    
    # ─── Original Tools ───
    @staticmethod
    def _get_current_time() -> Dict:
        return {"timestamp": datetime.now().isoformat(), "timezone": "UTC"}
    
    @staticmethod
    def _calculate(expression: str) -> Dict:
        try:
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return {"error": "Invalid characters in expression"}
            return {"expression": expression, "result": eval(expression, {"__builtins__": {}}, {})}
        except Exception as e:
            return {"error": f"Calculation failed: {e}"}
    
    @staticmethod
    def _search_web(query: str) -> Dict:
        return {"query": query, "results": [{"title": f"Result for '{query}'", "url": "https://example.com"}], "note": "Simulated - implement real search in production"}
    
    @staticmethod
    def _get_weather(location: str) -> Dict:
        import random
        return {"location": location, "temperature_c": random.randint(-10, 35), "condition": random.choice(["sunny", "cloudy", "rainy"]), "note": "Simulated"}
    
    # ─── 🖥️  Terminal Command ───
    @staticmethod
    def _run_command(command: str, cwd: Optional[str] = None, timeout: int = 60) -> Dict:
        """Execute shell command. ⚠️  Client MUST implement permission prompt before calling."""
        log(f"⚠️  COMMAND REQUESTED: {command}")
        
        try:
            work_dir = Path(cwd).resolve() if cwd else Path(WORKING_DIR)
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            return {
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": timeout,
                "note": "⚠️  This command was executed. Client should implement permission prompts."
            }
        except subprocess.TimeoutExpired as e:
            return {"error": "Command timed out", "command": command, "timeout": timeout, "partial_output": e.stdout.decode() if e.stdout else ""}
        except Exception as e:
            return {"error": f"Command execution failed: {type(e).__name__}: {e}"}
    
    # ─── 📁 File Operations ───
    @staticmethod
    def _create_file(path: str, content: str, encoding: str = "utf-8") -> Dict:
        """Create file with content. ⚠️  Permission checks delegated to client."""
        file_path = ToolExecutor._resolve_path(path)
        is_safe, resolved = ToolExecutor._safe_path(file_path)
        if not is_safe:
            return {"error": "Path validation failed", "details": resolved}
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved, 'w', encoding=encoding) as f:
                f.write(content)
            return {"success": True, "path": resolved, "bytes_written": len(content.encode(encoding)), "action": "created"}
        except Exception as e:
            return {"error": f"Failed to create file: {e}", "path": path}
    
    @staticmethod
    def _read_file(path: str, max_bytes: int = 1048576) -> Dict:
        """Read file contents."""
        file_path = ToolExecutor._resolve_path(path)
        is_safe, resolved = ToolExecutor._safe_path(file_path)
        if not is_safe:
            return {"error": "Path validation failed", "details": resolved}
        
        try:
            if not Path(resolved).exists():
                return {"error": "File not found", "path": path}
            if not Path(resolved).is_file():
                return {"error": "Not a file", "path": path}
            
            with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(max_bytes)
            return {
                "success": True,
                "path": resolved,
                "content": content,
                "truncated": len(content) >= max_bytes,
                "size_bytes": len(content.encode('utf-8'))
            }
        except Exception as e:
            return {"error": f"Failed to read file: {e}", "path": path}
    
    @staticmethod
    def _edit_file(path: str, new_content: str, old_content: Optional[str] = None, operation: str = "replace") -> Dict:
        """Edit file content."""
        file_path = ToolExecutor._resolve_path(path)
        is_safe, resolved = ToolExecutor._safe_path(file_path)
        if not is_safe:
            return {"error": "Path validation failed", "details": resolved}
        
        try:
            if not Path(resolved).exists():
                return {"error": "File not found", "path": path}
            
            # Read current content for replace/append operations
            with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
                current = f.read()
            
            if operation == "replace" and old_content:
                # Find and replace specific content
                if old_content not in current:
                    return {"error": "old_content not found in file", "path": path}
                updated = current.replace(old_content, new_content, 1)
            elif operation == "append":
                updated = current + new_content
            elif operation == "prepend":
                updated = new_content + current
            else:
                # Full replace
                updated = new_content
            
            with open(resolved, 'w', encoding='utf-8') as f:
                f.write(updated)
            
            return {
                "success": True,
                "path": resolved,
                "operation": operation,
                "bytes_written": len(updated.encode('utf-8')),
                "action": "edited"
            }
        except Exception as e:
            return {"error": f"Failed to edit file: {e}", "path": path}
    
    @staticmethod
    def _delete_file(path: str) -> Dict:
        """Delete a file. ⚠️  Client must confirm with user first."""
        file_path = ToolExecutor._resolve_path(path)
        is_safe, resolved = ToolExecutor._safe_path(file_path)
        if not is_safe:
            return {"error": "Path validation failed", "details": resolved}
        
        try:
            if not Path(resolved).exists():
                return {"error": "File not found", "path": path}
            if not Path(resolved).is_file():
                return {"error": "Not a file", "path": path}
            
            Path(resolved).unlink()
            return {"success": True, "path": resolved, "action": "deleted"}
        except Exception as e:
            return {"error": f"Failed to delete file: {e}", "path": path}
    
    # ─── 📂 Folder Operations ───
    @staticmethod
    def _create_folder(path: str, parents: bool = True) -> Dict:
        """Create directory."""
        dir_path = ToolExecutor._resolve_path(path)
        is_safe, resolved = ToolExecutor._safe_path(dir_path)
        if not is_safe:
            return {"error": "Path validation failed", "details": resolved}
        
        try:
            if parents:
                Path(resolved).mkdir(parents=True, exist_ok=True)
            else:
                Path(resolved).mkdir(exist_ok=False)
            return {"success": True, "path": resolved, "action": "created"}
        except FileExistsError:
            return {"success": True, "path": resolved, "action": "already_exists"}
        except Exception as e:
            return {"error": f"Failed to create folder: {e}", "path": path}
    
    @staticmethod
    def _delete_folder(path: str, recursive: bool = True) -> Dict:
        """Delete directory. ⚠️  Client must get explicit user confirmation."""
        dir_path = ToolExecutor._resolve_path(path)
        is_safe, resolved = ToolExecutor._safe_path(dir_path)
        if not is_safe:
            return {"error": "Path validation failed", "details": resolved}
        
        try:
            if not Path(resolved).exists():
                return {"error": "Directory not found", "path": path}
            if not Path(resolved).is_dir():
                return {"error": "Not a directory", "path": path}
            
            if recursive:
                shutil.rmtree(resolved)
            else:
                Path(resolved).rmdir()
            return {"success": True, "path": resolved, "action": "deleted", "recursive": recursive}
        except Exception as e:
            return {"error": f"Failed to delete folder: {e}", "path": path}
    
    @staticmethod
    def _list_directory(path: str = ".", recursive: bool = False) -> Dict:
        """List directory contents."""
        dir_path = ToolExecutor._resolve_path(path)
        is_safe, resolved = ToolExecutor._safe_path(dir_path, allow_outside=True)
        if not is_safe:
            return {"error": "Path validation failed", "details": resolved}
        
        try:
            if not Path(resolved).exists():
                return {"error": "Directory not found", "path": path}
            if not Path(resolved).is_dir():
                return {"error": "Not a directory", "path": path}
            
            items = []
            if recursive:
                for root, dirs, files in os.walk(resolved):
                    rel_root = Path(root).relative_to(resolved)
                    for d in dirs:
                        items.append({"type": "directory", "name": d, "path": str(rel_root / d)})
                    for f in files:
                        fp = rel_root / f
                        items.append({"type": "file", "name": f, "path": str(fp), "size": (Path(root) / f).stat().st_size})
            else:
                for item in Path(resolved).iterdir():
                    items.append({
                        "type": "directory" if item.is_dir() else "file",
                        "name": item.name,
                        "path": str(item.relative_to(resolved)),
                        "size": item.stat().st_size if item.is_file() else None
                    })
            
            return {"success": True, "path": resolved, "items": items, "count": len(items)}
        except Exception as e:
            return {"error": f"Failed to list directory: {e}", "path": path}

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
            resp = session.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            if resp.status_code == 200:
                log(f"💓 Ollama connection OK")
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
    payload = {
        "model": MODEL, 
        "stream": False, 
        "keep_alive": -1, 
        "messages": _oai_messages_to_ollama(messages), 
        "options": {"num_predict": max_tokens, "temperature": temperature}
    }
    if tools is not None:
        payload["tools"] = _oai_tools_to_ollama(tools)
    
    chat_url = f"{OLLAMA_BASE}/api/chat"
    log(f"🔗 Calling Ollama: {chat_url}")
    
    try:
        resp = ollama_session.post(chat_url, json=payload, timeout=(10, 300))
        if resp.status_code == 404:
            log("❌ /api/chat endpoint not found - check Ollama version (requires >= 0.1.16)")
            raise HTTPException(503, "Ollama /api/chat endpoint not found. Please upgrade Ollama to version >= 0.1.16")
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        log("⏱️ Ollama request timed out")
        raise HTTPException(504, "Ollama timed out")
    except requests.exceptions.ConnectionError:
        log(f"🔌 Cannot connect to Ollama at {OLLAMA_BASE}")
        raise HTTPException(503, f"Cannot connect to Ollama at {OLLAMA_BASE}. Is Ollama running?")
    except requests.exceptions.RequestException as e:
        log(f"❌ Ollama request failed: {e}")
        raise HTTPException(503, f"Ollama error: {e}")

def _ollama_chat_stream(messages: List[Dict], tools: Optional[List[Dict]] = None, max_tokens: int = 2048, temperature: float = 0.7):
    payload = {
        "model": MODEL, 
        "stream": True, 
        "keep_alive": -1, 
        "messages": _oai_messages_to_ollama(messages), 
        "options": {"num_predict": max_tokens, "temperature": temperature}
    }
    if tools is not None:
        payload["tools"] = _oai_tools_to_ollama(tools)
    
    chat_url = f"{OLLAMA_BASE}/api/chat"
    try:
        return ollama_session.post(chat_url, json=payload, stream=True, timeout=(10, 300))
    except requests.exceptions.RequestException as e:
        log(f"❌ Stream request failed: {e}")
        raise

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
                        yield "event: content_block_stop\n "
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
    return {
        "service": "Ollama Proxy with Tool Calling + System Tools", 
        "version": "1.0.0", 
        "model": MODEL, 
        "ollama_base": OLLAMA_BASE,
        "working_dir": WORKING_DIR,
        "tools_available": [t["function"]["name"] for t in TOOL_DEFINITIONS]
    }

@app.get("/health")
async def health():
    try:
        resp = ollama_session.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return {
                "status": "ready" if MODEL in models else "degraded", 
                "model": MODEL, 
                "model_loaded": MODEL in models, 
                "ollama_ok": True, 
                "ollama_base": OLLAMA_BASE,
                "tools_count": len(TOOL_DEFINITIONS)
            }
    except Exception as e:
        return {"status": "down", "model": MODEL, "error": str(e), "ollama_base": OLLAMA_BASE}
    return {"status": "down", "model": MODEL, "ollama_base": OLLAMA_BASE}

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
    except HTTPException:
        raise
    except Exception as e:
        log(f"❌ Chat failed: {e}")
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
        return {
            "id": f"msg_{uuid.uuid4().hex[:8]}", 
            "type": "message", 
            "role": "assistant", 
            "model": MODEL, 
            "content": content_blocks, 
            "stop_reason": stop_reason, 
            "usage": {
                "input_tokens": result.get("prompt_eval_count", 0), 
                "output_tokens": result.get("eval_count", 0)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Messages failed: {e}")

@app.post("/v1/tool/execute")
async def execute_tool(request: Request):
    """
    Execute a tool by name with arguments.
    ⚠️  PERMISSION HANDLING: This endpoint does NOT implement permission checks.
    The client application (Roo Code, Continue, etc.) MUST prompt the user for 
    confirmation BEFORE calling this endpoint with system tools.
    """
    try:
        body = await request.json()
        tool_name = body.get("name")
        arguments = body.get("arguments", {})
        if not tool_name:
            raise HTTPException(400, "Missing 'name' field")
        
        # Log tool execution for audit (client should also log)
        log(f"🔧 Tool execution request: {tool_name}")
        
        result = tool_executor.execute(tool_name, arguments)
        result_obj = json.loads(result)
        
        # Add metadata for client-side permission UI
        if tool_name in ["run_command", "delete_file", "delete_folder", "edit_file", "create_file"]:
            result_obj["requires_permission"] = True
            result_obj["permission_note"] = "Client should have prompted user before calling this tool"
        
        return {"tool": tool_name, "result": result_obj}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Tool execution failed: {e}")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str):
    return JSONResponse({
        "error": "Not found", 
        "path": f"/{path}", 
        "available": ["/", "/health", "/v1/models", "/v1/tools", "/v1/chat/completions", "/v1/messages", "/v1/tool/execute"],
        "tip": "For agentic mode, use /v1/chat/completions with tools in the request"
    }, status_code=404)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 70)
    log("  Ollama Proxy with Tool Calling + System Tools [AGENTIC MODE]")
    log("=" * 70)
    log(f"🤖 Model: {MODEL}")
    log(f"🔌 Ollama: {OLLAMA_BASE}")
    log(f"🌐 Proxy Port: {PROXY_PORT}")
    log(f"📁 Working Dir: {WORKING_DIR}")
    log(f"🔧 Tools Available: {len(TOOL_DEFINITIONS)}")
    log("⚠️  System tools require CLIENT-SIDE permission handling")
    log("=" * 70)
    
    check_ollama_version()
    
    try:
        import psutil
    except ImportError:
        log("📦 Installing psutil...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "-q"], check=True)
        import psutil
    
    _start_background_threads()
    log(f"🚀 Starting Uvicorn on port {PROXY_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="warning", timeout_keep_alive=300)
