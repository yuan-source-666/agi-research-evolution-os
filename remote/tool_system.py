"""
AGI Tool System v0.1
====================
Standardized tool interface: discover, register, select, execute, log
Compatible with MCP protocol concepts, simplified as standalone Python.

Dependencies: standard library only
Design principles:
  - Each tool has Schema / Permission / Timeout / Logging / Error Handling
  - Support tool discovery, selection, composition
  - Execution results recorded to Memory System
  - Extensible (agent can register new tools)
"""

import json
import time
import os
import subprocess
import threading
import hashlib
import inspect
from typing import Any, Optional, Callable
from dataclasses import dataclass, field, asdict


@dataclass
class ToolSchema:
    """Tool parameter schema."""
    name: str
    description: str
    parameters: dict  # JSON Schema format
    returns: dict = field(default_factory=lambda: {"type": "string"})


@dataclass
class Tool:
    """A tool definition."""
    id: str
    name: str
    description: str
    schema: dict           # parameter schema
    handler: str = ""     # handler identifier (function name or command)
    permission: str = "user"   # user/admin/sandbox
    timeout: int = 30       # seconds
    version: str = "0.1"
    enabled: bool = True
    call_count: int = 0
    last_used: float = 0.0
    avg_latency: float = 0.0
    error_rate: float = 0.0


class ToolSystem:
    """AGI Tool System."""

    def __init__(self, memory=None):
        self.tools: dict[str, Tool] = {}
        self.handlers: dict[str, Callable] = {}  # name -> callable
        self.log: list[dict] = []  # execution log
        self.memory = memory  # optional MemorySystem reference
        self.lock = threading.RLock()

        # Register built-in tools
        self._register_builtin()

    def _register_builtin(self):
        """Register built-in tools."""

        # 1. Code execution
        self.register(
            name="code_exec",
            description="Execute Python code and return output",
            schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code"}
                },
                "required": ["code"]
            },
            handler=self._tool_code_exec,
            permission="sandbox",
            timeout=30
        )

        # 2. Shell execution
        self.register(
            name="shell_exec",
            description="Execute Shell command and return output",
            schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command"}
                },
                "required": ["command"]
            },
            handler=self._tool_shell_exec,
            permission="sandbox",
            timeout=30
        )

        # 3. File read/write
        self.register(
            name="file_read",
            description="Read file content",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            },
            handler=self._tool_file_read,
            permission="user",
            timeout=10
        )

        self.register(
            name="file_write",
            description="Write file",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"}
                },
                "required": ["path", "content"]
            },
            handler=self._tool_file_write,
            permission="user",
            timeout=10
        )

        # 4. HTTP request
        self.register(
            name="http_get",
            description="Send HTTP GET request",
            schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL"},
                    "headers": {"type": "object", "description": "Request headers"}
                },
                "required": ["url"]
            },
            handler=self._tool_http_get,
            permission="user",
            timeout=30
        )

        # 5. Memory storage
        if self.memory:
            self.register(
                name="memory_store",
                description="Store episodic memory",
                schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "importance": {"type": "number", "description": "Importance 0-1"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["content"]
                },
                handler=self._tool_memory_store,
                permission="user",
                timeout=5
            )

            self.register(
                name="memory_retrieve",
                description="Retrieve memories",
                schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query text"},
                        "top_k": {"type": "integer", "description": "Number of results"}
                    },
                    "required": ["query"]
                },
                handler=self._tool_memory_retrieve,
                permission="user",
                timeout=5
            )

    def register(self, name, description, schema, handler,
                 permission="user", timeout=30, version="0.1"):
        """Register a tool."""
        with self.lock:
            tool = Tool(
                id=hashlib.md5(name.encode()).hexdigest()[:12],
                name=name,
                description=description,
                schema=schema,
                handler=name,
                permission=permission,
                timeout=timeout,
                version=version
            )
            self.tools[name] = tool
            self.handlers[name] = handler

    def discover(self):
        """Discover all available tools."""
        with self.lock:
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "schema": t.schema,
                    "permission": t.permission,
                    "version": t.version
                }
                for t in self.tools.values() if t.enabled
            ]

    def select(self, task_description: str, top_k=5):
        """Select most relevant tools for a task (simple keyword matching)."""
        with self.lock:
            scores = []
            for name, tool in self.tools.items():
                if not tool.enabled:
                    continue
                keywords = set(tool.name.split("_") + tool.description.split())
                task_words = set(task_description.lower().split())
                overlap = len(keywords & task_words)
                freq_score = min(tool.call_count / 10, 0.5)
                score = overlap + freq_score
                scores.append((score, tool))
            scores.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scores[:top_k]]

    def execute(self, tool_name: str, params: dict) -> dict:
        """Execute a tool."""
        with self.lock:
            if tool_name not in self.tools:
                return {"error": f"Tool '{tool_name}' not found", "available": list(self.tools.keys())}

            tool = self.tools[tool_name]
            if not tool.enabled:
                return {"error": f"Tool '{tool_name}' is disabled"}

            handler = self.handlers.get(tool_name)
            if not handler:
                return {"error": f"No handler for '{tool_name}'"}

        # Execute outside lock to avoid blocking
        start = time.perf_counter()
        try:
            result = handler(params)
            latency = time.perf_counter() - start

            with self.lock:
                tool.call_count += 1
                tool.last_used = time.time()
                tool.avg_latency = (tool.avg_latency * (tool.call_count - 1) + latency) / tool.call_count
                if isinstance(result, dict) and result.get("error"):
                    tool.error_rate = (tool.error_rate * (tool.call_count - 1) + 1) / tool.call_count

                log_entry = {
                    "tool": tool_name,
                    "params": str(params)[:200],
                    "latency": round(latency, 3),
                    "success": not (isinstance(result, dict) and result.get("error")),
                    "timestamp": time.time()
                }
                self.log.append(log_entry)

                if self.memory and tool_name not in ("memory_store", "memory_retrieve"):
                    self.memory.store_working(
                        f"Tool call: {tool_name} -> {'success' if log_entry['success'] else 'failed'}",
                        importance=0.3
                    )

            return result

        except Exception as e:
            latency = time.perf_counter() - start
            with self.lock:
                tool.call_count += 1
                tool.error_rate = (tool.error_rate * (tool.call_count - 1) + 1) / tool.call_count
                self.log.append({
                    "tool": tool_name,
                    "params": str(params)[:200],
                    "latency": round(latency, 3),
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                })
            return {"error": str(e), "tool": tool_name}

    # ========== Built-in tool implementations ==========

    def _tool_code_exec(self, params):
        """Execute Python code."""
        code = params.get("code", "")
        if not code:
            return {"error": "No code provided"}

        try:
            import io
            from contextlib import redirect_stdout, redirect_stderr
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            local_ns = {}
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, {"__builtins__": __builtins__}, local_ns)

            return {
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
                "result": str(local_ns.get("result", ""))
            }
        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    def _tool_shell_exec(self, params):
        """Execute Shell command."""
        cmd = params.get("command", "")
        if not cmd:
            return {"error": "No command provided"}

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out", "timeout": 30}
        except Exception as e:
            return {"error": str(e)}

    def _tool_file_read(self, params):
        path = params.get("path", "")
        if not path or not os.path.exists(path):
            return {"error": f"File not found: {path}"}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {"content": f.read(), "path": path}
        except Exception as e:
            return {"error": str(e)}

    def _tool_file_write(self, params):
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return {"error": "No path provided"}
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "ok", "path": path, "bytes": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def _tool_http_get(self, params):
        url = params.get("url", "")
        if not url:
            return {"error": "No URL provided"}
        try:
            import urllib.request
            req = urllib.request.Request(url, headers=params.get("headers", {}))
            resp = urllib.request.urlopen(req, timeout=30)
            return {
                "status": resp.status,
                "body": resp.read().decode("utf-8", errors="replace")[:10000],
                "url": url
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_memory_store(self, params):
        if not self.memory:
            return {"error": "No memory system"}
        content = params.get("content", "")
        importance = params.get("importance", 0.5)
        tags = params.get("tags", [])
        mid = self.memory.store_episodic(content, source="tool", importance=importance, tags=tags)
        return {"status": "ok", "memory_id": mid}

    def _tool_memory_retrieve(self, params):
        if not self.memory:
            return {"error": "No memory system"}
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        results = self.memory.retrieve(query, top_k=top_k)
        return {
            "results": [
                {"score": round(s, 4), "content": item.content, "type": item.type}
                for s, item in results
            ]
        }

    # ========== Statistics ==========

    def stats(self):
        with self.lock:
            return {
                "total_tools": len(self.tools),
                "enabled_tools": sum(1 for t in self.tools.values() if t.enabled),
                "total_calls": sum(t.call_count for t in self.tools.values()),
                "avg_error_rate": sum(t.error_rate for t in self.tools.values()) / max(len(self.tools), 1),
                "recent_logs": len(self.log[-100:])
            }

    def get_log(self, n=20):
        with self.lock:
            return self.log[-n:]
