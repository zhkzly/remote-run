#!/usr/bin/env python3
"""Standardized zero-dependency MCP stdio server wrapping remote_run.py."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

CLI = str(Path(__file__).resolve().parent / "remote_run.py")
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
STDOUT_LOCK = threading.Lock()
CONTROL_TOOLS = {"remote_hosts", "remote_cancel"}
MAX_PENDING = 32
MAX_REQUEST_CHARS = 1024 * 1024
REQUEST = threading.local()

TOOLS = [
    {
        "name": "remote_hosts",
        "description": "List configured remote compute hosts, devices, named environments (PATH, env vars, optional Python), and host-level shared assets.",
        "inputSchema": {
            "type": "object",
            "properties": {"config": {"type": "string", "description": "Optional path to hosts.toml"}},
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "remote_config",
        "description": "Validate hosts.toml and optional .remote-run.toml and return the merged remote workspace config.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to the local project directory"},
                "host": {"type": "string", "description": "Optional host name override"},
                "remote_dir": {"type": "string", "description": "Optional explicit remote folder override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["repo"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "remote_doctor",
        "description": "Check SSH connectivity, bash/tmux/rsync, GPUs, commands on PATH, optional Python env, and shared assets on a host.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to local project dir to infer host, env, and [share].host from .remote-run.toml (pass explicitly when calling via user-level MCP)"},
                "host": {"type": "string", "description": "Optional host name override"},
                "env": {"type": "string", "description": "Environment name to probe (or 'venv' / 'none')"},
                "shared": {"type": "array", "items": {"type": "string"}, "description": "Host shared asset names to verify"},
                "check_cmds": {"type": "array", "items": {"type": "string"}, "description": "Commands/tools to check on remote PATH (e.g., cargo, go, node, uv)"},
                "check_imports": {"type": "array", "items": {"type": "string"}, "description": "Python modules to test importing"},
                "check_cuda": {"type": "boolean", "description": "Probe torch.cuda.is_available()"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    },
    {
        "name": "remote_sync",
        "description": "One-way incremental sync to a remote workspace. Can overwrite source paths and prune owned files. Use dry_run to preview.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to the local project directory"},
                "host": {"type": "string", "description": "Optional host name override"},
                "remote_dir": {"type": "string", "description": "Optional explicit remote folder override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "dry_run": {"type": "boolean", "description": "Preview files that would be updated without modifying remote"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["repo"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    },
    {
        "name": "remote_exec",
        "description": "Sync local code incrementally (unless no_sync=true) and run a command or shell snippet on a remote host; detaches cleanly if longer than wait.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to local project dir (required when syncing, and still recommended with no_sync=true whenever .remote-run.toml host/env/shared settings should apply)"},
                "host": {"type": "string", "description": "Optional host name override"},
                "remote_dir": {"type": "string", "description": "Optional explicit remote folder override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "command": {"type": "array", "items": {"type": "string"}, "description": "Argv array to execute (use either command or shell)"},
                "shell": {"type": "string", "description": "Compound bash command string (use either shell or command)"},
                "no_sync": {"type": "boolean", "description": "Skip rsync upload before executing"},
                "no_project": {"type": "boolean", "description": "Ignore project config (requires no_sync)"},
                "session": {"type": "string", "description": "Opt-in persistent shell session name"},
                "wait": {"type": "number", "description": "Seconds to wait before detaching (0=async, default 15)"},
                "device": {"type": "string", "description": "Per-run device override: 'cpu' or 'cuda:N[,M...]'"},
                "no_gpu": {"type": "boolean", "description": "Skip GPU slot locks and run without CUDA allocation"},
                "env": {"type": "string", "description": "Host environment name override, 'venv', or 'none'"},
                "shared": {"type": "array", "items": {"type": "string"}, "description": "Bind host shared assets (NAME or ALIAS=NAME)"},
                "env_vars": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Extra env vars (supports {artifacts}/{shared.*})"},
                "run_id": {"type": "string", "description": "Optional run ID for idempotent retries"},
                "tail_bytes": {"type": "integer", "description": "Max trailing log bytes to return (default 8192)"},
                "raw_log": {"type": "boolean", "description": "Disable carriage-return folding in log output"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    },
    {
        "name": "remote_status",
        "description": "Check a remote run status and log slice by run_id, or list recent runs if run_id is omitted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Optional host name override"},
                "run_id": {"type": "string", "description": "Run ID to inspect (omit to list recent runs)"},
                "wait": {"type": "number", "description": "Seconds to wait for terminal status (keep short in MCP, e.g. 0-60s, to avoid client tool-call timeouts; poll or use timers for longer runs)"},
                "offset": {"type": "integer", "description": "Byte offset to read log from"},
                "tail_bytes": {"type": "integer", "description": "Max log bytes to return"},
                "raw_log": {"type": "boolean", "description": "Disable carriage-return folding"},
                "limit": {"type": "integer", "description": "Max recent runs to list"},
                "project": {"type": "string", "description": "Filter listed runs by project_id"},
                "repo": {"type": "string", "description": "Absolute path to local project dir to filter listed runs by project"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_cancel",
        "description": "Request cancellation of a queued or running job, report the observed status and retain logs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Optional host name override"},
                "run_id": {"type": "string", "description": "Run ID to cancel"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_pull",
        "description": "Pull logs, metadata, eligible artifacts, and optional workspace paths (saved under <destination>/workspace/) to a local directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Optional host name override"},
                "run_id": {"type": "string", "description": "Terminal run ID to pull from"},
                "destination": {"type": "string", "description": "Local destination path (must not exist unless force=true)"},
                "all_artifacts": {"type": "boolean", "description": "Include checkpoint files and artifacts exceeding max_file_mb"},
                "max_file_mb": {"type": "number", "description": "Max artifact size in MB when all_artifacts=false"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Relative paths inside remote workspace to pull (saved under <destination>/workspace/<path>, reported as workdir_destination)"},
                "paths_only": {"type": "boolean", "description": "Pull only requested workspace paths without re-fetching job logs/artifacts (requires paths)"},
                "force": {"type": "boolean", "description": "Merge pulled files into an existing destination directory without deleting other files"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["run_id", "destination"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_session",
        "description": "List or reset opt-in persistent shell sessions on a remote host.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "reset"], "description": "Session operation"},
                "name": {"type": "string", "description": "Session name (required when action=reset)"},
                "repo": {"type": "string", "description": "Local project directory"},
                "host": {"type": "string", "description": "Optional host name override"},
                "remote_dir": {"type": "string", "description": "Optional explicit remote folder override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_prune",
        "description": "Inspect or remove old terminal run directories on the remote host (dry-run unless apply=true).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Optional host name override"},
                "days": {"type": "number", "description": "Remove terminal runs older than this many days (default 14)"},
                "apply": {"type": "boolean", "description": "Actually delete old terminal runs (default false)"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    },
]


SCHEMAS = {tool["name"]: tool["inputSchema"] for tool in TOOLS}


def tool_error(message):
    return {"content": [{"type": "text", "text": message}], "isError": True}


def validate_argument(value, schema, label):
    kind = schema["type"]
    if kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
    elif type(value) is not {"boolean": bool, "integer": int, "string": str, "array": list, "object": dict}[kind]:
        raise ValueError(f"{label} must be a JSON {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{label} must be one of {schema['enum']}")
    if kind == "array":
        for index, entry in enumerate(value):
            validate_argument(entry, schema["items"], f"{label}[{index}]")
    elif kind == "object":
        properties = schema.get("properties", {})
        missing = schema.get("required", ())
        if any(key not in value for key in missing):
            raise ValueError(f"{label} requires {', '.join(key for key in missing if key not in value)}")
        additional = schema["additionalProperties"]
        for key, entry in value.items():
            child = properties.get(key, additional)
            if child is False:
                raise ValueError(f"Unknown argument {label}.{key}")
            validate_argument(entry, child, f"{label}.{key}")


def safe_pos(val, label):
    if not isinstance(val, str) or not val or val.startswith("-"):
        raise ValueError(f"Invalid {label}: {val!r}")
    return val


def run_cli(argv, *, timeout=120):
    stopped = getattr(REQUEST, "cancelled", None)
    if stopped is not None and stopped.is_set():
        return tool_error("Request cancelled before launch")
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        [sys.executable, "-B", CLI, *argv], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    ) as proc:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or (stopped is not None and stopped.is_set()):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    stdout, stderr = proc.communicate(timeout=1)
                except subprocess.TimeoutExpired as expired:
                    stdout = (expired.stdout or b"").decode(errors="replace")
                    stderr = (expired.stderr or b"").decode(errors="replace")
                why = "Request cancelled locally" if stopped is not None and stopped.is_set() else "Request timed out"
                return tool_error(f"{why}. {stderr.strip() or stdout.strip()}".strip())
            try:
                stdout, stderr = proc.communicate(timeout=min(remaining, 0.25))
                break
            except subprocess.TimeoutExpired:
                continue
    if proc.returncode == 0 or (proc.returncode == 2 and stdout.strip()):
        return {"content": [{"type": "text", "text": stdout.strip()}], "isError": proc.returncode != 0}
    return tool_error(stderr.strip() or stdout.strip() or f"exit {proc.returncode}")


def check_list(val, label):
    if val is None:
        return []
    if not isinstance(val, list):
        raise ValueError(f"{label} must be an array")
    return val


def check_str_map(val, label):
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ValueError(f"{label} must be an object")
    return val


def common_flags(a):
    out = []
    if a.get("config") is not None:
        out.extend(["--config", str(a["config"])])
    if a.get("host") is not None:
        out.extend(["--host", str(a["host"])])
    return out


def call_tool(name, a):
    if not isinstance(name, str) or name not in SCHEMAS:
        return tool_error(f"Unknown tool: {name!r}")
    try:
        validate_argument(a, SCHEMAS[name], name)
    except (OverflowError, ValueError) as exc:
        return tool_error(str(exc))
    if name == "remote_exec" and bool(a.get("command")) == bool(a.get("shell")):
        return tool_error("remote_exec requires exactly one non-empty command or shell")
    if name == "remote_exec" and a.get("no_project") and not a.get("no_sync"):
        return tool_error("remote_exec no_project requires no_sync")
    if name == "remote_pull" and a.get("paths_only") and not a.get("paths"):
        return tool_error("remote_pull paths_only requires paths")
    if name == "remote_hosts":
        return run_cli([*(["--config", str(a["config"])] if a.get("config") is not None else []), "hosts"])
    if name == "remote_config":
        if not a.get("repo"):
            return {"content": [{"type": "text", "text": "remote_config requires 'repo'"}], "isError": True}
        argv = [*common_flags(a), "config", "--repo", str(a["repo"])]
        for k, flag in (("remote_dir", "--remote-dir"), ("workspace", "--workspace")):
            if a.get(k) is not None:
                argv.extend([flag, str(a[k])])
        return run_cli(argv)
    if name == "remote_doctor":
        argv = [*common_flags(a), "doctor"]
        if a.get("repo") is not None:
            argv.extend(["--repo", str(a["repo"])])
        if a.get("env") is not None:
            argv.extend(["--env", str(a["env"])])
        for s in check_list(a.get("shared"), "shared"):
            argv.extend(["--shared", s])
        for c in check_list(a.get("check_cmds"), "check_cmds"):
            argv.extend(["--check-cmd", c])
        for m in check_list(a.get("check_imports"), "check_imports"):
            argv.extend(["--check-import", m])
        if a.get("check_cuda"):
            argv.append("--check-cuda")
        return run_cli(argv)
    if name == "remote_sync":
        if not a.get("repo"):
            return {"content": [{"type": "text", "text": "remote_sync requires 'repo'"}], "isError": True}
        argv = [*common_flags(a), "sync", "--repo", str(a["repo"])]
        for k, flag in (("remote_dir", "--remote-dir"), ("workspace", "--workspace")):
            if a.get(k) is not None:
                argv.extend([flag, str(a[k])])
        if a.get("dry_run"):
            argv.append("--dry-run")
        return run_cli(argv, timeout=180)
    if name == "remote_exec":
        if not a.get("no_sync") and not a.get("repo"):
            return {"content": [{"type": "text", "text": "remote_exec requires 'repo' unless no_sync=true"}], "isError": True}
        argv = [*common_flags(a), "exec"]
        if a.get("no_project"):
            argv.append("--no-project")
        for k, flag in (("repo", "--repo"), ("remote_dir", "--remote-dir"), ("workspace", "--workspace"),
                        ("session", "--session"), ("shell", "--shell"), ("device", "--device"),
                        ("env", "--env"), ("run_id", "--run-id")):
            if a.get(k) is not None:
                argv.extend([flag, str(a[k])])
        if a.get("no_sync"):
            argv.append("--no-sync")
        if a.get("no_gpu"):
            argv.append("--no-gpu")
        if a.get("wait") is not None:
            argv.extend(["--wait", str(float(a["wait"]))])
        if a.get("tail_bytes") is not None:
            argv.extend(["--tail-bytes", str(int(a["tail_bytes"]))])
        if a.get("raw_log"):
            argv.append("--raw-log")
        for s in check_list(a.get("shared"), "shared"):
            argv.extend(["--shared", s])
        for k, v in check_str_map(a.get("env_vars"), "env_vars").items():
            argv.extend(["--env-var", f"{k}={v}"])
        cmd_list = check_list(a.get("command"), "command")
        if cmd_list:
            argv.append("--")
            argv.extend(cmd_list)
        wait_val = float(a["wait"]) if a.get("wait") is not None else 15.0
        base = 360 if not a.get("no_sync") else 90
        extra = 180 if not a.get("no_sync") else 60
        return run_cli(argv, timeout=max(base, int(wait_val) + extra))
    if name == "remote_status":
        argv = [*common_flags(a), "status"]
        if a.get("run_id") is not None:
            argv.append(safe_pos(a["run_id"], "run_id"))
        if a.get("wait") is not None:
            argv.extend(["--wait", str(float(a["wait"]))])
        if a.get("offset") is not None:
            argv.extend(["--offset", str(int(a["offset"]))])
        if a.get("tail_bytes") is not None:
            argv.extend(["--tail-bytes", str(int(a["tail_bytes"]))])
        if a.get("raw_log"):
            argv.append("--raw-log")
        if a.get("limit") is not None:
            argv.extend(["--limit", str(int(a["limit"]))])
        if a.get("project") is not None:
            argv.extend(["--project", str(a["project"])])
        if a.get("repo") is not None:
            argv.extend(["--repo", str(a["repo"])])
        wait_val = float(a["wait"]) if a.get("wait") is not None else 0.0
        return run_cli(argv, timeout=max(60, int(wait_val) + 45))
    if name == "remote_cancel":
        if not a.get("run_id"):
            return {"content": [{"type": "text", "text": "remote_cancel requires 'run_id'"}], "isError": True}
        return run_cli([*common_flags(a), "cancel", safe_pos(a["run_id"], "run_id")])
    if name == "remote_pull":
        if not a.get("run_id") or not a.get("destination"):
            return {"content": [{"type": "text", "text": "remote_pull requires 'run_id' and 'destination'"}], "isError": True}
        argv = [*common_flags(a), "pull", safe_pos(a["run_id"], "run_id"), safe_pos(a["destination"], "destination")]
        if a.get("all_artifacts"):
            argv.append("--all-artifacts")
        if a.get("max_file_mb") is not None:
            argv.extend(["--max-file-mb", str(float(a["max_file_mb"]))])
        for p in check_list(a.get("paths"), "paths"):
            argv.extend(["--path", p])
        if a.get("paths_only"):
            argv.append("--paths-only")
        if a.get("force"):
            argv.append("--force")
        return run_cli(argv, timeout=360)
    if name == "remote_session":
        sub = a.get("action")
        if sub not in ("list", "reset"):
            return {"content": [{"type": "text", "text": "remote_session action must be 'list' or 'reset'"}], "isError": True}
        argv = [*common_flags(a), "session", sub]
        if sub == "reset":
            if not a.get("name"):
                return {"content": [{"type": "text", "text": "remote_session reset requires 'name'"}], "isError": True}
            argv.append(safe_pos(a["name"], "name"))
        for k, flag in (("repo", "--repo"), ("remote_dir", "--remote-dir"), ("workspace", "--workspace")):
            if a.get(k) is not None:
                argv.extend([flag, str(a[k])])
        return run_cli(argv)
    if name == "remote_prune":
        argv = [*common_flags(a), "prune"]
        if a.get("days") is not None:
            argv.extend(["--days", str(float(a["days"]))])
        if a.get("apply"):
            argv.append("--apply")
        return run_cli(argv)
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


def respond(msg_id, result=None, error=None):
    payload = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    line = json.dumps(payload, allow_nan=False) + "\n"
    with STDOUT_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


def handle_tool_call(msg_id, params, cancelled):
    REQUEST.cancelled = cancelled
    try:
        respond(msg_id, call_tool(params.get("name", ""), params.get("arguments", {})))
    except Exception as exc:
        respond(msg_id, tool_error(f"Tool error: {exc}"))
    finally:
        del REQUEST.cancelled


def main():
    capacity = threading.BoundedSemaphore(MAX_PENDING)
    control_capacity = threading.BoundedSemaphore(MAX_PENDING)
    pending = {}
    with ThreadPoolExecutor(max_workers=8) as pool, ThreadPoolExecutor(max_workers=2) as control_pool:
        while line := sys.stdin.readline(MAX_REQUEST_CHARS + 1):
            if len(line) > MAX_REQUEST_CHARS:
                while line and not line.endswith("\n"):
                    line = sys.stdin.readline(MAX_REQUEST_CHARS + 1)
                respond(None, error={"code": -32700, "message": "Request exceeds the size limit"})
                continue
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except (ValueError, RecursionError) as exc:
                respond(None, error={"code": -32700, "message": f"Parse error: {exc}"})
                continue
            if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" or not isinstance(req.get("method"), str):
                respond(None, error={"code": -32600, "message": "Invalid JSON-RPC request"})
                continue
            msg_id = req.get("id")
            method = req.get("method", "")
            params = req.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            if method == "notifications/cancelled":
                request_id = params.get("requestId")
                if type(request_id) in (int, str) and request_id in pending:
                    future, event = pending[request_id]
                    event.set()
                    if future.cancel():
                        respond(request_id, tool_error("Request cancelled before launch"))
                continue
            if method == "notifications/initialized" or msg_id is None:
                continue
            if type(msg_id) not in (str, int):
                respond(None, error={"code": -32600, "message": "Invalid JSON-RPC request ID"})
                continue
            if method == "initialize":
                req_ver = params.get("protocolVersion", "2024-11-05")
                proto = req_ver if req_ver in SUPPORTED_PROTOCOLS else "2024-11-05"
                respond(msg_id, {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "remote-run", "version": "0.5.0"},
                })
            elif method == "ping":
                respond(msg_id, {})
            elif method == "tools/list":
                respond(msg_id, {"tools": TOOLS})
            elif method == "tools/call":
                pending = {key: entry for key, entry in pending.items() if not entry[0].done()}
                if msg_id in pending:
                    respond(msg_id, error={"code": -32600, "message": "Request ID already active"})
                    continue
                name = params.get("name")
                control_tool = isinstance(name, str) and name in CONTROL_TOOLS
                slots = control_capacity if control_tool else capacity
                if not slots.acquire(blocking=False):
                    respond(msg_id, tool_error("Server busy. Retry after another request finishes."))
                    continue
                executor = control_pool if control_tool else pool
                cancelled = threading.Event()
                future = executor.submit(handle_tool_call, msg_id, params, cancelled)
                pending[msg_id] = (future, cancelled)
                future.add_done_callback(lambda _, semaphore=slots: semaphore.release())
            else:
                respond(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})


if __name__ == "__main__":
    main()
