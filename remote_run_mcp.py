#!/usr/bin/env python3
"""Zero-dependency stdio MCP server for remote-run v2 (typed service subprocess dispatch)."""
from __future__ import annotations

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

from remote_run_local import __version__

CLI = str(Path(__file__).resolve().parent / "remote_run.py")
SUPPORTED_PROTOCOLS = ("2025-06-18", "2024-11-05")
STDOUT_LOCK = threading.Lock()
CONTROL_TOOLS = {"remote_hosts", "remote_cancel"}
MAX_PENDING = 32
MAX_REQUEST_BYTES = 1024 * 1024
REQUEST = threading.local()

TOOLS = [
    {
        "name": "remote_hosts",
        "description": "List configured remote Linux hosts, roots, base PATH/env keys, and optional GPU devices.",
        "inputSchema": {
            "type": "object",
            "properties": {"config": {"type": "string", "description": "Optional path to hosts.toml"}},
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "remote_bind",
        "description": "Bind a local repository (all worktrees) or checkout (single worktree) to a remote host, GPU subset, note, PATH entries, or env vars in XDG bindings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to the local project directory"},
                "scope": {"type": "string", "enum": ["checkout", "repo"], "description": "'repo' (inherited by all worktrees) or 'checkout' (this checkout only)"},
                "host": {"type": "string", "description": "Host alias from hosts.toml"},
                "workspace": {"type": "string", "description": "Workspace ID pin (checkout scope only)"},
                "gpus": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "description": "Default GPU indices for this binding (use clear_gpus=true to reset to CPU)"},
                "clear_gpus": {"type": "boolean", "description": "Clear bound GPUs back to CPU default"},
                "note": {"type": "string", "description": "Short note describing remote preinstalled tools"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Remote PATH entries for this project on the bound host"},
                "env_vars": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Literal env vars for this project on the bound host"},
                "unset_env": {"type": "array", "items": {"type": "string"}, "description": "Env keys to remove from binding"},
                "replace": {"type": "boolean", "description": "Replace existing binding instead of merging"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["repo"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "remote_unbind",
        "description": "Remove a private repo- or checkout-scoped XDG binding.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to the local project directory"},
                "scope": {"type": "string", "enum": ["checkout", "repo"], "description": "Binding scope to remove"},
            },
            "required": ["repo"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "remote_config",
        "description": "Validate hosts.toml, .remote-run.toml, and bindings and return the merged workspace, PATH layers, redacted env origins, GPU selection, sync and pull policies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to local project dir (omit for host-only view)"},
                "host": {"type": "string", "description": "Optional host alias override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "gpus": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "description": "Optional GPU index preview (use no_gpu=true for CPU preview)"},
                "no_gpu": {"type": "boolean", "description": "Preview CPU-only execution"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Extra PATH entries to preview"},
                "env_vars": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Extra env vars to preview"},
                "max_untracked_mib": {"type": "integer", "minimum": 0, "description": "Override untracked upload budget (MiB)"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
    {
        "name": "remote_doctor",
        "description": "Read-only check of SSH connectivity, remote bash/tmux/rsync, root/workspace readiness, layered PATH, bare or literal-path commands, and optional NVIDIA GPUs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to local project dir (optional for host-only check)"},
                "host": {"type": "string", "description": "Optional host alias override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Extra PATH entries to include in probe"},
                "check_cmds": {"type": "array", "items": {"type": "string"}, "description": "Bare command names or literal paths (/..., ~/..., relative) to verify"},
                "session": {"type": "string", "description": "Optional session name to match saved cwd for relative probes"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_sync",
        "description": "One-way manifest-tracked sync to a remote workspace with untracked size budget protection and verified ghost cleanup. Use dry_run=true to preview.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to the local project directory"},
                "host": {"type": "string", "description": "Optional host alias override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "dry_run": {"type": "boolean", "description": "Preview files, budget, and ghost cleanup without modifying remote"},
                "max_untracked_mib": {"type": "integer", "minimum": 0, "description": "Override untracked upload budget (MiB, 0=unlimited)"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["repo"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    },
    {
        "name": "remote_exec",
        "description": "Sync local code (unless no_sync=true) and execute either an argv array OR a bash shell string remotely; detaches cleanly if longer than wait.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to local project dir (required for project runs; omit with no_sync=true for ad-hoc host runs)"},
                "host": {"type": "string", "description": "Optional host alias override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "remote_dir": {"type": "string", "description": "Existing remote dir (only for ad-hoc runs without repo + no_sync=true)"},
                "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Executable + arguments array (mutually exclusive with shell)"},
                "shell": {"type": "string", "minLength": 1, "description": "Explicit bash command string (mutually exclusive with argv)"},
                "no_sync": {"type": "boolean", "description": "Skip manifest upload before executing"},
                "session": {"type": "string", "description": "Opt-in persistent shell session name"},
                "wait": {"type": "number", "minimum": 0, "description": "Seconds to wait before detaching (0=async, default 15)"},
                "gpus": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "description": "Allocate NVIDIA GPU indices (use no_gpu=true for CPU)"},
                "no_gpu": {"type": "boolean", "description": "Force CPU execution without GPU locks"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Prepend remote or workspace-relative PATH directories"},
                "env_vars": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Extra literal KEY=VALUE env vars"},
                "run_id": {"type": "string", "description": "Optional run ID for idempotent retries"},
                "tail_bytes": {"type": "integer", "minimum": 0, "description": "Max trailing log bytes to return (default 8192)"},
                "raw_log": {"type": "boolean", "description": "Disable carriage-return folding"},
                "max_untracked_mib": {"type": "integer", "minimum": 0, "description": "Override untracked upload budget (MiB)"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "oneOf": [
                {"required": ["argv"], "not": {"required": ["shell"]}},
                {"required": ["shell"], "not": {"required": ["argv"]}},
            ],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    },
    {
        "name": "remote_status",
        "description": "Inspect a remote run status and UTF-8-aligned log slice by run_id, or list recent runs when run_id is omitted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Run ID to inspect (omit to list recent runs)"},
                "host": {"type": "string", "description": "Host alias from the submission receipt"},
                "repo": {"type": "string", "description": "Optional local project dir to infer host or filter listed runs"},
                "project": {"type": "string", "description": "Optional project_id filter when listing runs"},
                "wait": {"type": "number", "minimum": 0, "description": "Seconds to wait for terminal status (default 0; prefer <= 90s over MCP to avoid client call timeouts)"},
                "offset": {"type": "integer", "minimum": 0, "description": "Byte offset to read log from"},
                "tail_bytes": {"type": "integer", "minimum": 0, "description": "Max log bytes to return (default 8192)"},
                "raw_log": {"type": "boolean", "description": "Disable carriage-return folding"},
                "limit": {"type": "integer", "minimum": 1, "description": "Max recent runs to list (default 20)"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_cancel",
        "description": "Request cancellation of a running or queued remote job and return its last observed status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Run ID to cancel"},
                "host": {"type": "string", "description": "Host alias from the submission receipt"},
                "repo": {"type": "string", "description": "Optional repo to infer bound host"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_pull",
        "description": "Pull terminal run logs, metadata, eligible artifacts (size + glob policy), explicit artifacts, or workspace paths atomically to a local directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Terminal run ID to pull from"},
                "destination": {"type": "string", "description": "Local destination directory (must not exist unless force=true)"},
                "host": {"type": "string", "description": "Host alias from the submission receipt"},
                "repo": {"type": "string", "description": "Optional repo to infer bound host"},
                "max_file_mib": {"type": "integer", "minimum": 0, "description": "Override recorded artifact size cap (MiB, 0=unlimited)"},
                "artifacts": {"type": "array", "items": {"type": "string"}, "description": "Exact artifact relative paths to pull (bypasses size/exclude filters)"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Relative workspace paths pulled into <destination>/workspace/<path> (omit files already in $REMOTE_RUN_ARTIFACTS to avoid duplicate downloads)"},
                "paths_only": {"type": "boolean", "description": "Pull only workspace paths"},
                "force": {"type": "boolean", "description": "Atomically merge into existing destination directory"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["run_id", "destination"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    },
    {
        "name": "remote_session",
        "description": "List or reset opt-in persistent shell sessions on a remote host.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "reset"], "description": "Session operation"},
                "name": {"type": "string", "description": "Session name (required for reset)"},
                "host": {"type": "string", "description": "Optional host alias override"},
                "repo": {"type": "string", "description": "Optional local project dir"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "remote_dir": {"type": "string", "description": "Optional explicit remote dir"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_prune",
        "description": "Inspect (default dry-run) or remove old terminal run directories on a remote host, returning summary counts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Optional host alias override"},
                "repo": {"type": "string", "description": "Optional repo to infer bound host"},
                "days": {"type": "number", "minimum": 0, "description": "Age threshold in days (default 14)"},
                "apply": {"type": "boolean", "description": "Actually delete eligible runs (default false = dry-run)"},
                "verbose": {"type": "boolean", "description": "Include full skipped_unmanaged list instead of 20-item sample"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
    },
]

TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


class RequestCancelled(Exception):
    pass


class RequestSlot:
    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None

    def attach(self, proc: subprocess.Popen) -> None:
        with self.lock:
            self.proc = proc
            cancelled = self.cancelled.is_set()
        if cancelled:
            terminate_process(proc)

    def detach(self, proc: subprocess.Popen) -> None:
        with self.lock:
            if self.proc is proc:
                self.proc = None

    def cancel(self) -> None:
        with self.lock:
            self.cancelled.set()
            proc = self.proc
        if proc is not None:
            terminate_process(proc)


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            return
    deadline = time.monotonic() + 1.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass


def send(msg: dict) -> None:
    with STDOUT_LOCK:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def ok(req_id, result: dict) -> None:
    send({"jsonrpc": "2.0", "id": req_id, "result": result})


def err(req_id, code: int, message: str) -> None:
    send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def check_allowed_keys(args: dict, tool_name: str) -> None:
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object")
    allowed = set(TOOLS_BY_NAME[tool_name]["inputSchema"]["properties"])
    unknown = set(args) - allowed
    if unknown:
        raise ValueError(f"Unknown argument(s) for {tool_name}: {', '.join(sorted(unknown))}")


def str_arg(args: dict, key: str, *, required: bool = False) -> str | None:
    if key not in args or args[key] is None:
        if required:
            raise ValueError(f"{key} is required")
        return None
    val = args[key]
    if not isinstance(val, str) or (required and not val):
        raise ValueError(f"{key} must be a non-empty string" if required else f"{key} must be a string")
    return val


def bool_arg(args: dict, key: str, default: bool = False) -> bool:
    if key not in args or args[key] is None:
        return default
    val = args[key]
    if not isinstance(val, bool):
        raise ValueError(f"{key} must be a boolean")
    return val


def int_arg(args: dict, key: str, *, minimum: int = 0) -> int | None:
    if key not in args or args[key] is None:
        return None
    val = args[key]
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return val


def num_arg(args: dict, key: str, default: float) -> float:
    if key not in args or args[key] is None:
        return default
    val = args[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
        raise ValueError(f"{key} must be a finite non-negative number")
    return float(val)


def str_list_arg(args: dict, key: str) -> list[str] | None:
    if key not in args or args[key] is None:
        return None
    val = args[key]
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise ValueError(f"{key} must be an array of strings")
    return list(val)


def int_list_arg(args: dict, key: str, *, non_empty: bool = False) -> list[int] | None:
    if key not in args or args[key] is None:
        return None
    val = args[key]
    if not isinstance(val, list) or not all(not isinstance(x, bool) and isinstance(x, int) and x >= 0 for x in val):
        raise ValueError(f"{key} must be an array of non-negative integers")
    if non_empty and not val:
        raise ValueError(f"{key} must be non-empty (use no_gpu=true or clear_gpus=true for CPU)")
    return list(val)


def env_map_arg(args: dict, key: str) -> dict[str, str] | None:
    if key not in args or args[key] is None:
        return None
    val = args[key]
    if not isinstance(val, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in val.items()):
        raise ValueError(f"{key} must be an object mapping strings to strings")
    return dict(val)


def run_service_subprocess(op: str, params: dict, timeout: float = 180) -> tuple[int, str, str]:
    slot = getattr(REQUEST, "slot", None)
    if slot is not None and slot.cancelled.is_set():
        raise RequestCancelled("Request cancelled")
    payload = json.dumps({"op": op, "params": params}, ensure_ascii=False)
    proc = subprocess.Popen(
        [sys.executable, "-B", CLI, "_service_rpc"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    if slot is not None:
        slot.attach(proc)
    try:
        try:
            stdout, stderr = proc.communicate(input=payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_process(proc)
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"{op} timed out after {int(timeout)}s")
    finally:
        if slot is not None:
            slot.detach(proc)
    if slot is not None and slot.cancelled.is_set():
        raise RequestCancelled("Request cancelled")
    return proc.returncode, stdout.strip(), stderr.strip()


def translate_tool(name: str, args: dict) -> tuple[str, dict, float]:
    check_allowed_keys(args, name)
    timeout = 180.0
    if name == "remote_hosts":
        return "hosts", {"config": str_arg(args, "config")}, 30.0
    if name == "remote_bind":
        return "bind", {
            "repo": str_arg(args, "repo", required=True),
            "scope": str_arg(args, "scope") or "checkout",
            "host": str_arg(args, "host") or "",
            "workspace": str_arg(args, "workspace") or "",
            "gpus": int_list_arg(args, "gpus", non_empty=True),
            "clear_gpus": bool_arg(args, "clear_gpus"),
            "note": str_arg(args, "note"),
            "paths": str_list_arg(args, "paths"),
            "env_vars": env_map_arg(args, "env_vars"),
            "unset_env": str_list_arg(args, "unset_env") or [],
            "replace": bool_arg(args, "replace"),
            "config": str_arg(args, "config"),
        }, 30.0
    if name == "remote_unbind":
        return "unbind", {
            "repo": str_arg(args, "repo", required=True),
            "scope": str_arg(args, "scope") or "checkout",
        }, 30.0
    if name == "remote_config":
        return "config", {
            "repo": str_arg(args, "repo"),
            "host": str_arg(args, "host"),
            "workspace": str_arg(args, "workspace") or "",
            "gpus": int_list_arg(args, "gpus", non_empty=True),
            "no_gpu": bool_arg(args, "no_gpu"),
            "paths": str_list_arg(args, "paths") or [],
            "env_vars": env_map_arg(args, "env_vars"),
            "max_untracked_mib": int_arg(args, "max_untracked_mib"),
            "config": str_arg(args, "config"),
        }, 30.0
    if name == "remote_doctor":
        return "doctor", {
            "repo": str_arg(args, "repo"),
            "host": str_arg(args, "host"),
            "workspace": str_arg(args, "workspace") or "",
            "paths": str_list_arg(args, "paths") or [],
            "check_cmds": str_list_arg(args, "check_cmds") or [],
            "session": str_arg(args, "session") or "",
            "config": str_arg(args, "config"),
        }, 90.0
    if name == "remote_sync":
        return "sync", {
            "repo": str_arg(args, "repo", required=True),
            "host": str_arg(args, "host"),
            "workspace": str_arg(args, "workspace") or "",
            "dry_run": bool_arg(args, "dry_run"),
            "max_untracked_mib": int_arg(args, "max_untracked_mib"),
            "config": str_arg(args, "config"),
        }, 300.0
    if name == "remote_exec":
        repo = str_arg(args, "repo")
        no_sync = bool_arg(args, "no_sync")
        no_project = repo is None
        if no_project and not no_sync:
            raise ValueError("remote_exec requires 'repo' when no_sync is false; pass no_sync=true for ad-hoc runs")
        # Reject explicit null or simultaneous key presence (matches JSON Schema `oneOf`).
        has_argv_key = "argv" in args
        has_shell_key = "shell" in args
        if has_argv_key and has_shell_key:
            raise ValueError("Provide either 'argv' OR 'shell', never both")
        if not has_argv_key and not has_shell_key:
            raise ValueError("Provide either non-empty 'argv' or non-empty 'shell'")
        argv = str_list_arg(args, "argv") if has_argv_key else None
        shell = str_arg(args, "shell", required=True) if has_shell_key else None
        if has_argv_key and not argv:
            raise ValueError("argv must be a non-empty array of strings")
        shell_str = shell or ""
        wait = num_arg(args, "wait", 15.0)
        timeout = max(300.0, wait + 180.0)
        return "exec", {
            "repo": repo,
            "host": str_arg(args, "host"),
            "workspace": str_arg(args, "workspace") or "",
            "remote_dir": str_arg(args, "remote_dir") or "",
            "argv": argv,
            "shell": shell_str,
            "no_sync": no_sync,
            "no_project": no_project,
            "session": str_arg(args, "session") or "",
            "wait": wait,
            "gpus": int_list_arg(args, "gpus", non_empty=True),
            "no_gpu": bool_arg(args, "no_gpu"),
            "paths": str_list_arg(args, "paths") or [],
            "env_vars": env_map_arg(args, "env_vars"),
            "run_id": str_arg(args, "run_id") or "",
            "tail_bytes": int_arg(args, "tail_bytes") if "tail_bytes" in args else 8192,
            "raw_log": bool_arg(args, "raw_log"),
            "max_untracked_mib": int_arg(args, "max_untracked_mib"),
            "config": str_arg(args, "config"),
            "emit_banner": False,
        }, timeout
    if name == "remote_status":
        wait = num_arg(args, "wait", 0.0)
        return "status", {
            "run_id": str_arg(args, "run_id"),
            "host": str_arg(args, "host"),
            "repo": str_arg(args, "repo"),
            "project": str_arg(args, "project") or "",
            "wait": wait,
            "offset": int_arg(args, "offset"),
            "tail_bytes": int_arg(args, "tail_bytes") if "tail_bytes" in args else 8192,
            "raw_log": bool_arg(args, "raw_log"),
            "limit": int_arg(args, "limit", minimum=1) if "limit" in args else 20,
            "config": str_arg(args, "config"),
        }, max(90.0, wait + 60.0)
    if name == "remote_cancel":
        return "cancel", {
            "run_id": str_arg(args, "run_id", required=True),
            "host": str_arg(args, "host"),
            "repo": str_arg(args, "repo"),
            "config": str_arg(args, "config"),
        }, 60.0
    if name == "remote_pull":
        return "pull", {
            "run_id": str_arg(args, "run_id", required=True),
            "destination": str_arg(args, "destination", required=True),
            "host": str_arg(args, "host"),
            "repo": str_arg(args, "repo"),
            "max_file_mib": int_arg(args, "max_file_mib"),
            "artifacts": str_list_arg(args, "artifacts") or [],
            "paths": str_list_arg(args, "paths") or [],
            "paths_only": bool_arg(args, "paths_only"),
            "force": bool_arg(args, "force"),
            "config": str_arg(args, "config"),
        }, 600.0
    if name == "remote_session":
        action = str_arg(args, "action", required=True)
        if action not in ("list", "reset"):
            raise ValueError("action must be 'list' or 'reset'")
        return "session", {
            "action": action,
            "name": str_arg(args, "name", required=(action == "reset")) or "",
            "host": str_arg(args, "host"),
            "repo": str_arg(args, "repo"),
            "workspace": str_arg(args, "workspace") or "",
            "remote_dir": str_arg(args, "remote_dir") or "",
            "config": str_arg(args, "config"),
        }, 60.0
    if name == "remote_prune":
        return "prune", {
            "host": str_arg(args, "host"),
            "repo": str_arg(args, "repo"),
            "days": num_arg(args, "days", 14.0),
            "apply": bool_arg(args, "apply"),
            "verbose": bool_arg(args, "verbose"),
            "config": str_arg(args, "config"),
        }, 120.0
    raise KeyError(f"Unknown tool: {name}")


def call_tool(name: str, args: dict) -> dict:
    op, params, timeout = translate_tool(name, args)
    rc, stdout, stderr = run_service_subprocess(op, params, timeout=timeout)
    if rc in (0, 2) and stdout:
        try:
            parsed = json.loads(stdout)
            return {
                "content": [{"type": "text", "text": json.dumps(parsed, ensure_ascii=False, indent=2)}],
                "structuredContent": parsed,
                "isError": rc == 2,
            }
        except json.JSONDecodeError:
            pass
    msg = stderr or stdout or f"{op} failed with exit code {rc}"
    return {"content": [{"type": "text", "text": msg}], "isError": True}


def handle_tool_call(req_id, params: dict, slot: RequestSlot, inflight: dict, lock: threading.Lock) -> None:
    REQUEST.slot = slot
    try:
        name = params.get("name")
        if not isinstance(name, str) or name not in TOOLS_BY_NAME:
            err(req_id, -32602, f"Unknown tool: {name!r}")
            return
        result = call_tool(name, params.get("arguments") or {})
        if not slot.cancelled.is_set():
            ok(req_id, result)
    except RequestCancelled:
        pass
    except (ValueError, KeyError) as exc:
        if not slot.cancelled.is_set():
            ok(req_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
    except Exception as exc:
        if not slot.cancelled.is_set():
            ok(req_id, {"content": [{"type": "text", "text": f"remote-run MCP error: {exc}"}], "isError": True})
    finally:
        REQUEST.slot = None
        with lock:
            if inflight.get(req_id) is slot:
                inflight.pop(req_id, None)


def _read_bounded_line(stream) -> tuple[bytes | None, bool]:
    """Read one newline-delimited message bounded by MAX_REQUEST_BYTES without unbounded buffering."""
    chunk = stream.readline(MAX_REQUEST_BYTES + 1)
    if not chunk:
        return None, False
    if len(chunk) > MAX_REQUEST_BYTES and not chunk.endswith(b"\n"):
        # Drain the remainder of the oversized line without holding it in memory.
        while True:
            extra = stream.readline(65536)
            if not extra or extra.endswith(b"\n"):
                break
        return b"", True
    return chunk, len(chunk) > MAX_REQUEST_BYTES


def main() -> None:
    general_pool = ThreadPoolExecutor(max_workers=8)
    control_pool = ThreadPoolExecutor(max_workers=4)
    cancel_pool = ThreadPoolExecutor(max_workers=4)
    inflight: dict = {}
    inflight_lock = threading.Lock()
    interrupted = False
    stdin_bin = sys.stdin.buffer
    try:
        while True:
            raw_bytes, oversized = _read_bounded_line(stdin_bin)
            if raw_bytes is None:
                break
            if oversized:
                err(None, -32600, "Request too large")
                continue
            try:
                line = raw_bytes.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                err(None, -32700, f"Invalid UTF-8: {exc}")
                continue
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                err(None, -32700, f"Parse error: {exc}")
                continue
            if not isinstance(msg, dict):
                err(None, -32600, "Invalid Request")
                continue
            method = msg.get("method")
            req_id = msg.get("id")
            params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

            if method == "notifications/initialized":
                continue
            if method == "notifications/cancelled":
                cancel_id = params.get("requestId")
                if isinstance(cancel_id, (str, int)) and not isinstance(cancel_id, bool):
                    with inflight_lock:
                        slot = inflight.get(cancel_id)
                    if slot is not None:
                        # Mark cancelled immediately and terminate subprocess asynchronously so stdin loop never blocks.
                        slot.cancelled.set()
                        cancel_pool.submit(slot.cancel)
                continue
            if method == "initialize":
                if not isinstance(req_id, (str, int)) or isinstance(req_id, bool):
                    err(None, -32600, "initialize requires a string or integer id")
                    continue
                req_ver = params.get("protocolVersion")
                proto = req_ver if req_ver in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
                ok(req_id, {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "remote-run", "version": __version__},
                })
                continue
            if method == "ping":
                if not isinstance(req_id, (str, int)) or isinstance(req_id, bool):
                    err(None, -32600, "ping requires a string or integer id")
                    continue
                ok(req_id, {})
                continue
            if method == "tools/list":
                if not isinstance(req_id, (str, int)) or isinstance(req_id, bool):
                    err(None, -32600, "tools/list requires a string or integer id")
                    continue
                ok(req_id, {"tools": TOOLS})
                continue
            if method == "tools/call":
                if not isinstance(req_id, (str, int)) or isinstance(req_id, bool):
                    err(None, -32600, "tools/call requires a string or integer id")
                    continue
                if "arguments" in params and not isinstance(params["arguments"], dict):
                    err(req_id, -32602, "tools/call arguments must be a JSON object")
                    continue
                tool_name = params.get("name")
                if not isinstance(tool_name, str) or tool_name not in TOOLS_BY_NAME:
                    err(req_id, -32602, f"Unknown tool: {tool_name!r}")
                    continue
                with inflight_lock:
                    if req_id in inflight:
                        err(req_id, -32600, f"Duplicate in-flight request id: {req_id!r}")
                        continue
                    if len(inflight) >= MAX_PENDING:
                        err(req_id, -32000, "Too many concurrent requests")
                        continue
                    slot = RequestSlot()
                    inflight[req_id] = slot
                pool = control_pool if tool_name in CONTROL_TOOLS else general_pool
                pool.submit(handle_tool_call, req_id, params, slot, inflight, inflight_lock)
                continue
            if req_id is not None:
                err(req_id, -32601, f"Method not found: {method}")
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if interrupted:
            with inflight_lock:
                slots = list(inflight.values())
            for slot in slots:
                slot.cancel()
            general_pool.shutdown(wait=True, cancel_futures=True)
            control_pool.shutdown(wait=True, cancel_futures=True)
            cancel_pool.shutdown(wait=True, cancel_futures=True)
        else:
            general_pool.shutdown(wait=True)
            control_pool.shutdown(wait=True)
            cancel_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
