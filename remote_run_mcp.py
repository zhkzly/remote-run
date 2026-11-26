#!/usr/bin/env python3
"""Standardized zero-dependency MCP stdio server wrapping remote_run.py."""
import json
from pathlib import Path
import subprocess
import sys

CLI = str(Path(__file__).resolve().parent / "remote_run.py")
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

TOOLS = [
    {
        "name": "remote_hosts",
        "description": "List configured remote compute hosts, devices, Python environments, and host-level shared assets.",
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
        "description": "Check SSH connectivity, git/tmux/rsync, GPU availability, Python env, and shared assets on a host.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Optional host name override"},
                "env": {"type": "string", "description": "Environment name to probe (or 'venv' / 'none')"},
                "shared": {"type": "array", "items": {"type": "string"}, "description": "Host shared asset names to verify"},
                "check_imports": {"type": "array", "items": {"type": "string"}, "description": "Python modules to test importing"},
                "check_cuda": {"type": "boolean", "description": "Probe torch.cuda.is_available()"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_sync",
        "description": "One-way additive rsync from local repository to its remote workspace (never deletes remote files).",
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
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_exec",
        "description": "Sync local code additively (unless no_sync=true) and run a command or shell snippet on a remote host; detaches cleanly if longer than wait.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Absolute path to local project dir (required unless no_sync=true)"},
                "host": {"type": "string", "description": "Optional host name override"},
                "remote_dir": {"type": "string", "description": "Optional explicit remote folder override"},
                "workspace": {"type": "string", "description": "Optional workspace ID override"},
                "command": {"type": "array", "items": {"type": "string"}, "description": "Argv array to execute (use either command or shell)"},
                "shell": {"type": "string", "description": "Compound bash command string (use either shell or command)"},
                "no_sync": {"type": "boolean", "description": "Skip rsync upload before executing"},
                "session": {"type": "string", "description": "Opt-in persistent shell session name"},
                "wait": {"type": "number", "description": "Seconds to wait before detaching (0=async, default 15)"},
                "no_gpu": {"type": "boolean", "description": "Skip GPU slot lock for quick inspection"},
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
                "wait": {"type": "number", "description": "Seconds to wait for terminal status"},
                "offset": {"type": "integer", "description": "Byte offset to read log from"},
                "tail_bytes": {"type": "integer", "description": "Max log bytes to return"},
                "raw_log": {"type": "boolean", "description": "Disable carriage-return folding"},
                "limit": {"type": "integer", "description": "Max recent runs to list"},
                "project": {"type": "string", "description": "Filter listed runs by project_id"},
                "repo": {"type": "string", "description": "Filter listed runs by local project directory"},
                "config": {"type": "string", "description": "Optional path to hosts.toml"},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
    },
    {
        "name": "remote_cancel",
        "description": "Cancel a queued or running remote job cleanly (kills process group + tmux session, keeps logs).",
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
        "description": "Pull logs, metadata, eligible artifacts, and optional workspace paths to a local directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Optional host name override"},
                "run_id": {"type": "string", "description": "Terminal run ID to pull from"},
                "destination": {"type": "string", "description": "Local destination path (must not exist unless force=true)"},
                "all_artifacts": {"type": "boolean", "description": "Include checkpoint files and artifacts exceeding max_file_mb"},
                "max_file_mb": {"type": "number", "description": "Max artifact size in MB when all_artifacts=false"},
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Relative paths inside remote workspace to pull"},
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


def safe_pos(val, label):
    if not isinstance(val, str) or not val or val.startswith("-"):
        raise ValueError(f"Invalid {label}: {val!r}")
    return val


def run_cli(argv, *, timeout=120):
    proc = subprocess.run([sys.executable, "-B", CLI, *argv], capture_output=True, text=True, timeout=timeout)
    if proc.returncode == 0:
        return {"content": [{"type": "text", "text": proc.stdout.strip()}], "isError": False}
    if proc.returncode == 2 and proc.stdout.strip():
        return {"content": [{"type": "text", "text": proc.stdout.strip()}], "isError": True}
    msg = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
    return {"content": [{"type": "text", "text": msg}], "isError": True}


def common_flags(a):
    out = []
    if a.get("config") is not None:
        out.extend(["--config", str(a["config"])])
    if a.get("host") is not None:
        out.extend(["--host", str(a["host"])])
    return out


def call_tool(name, a):
    if a is None:
        a = {}
    if not isinstance(a, dict):
        return {"content": [{"type": "text", "text": "Tool arguments must be a JSON object"}], "isError": True}
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
        if a.get("env") is not None:
            argv.extend(["--env", str(a["env"])])
        for s in (a.get("shared") or []):
            argv.extend(["--shared", str(s)])
        for m in (a.get("check_imports") or []):
            argv.extend(["--check-import", str(m)])
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
        if a.get("no_sync") and not a.get("repo"):
            argv.append("--no-project")
        for k, flag in (("repo", "--repo"), ("remote_dir", "--remote-dir"), ("workspace", "--workspace"),
                        ("session", "--session"), ("shell", "--shell"), ("env", "--env"), ("run_id", "--run-id")):
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
        for s in (a.get("shared") or []):
            argv.extend(["--shared", str(s)])
        for k, v in (a.get("env_vars") or {}).items():
            argv.extend(["--env-var", f"{k}={v}"])
        if a.get("command"):
            argv.append("--")
            argv.extend(str(x) for x in a["command"])
        wait_val = float(a["wait"]) if a.get("wait") is not None else 15.0
        return run_cli(argv, timeout=max(90, int(wait_val) + 60))
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
        for p in (a.get("paths") or []):
            argv.extend(["--path", str(p)])
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
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            respond(None, error={"code": -32700, "message": f"Parse error: {exc}"})
            continue
        if not isinstance(req, dict):
            respond(None, error={"code": -32600, "message": "Invalid Request: expected JSON object"})
            continue
        msg_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        if method in ("notifications/initialized", "notifications/cancelled") or msg_id is None:
            continue
        if method == "initialize":
            req_ver = params.get("protocolVersion", "2024-11-05")
            proto = req_ver if req_ver in SUPPORTED_PROTOCOLS else "2024-11-05"
            respond(msg_id, {
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "remote-run", "version": "0.3.0"},
            })
        elif method == "ping":
            respond(msg_id, {})
        elif method == "tools/list":
            respond(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            try:
                respond(msg_id, call_tool(params.get("name", ""), params.get("arguments", {})))
            except (KeyError, TypeError, ValueError, OverflowError, OSError, subprocess.TimeoutExpired) as exc:
                respond(msg_id, {"content": [{"type": "text", "text": f"Tool error: {exc}"}], "isError": True})
        else:
            respond(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})


if __name__ == "__main__":
    main()
