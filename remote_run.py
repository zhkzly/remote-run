#!/usr/bin/env python3
"""Cross-project persistent remote shell, smart sync, and GPU job runner."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from uuid import uuid4


SSH = [
    "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=7",
    "-o", "ConnectionAttempts=1", "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
    "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
    "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
]
TERMINAL = {"completed", "failed", "interrupted", "cancelled"}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx", ".h5"}
ROOT_MARKER = ".remote-run-root.json"
WORKSPACE_MARKER = ".remote-run-workspace.json"
BUILTIN_IGNORE = [
    ".venv/", "venv/", "__pycache__/", "*.pyc",
    ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", "node_modules/", ".DS_Store", "/results/",
]
BUILTIN_PRESERVE = [".venv", "outputs", "checkpoints", ".cache"]
DEFAULT_CONFIG = Path.home() / ".config" / "remote-run" / "hosts.toml"
SKIP_SESSION_ENV = {"_", "PWD", "OLDPWD", "SHLVL", "BASH_EXECUTION_STRING", "PPID"}
SKIP_SESSION_PREFIXES = ("TMUX", "REMOTE_RUN_", "__RR_")

HOST_TOP_KEYS = {"schema_version", "default_host", "hosts"}
HOST_ENTRY_KEYS = {"ssh", "root", "device", "gpu_slot", "locks_dir", "default_env", "path_prepend", "envs", "shared"}
HOST_ENV_KEYS = {"python", "path_prepend", "env"}
HOST_SHARED_KEYS = {"path", "manifest", "sha256", "version"}
PROJECT_TOP_KEYS = {"schema_version", "project", "host", "env", "workspace", "remote_dir", "sync", "share", "env_vars"}
PROJECT_SYNC_KEYS = {"use_gitignore", "ignore", "preserve"}
PROJECT_SHARE_KEYS = {"project", "host"}


def now():
    return datetime.now(timezone.utc).isoformat()


def check_keys(mapping, allowed, section):
    if not isinstance(mapping, dict):
        raise ValueError(f"{section} must be a TOML table")
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {section}: {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def identifier(value, label="name"):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", value):
        raise ValueError(f"Invalid {label} {value!r}; use 1-64 letters, numbers, dots, hyphens or underscores")
    return value


def sanitize_id(value, fallback="item"):
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip(".-_")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalnum():
        cleaned = "x" + cleaned
    return cleaned[:64]


def remote_path(value, label="path"):
    if (not isinstance(value, str) or not value.startswith("/") or value.startswith("//")
            or ".." in Path(value).parts or any(ord(c) < 32 for c in value)):
        raise ValueError(f"Expected absolute remote {label} without '..': {value!r}")
    norm = str(Path(value))
    if norm == "/":
        raise ValueError(f"Refusing filesystem root '/' as remote {label}")
    return norm


def rel_path(value, label="relative path"):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid {label}: {value!r}")
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or value in (".", "./"):
        raise ValueError(f"Expected safe relative {label}: {value!r}")
    norm = p.as_posix().strip("/")
    if not norm or norm == ".":
        raise ValueError(f"Invalid {label}: {value!r}")
    return norm


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def run(argv, *, cwd=None, env=None, stdin=None, timeout=30):
    completed = subprocess.run(argv, cwd=cwd, env=env, input=stdin, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{argv[0]} exited {completed.returncode}: {detail[-1500:]}")
    return completed.stdout.strip()


def try_run(argv, *, cwd=None):
    completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    return completed.returncode == 0, completed.stdout.strip()


def load_hosts_config(path):
    if not path.is_file():
        raise ValueError(f"Host config not found: {path}")
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    check_keys(cfg, HOST_TOP_KEYS, "hosts.toml")
    if cfg.get("schema_version", 1) != 1:
        raise ValueError("hosts.toml schema_version must be 1")
    hosts = cfg.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError("hosts.toml must define at least one [hosts.<name>] entry")
    parsed = {}
    for hname, raw in hosts.items():
        identifier(hname, "host")
        check_keys(raw, HOST_ENTRY_KEYS, f"hosts.{hname}")
        if "ssh" not in raw or "root" not in raw:
            raise ValueError(f"hosts.{hname} requires 'ssh' and 'root'")
        device = raw.get("device", "cpu")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
            raise ValueError(f"hosts.{hname}.device must be 'cpu' or 'cuda[:N]'")
        root = remote_path(raw["root"], "root")
        locks_dir = remote_path(raw["locks_dir"], "locks_dir") if "locks_dir" in raw else str(Path(root) / "locks")
        default_slot = ("gpu-" + (device.partition(":")[2] or "0")) if device.startswith("cuda") else ""
        gpu_slot = identifier(raw["gpu_slot"], "gpu_slot") if raw.get("gpu_slot") else default_slot
        host_pp = raw.get("path_prepend", [])
        if not isinstance(host_pp, list) or not all(isinstance(x, str) for x in host_pp):
            raise ValueError(f"hosts.{hname}.path_prepend must be a list of paths")
        host_path_prepend = [remote_path(x, "path_prepend") for x in host_pp]
        envs = {}
        for ename, eraw in raw.get("envs", {}).items():
            identifier(ename, "env")
            check_keys(eraw, HOST_ENV_KEYS, f"hosts.{hname}.envs.{ename}")
            pp = eraw.get("path_prepend", [])
            if not isinstance(pp, list) or not all(isinstance(x, str) for x in pp):
                raise ValueError(f"hosts.{hname}.envs.{ename}.path_prepend must be a list of paths")
            senv = eraw.get("env", {})
            if not isinstance(senv, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in senv.items()):
                raise ValueError(f"hosts.{hname}.envs.{ename}.env must be a string map")
            envs[ename] = {
                "python": remote_path(eraw["python"], "python") if "python" in eraw else "",
                "path_prepend": [remote_path(x, "path_prepend") for x in pp],
                "env": senv,
            }
        default_env = raw.get("default_env", "")
        if default_env:
            identifier(default_env, "default_env")
            if default_env not in envs and default_env not in ("none", "venv"):
                raise ValueError(f"hosts.{hname}.default_env {default_env!r} is not in envs")
        shared = {}
        for sname, sraw in raw.get("shared", {}).items():
            identifier(sname, "shared asset")
            check_keys(sraw, HOST_SHARED_KEYS, f"hosts.{hname}.shared.{sname}")
            if "path" not in sraw:
                raise ValueError(f"hosts.{hname}.shared.{sname} requires 'path'")
            digest = sraw.get("sha256", "")
            if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"hosts.{hname}.shared.{sname}.sha256 must be 64 lowercase hex chars")
            shared[sname] = {
                "name": sname,
                "path": remote_path(sraw["path"], "shared path"),
                "manifest": rel_path(sraw["manifest"], "manifest") if "manifest" in sraw else "",
                "sha256": digest,
                "version": str(sraw.get("version", sname)),
            }
        parsed[hname] = {
            "name": hname,
            "ssh": identifier(raw["ssh"], "ssh"),
            "root": root,
            "locks_dir": locks_dir,
            "device": device,
            "gpu_slot": gpu_slot,
            "default_env": default_env,
            "path_prepend": host_path_prepend,
            "envs": envs,
            "shared": shared,
        }
    default_host = cfg.get("default_host") or next(iter(parsed))
    if default_host not in parsed:
        raise ValueError(f"default_host {default_host!r} not found in hosts")
    return default_host, parsed


def select_host(config_path, host_name=None):
    default_host, hosts = load_hosts_config(config_path)
    chosen = identifier(host_name or default_host, "host")
    if chosen not in hosts:
        raise ValueError(f"Unknown host {chosen!r}; configured: {', '.join(hosts)}")
    return hosts[chosen]


def resolve_repo_context(repo_dir):
    target = repo_dir.resolve()
    if not target.is_dir():
        raise ValueError(f"Directory does not exist: {target}")
    ok, git_top = try_run(["git", "rev-parse", "--show-toplevel"], cwd=target)
    if ok and git_top:
        root = Path(git_top).resolve()
        try:
            rel = target.relative_to(root).as_posix()
        except ValueError:
            rel = ""
        in_git = True
    else:
        root, rel, in_git = target, "", False
    if root == Path("/") or root == Path.home().resolve():
        raise ValueError(f"Refusing to use filesystem root or local home directory as a sync repo: {root}")
    return root, ("" if rel == "." else rel), in_git


def empty_project_config():
    return {
        "present": False, "path": None, "project": "", "host": "", "env": "",
        "workspace": "", "remote_dir": "",
        "sync": {"use_gitignore": True, "ignore": [], "preserve": []},
        "share": {"project": [], "host": []}, "env_vars": {},
    }


def load_project_config(repo_root):
    cfg_path = repo_root / ".remote-run.toml"
    if not cfg_path.is_file():
        return empty_project_config()
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)
    check_keys(raw, PROJECT_TOP_KEYS, ".remote-run.toml")
    if raw.get("schema_version", 1) != 1:
        raise ValueError(".remote-run.toml schema_version must be 1")
    sync = raw.get("sync", {})
    check_keys(sync, PROJECT_SYNC_KEYS, ".remote-run.toml [sync]")
    share = raw.get("share", {})
    check_keys(share, PROJECT_SHARE_KEYS, ".remote-run.toml [share]")
    env_vars = raw.get("env_vars", {})
    if not isinstance(env_vars, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env_vars.items()):
        raise ValueError(".remote-run.toml [env_vars] must be a string map")
    for key, label in (("ignore", "sync.ignore"), ("preserve", "sync.preserve")):
        val = sync.get(key, [])
        if not isinstance(val, list) or not all(isinstance(x, str) and x and not any(ord(c) < 32 for c in x) for x in val):
            raise ValueError(f".remote-run.toml [{label}] must be a list of clean non-empty strings")
    if not isinstance(sync.get("use_gitignore", True), bool):
        raise ValueError(".remote-run.toml [sync].use_gitignore must be boolean")
    p_shared = share.get("project", [])
    if not isinstance(p_shared, list) or not all(isinstance(x, str) for x in p_shared):
        raise ValueError(".remote-run.toml [share].project must be a list of relative directory names")
    h_shared = share.get("host", [])
    if not isinstance(h_shared, list) or not all(isinstance(x, str) for x in h_shared):
        raise ValueError(".remote-run.toml [share].host must be a list of strings")
    return {
        "present": True,
        "path": str(cfg_path),
        "project": identifier(raw["project"], "project") if "project" in raw else "",
        "host": identifier(raw["host"], "host") if "host" in raw else "",
        "env": identifier(raw["env"], "env") if "env" in raw else "",
        "workspace": identifier(raw["workspace"], "workspace") if "workspace" in raw else "",
        "remote_dir": remote_path(raw["remote_dir"], "remote_dir") if "remote_dir" in raw else "",
        "sync": {
            "use_gitignore": sync.get("use_gitignore", True),
            "ignore": list(sync.get("ignore", [])),
            "preserve": [rel_path(x, "sync.preserve") for x in sync.get("preserve", [])],
        },
        "share": {
            "project": [rel_path(x, "share.project") for x in p_shared],
            "host": list(h_shared),
        },
        "env_vars": dict(env_vars),
    }


def resolve_workspace(repo_root, proj_cfg, host, *, rel_subdir="", in_git=False,
                      workspace_override="", remote_dir_override="", ignore_share_project=False):
    repo_root = repo_root.resolve()
    root_commit = ""
    branch = ""
    git_commit = ""
    git_dirty = False
    if in_git:
        ok_rc, rc_out = try_run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo_root)
        if ok_rc and rc_out:
            root_commit = sorted(rc_out.splitlines())[0][:8]
        ok_hd_full, hd_full = try_run(["git", "rev-parse", "HEAD"], cwd=repo_root)
        if ok_hd_full and hd_full:
            git_commit = hd_full.strip()
        ok_st, st_out = try_run(["git", "status", "--porcelain"], cwd=repo_root)
        if ok_st:
            git_dirty = bool(st_out.strip())
        ok_br, br_out = try_run(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo_root)
        if ok_br and br_out:
            branch = br_out
        else:
            ok_hd, hd_out = try_run(["git", "rev-parse", "--short=8", "HEAD"], cwd=repo_root)
            head_tag = hd_out if (ok_hd and hd_out) else "empty"
            branch = f"detached-{head_tag}-{sha256(str(repo_root).encode()).hexdigest()[:6]}"

    if proj_cfg["project"]:
        project_id = proj_cfg["project"]
    elif in_git and root_commit:
        ok_cd, cd_out = try_run(["git", "rev-parse", "--git-common-dir"], cwd=repo_root)
        if ok_cd and cd_out:
            cpath = Path(cd_out)
            cpath = (repo_root / cpath).resolve() if not cpath.is_absolute() else cpath.resolve()
            common_name = cpath.parent.name if cpath.name == ".git" else cpath.name.removesuffix(".git")
        else:
            common_name = repo_root.name
        project_id = sanitize_id(f"{common_name[:55]}-{root_commit}", "project")
    else:
        path_hash = sha256(str(repo_root).encode()).hexdigest()[:6]
        project_id = sanitize_id(f"{repo_root.name[:57]}-{path_hash}", "project")

    ws_chosen = workspace_override or proj_cfg.get("workspace", "")
    if ws_chosen:
        workspace_id = identifier(ws_chosen, "workspace")
    elif in_git and branch:
        if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", branch):
            workspace_id = branch
        else:
            bhash = sha256(branch.encode()).hexdigest()[:6]
            workspace_id = sanitize_id(branch[:57], "branch") + f"-{bhash}"
    else:
        workspace_id = "default"

    explicit_dir = remote_dir_override or proj_cfg["remote_dir"]
    if explicit_dir and proj_cfg["share"]["project"] and not ignore_share_project:
        raise ValueError(
            "Cannot combine remote_dir with [share].project; use managed project workspaces for cross-worktree shared dirs"
        )
    if explicit_dir:
        wdir = remote_path(explicit_dir, "remote_dir")
        shared_dir = ""
        managed = False
    else:
        wdir = str(Path(host["root"]) / "projects" / project_id / "workspaces" / workspace_id)
        shared_dir = str(Path(host["root"]) / "projects" / project_id / "shared")
        managed = True
    return {
        "project_id": project_id, "workspace_id": workspace_id, "managed": managed,
        "workdir": wdir, "shared_dir": shared_dir, "rel_subdir": rel_subdir,
        "in_git": in_git, "branch": branch, "root_commit": root_commit,
        "git_commit": git_commit, "git_branch": branch, "git_dirty": git_dirty,
    }


def parse_shared_spec(spec):
    if "=" in spec:
        alias, name = spec.split("=", 1)
    else:
        alias, name = spec, spec
    return identifier(alias, "shared alias"), identifier(name, "shared asset")


def resolve_shared_bindings(host, specs):
    bindings = {}
    for spec in specs:
        alias, asset_name = parse_shared_spec(spec)
        if asset_name not in host["shared"]:
            avail = ", ".join(host["shared"]) or "<none>"
            raise ValueError(f"Shared asset {asset_name!r} not configured on host {host['name']!r}; available: {avail}")
        entry = dict(host["shared"][asset_name])
        entry["alias"] = alias
        bindings[alias] = entry
    return bindings


def resolve_env(host, env_name):
    host_pp = list(host.get("path_prepend", []))
    if not env_name or (env_name == "none" and "none" not in host["envs"]):
        return {"name": env_name or "none", "python": "", "path_prepend": host_pp, "env": {}, "auto_venv": False}
    if env_name == "venv" and "venv" not in host["envs"]:
        return {"name": "venv", "python": "", "path_prepend": host_pp, "env": {}, "auto_venv": True}
    identifier(env_name, "env")
    if env_name not in host["envs"]:
        avail = ", ".join(list(host["envs"]) + ["venv", "none"])
        raise ValueError(f"Environment {env_name!r} not configured on host {host['name']!r}; available: {avail}")
    cfg = host["envs"][env_name]
    return {
        "name": env_name,
        "python": cfg["python"],
        "path_prepend": host_pp + list(cfg["path_prepend"]),
        "env": dict(cfg["env"]),
        "auto_venv": False,
    }


def verify_shared_asset(binding):
    base = Path(binding["path"])
    if not base.exists():
        raise ValueError(f"Shared asset {binding['name']!r} missing on remote host: {base}")
    target = (base / binding["manifest"]) if binding["manifest"] else base
    if binding["sha256"]:
        if not target.is_file():
            raise ValueError(f"Shared asset {binding['name']!r} expected file for sha256 check: {target}")
        with target.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != binding["sha256"]:
            raise ValueError(
                f"Shared asset {binding['name']!r} sha256 mismatch at {target}: "
                f"got {digest}, expected {binding['sha256']}"
            )
    return str(base)


def expand_placeholders(text, mapping):
    def repl(match):
        expr = match.group(1)
        if expr in mapping:
            return mapping[expr]
        if expr.startswith("shared."):
            raise ValueError(f"Unknown shared placeholder {{{expr}}} (did you bind the shared asset?)")
        return match.group(0)
    return re.sub(r"\{([a-zA-Z0-9_.-]+)\}", repl, text)


def fold_cr(text):
    out = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if "\r" in line:
            parts = [p for p in line.split("\r") if p != ""]
            out.append(parts[-1] if parts else "")
        else:
            out.append(line)
    if text.endswith("\n") and out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def read_log_slice(log_path, *, offset=None, tail_bytes=8192, raw_log=False):
    if tail_bytes < 0:
        raise ValueError("--tail-bytes must be non-negative")
    if offset is not None and offset < 0:
        raise ValueError("--offset must be non-negative")
    if not log_path.is_file():
        return {"text": "", "total_bytes": 0, "offset": 0, "next_offset": 0, "truncated": False}
    size = log_path.stat().st_size
    with log_path.open("rb") as f:
        if offset is not None:
            start = max(0, min(offset, size))
            f.seek(start)
            chunk = f.read(tail_bytes) if tail_bytes > 0 else f.read()
            truncated = (start + len(chunk)) < size
        elif tail_bytes > 0 and size > tail_bytes:
            start = size - tail_bytes
            f.seek(start)
            chunk = f.read(tail_bytes)
            truncated = True
        else:
            start = 0
            chunk = f.read()
            truncated = False
    decoded = chunk.decode("utf-8", errors="replace")
    if not raw_log:
        decoded = fold_cr(decoded)
    return {"text": decoded, "total_bytes": size, "offset": start, "next_offset": start + len(chunk), "truncated": truncated}


def session_name(root, run_id):
    return "rr-" + sha256((str(root) + "/" + run_id).encode()).hexdigest()[:20]


def session_key_for(workdir, name):
    return sha256(f"{workdir}::{name}".encode()).hexdigest()[:20]


def pid_alive(pgid):
    if not isinstance(pgid, int) or pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def single_pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def validate_remote_root(root):
    r = root.resolve()
    home = Path.home().resolve()
    if r == Path("/") or r == home or home.is_relative_to(r):
        raise ValueError(f"Refusing unsafe remote root {r} (cannot be '/', remote $HOME, or an ancestor of $HOME)")
    return r


def validate_remote_workdir(root, wdir):
    r = validate_remote_root(root)
    w = wdir.resolve()
    home = Path.home().resolve()
    if w == Path("/") or w == home or home.is_relative_to(w):
        raise ValueError(f"Refusing unsafe remote workdir {w} (cannot be '/', remote $HOME, or an ancestor of $HOME)")
    if w == r or r.is_relative_to(w):
        raise ValueError(f"Refusing remote workdir {w}: cannot equal or contain remote root {r}")
    for reserved in ("runs", "locks", "sessions"):
        if w.is_relative_to(r / reserved):
            raise ValueError(f"Refusing remote workdir {w}: inside reserved control dir {r / reserved}")
    proj_base = r / "projects"
    if w.is_relative_to(proj_base):
        rel_parts = w.relative_to(proj_base).parts
        if len(rel_parts) >= 2 and rel_parts[1] == "shared":
            raise ValueError(f"Refusing remote workdir {w}: inside project shared dir")
        if len(rel_parts) < 3 or rel_parts[1] != "workspaces":
            raise ValueError(
                f"Refusing remote workdir {w}: paths under projects/ must be inside projects/<project>/workspaces/<workspace>"
            )
    return w


def ensure_root_owned(root, *, init_root=False, dry_run=False):
    validate_remote_root(root)
    marker = root / ROOT_MARKER
    if marker.is_file():
        return
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"Configured remote root is not a directory: {root}")
        if any(root.iterdir()) and not init_root:
            raise ValueError(
                f"Refusing to use non-empty unmanaged remote root {root}; "
                f"verify it belongs to remote-run and run `remote-run doctor --init-root` once"
            )
    if dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(marker, {"schema_version": 1, "created_at": now()})


def state_of(job, root, *, persist=True):
    state_path = job / "state.json"
    if not state_path.is_file():
        raise ValueError(f"No such run: {job.name}")
    state = read_json(state_path)
    if not isinstance(state, dict):
        raise ValueError(f"Corrupt run metadata for {job.name}")
    if state["status"] not in TERMINAL:
        has_tmux = subprocess.run(
            ["tmux", "has-session", "-t", "=" + session_name(root, job.name)], capture_output=True
        ).returncode == 0
        if not has_tmux:
            state = read_json(state_path)
            if not isinstance(state, dict):
                raise ValueError(f"Corrupt run metadata for {job.name}")
            if state["status"] not in TERMINAL:
                state["status"] = "cancelled" if (job / "cancel.request").exists() else "interrupted"
                state.setdefault("finished_at", now())
                if persist:
                    write_json(state_path, state)
    req = read_json(job / "request.json")
    if not isinstance(req, dict):
        raise ValueError(f"Corrupt run metadata for {job.name}")
    elapsed = None
    if state.get("started_at"):
        t0 = datetime.fromisoformat(state["started_at"])
        t1 = datetime.fromisoformat(state["finished_at"]) if state.get("finished_at") else datetime.now(timezone.utc)
        elapsed = round(max(0.0, (t1 - t0).total_seconds()), 3)
    return {
        **state,
        "host": req["host"],
        "device": req.get("physical_device", req["device"]),
        "env": req.get("env_name", ""),
        "project_id": req.get("project_id", ""),
        "workspace_id": req.get("workspace_id", ""),
        "workdir": req["workdir"],
        "shared_dir": req.get("shared_dir", ""),
        "session": req.get("session", ""),
        "command": req.get("command", []),
        "shell": req.get("shell", ""),
        "elapsed_sec": elapsed,
    }


def ssh_call(host, action, data, timeout=60):
    source = Path(__file__).read_text()
    payload = {"action": action, "host": host, "script": source, **data}
    boot = "import json,sys;p=json.load(sys.stdin);exec(compile(p['script'],'remote_run.py','exec'),{'PAYLOAD':p})"
    cmd = shlex.join(["python3", "-B", "-c", boot])
    out = run([*SSH, host["ssh"], cmd], stdin=json.dumps(payload), timeout=timeout)
    return json.loads(out)


def active_runs_for_workdir(root, workdir, *, persist=True):
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return []
    active = []
    for child in runs_dir.iterdir():
        if not child.is_dir() or not (child / "state.json").is_file() or not (child / "request.json").is_file():
            continue
        try:
            st = state_of(child, root, persist=persist)
        except (ValueError, KeyError, OSError):
            continue
        if st["status"] not in TERMINAL and st.get("workdir") == workdir:
            active.append(st["run_id"])
    return active


def prepare_remote_workspace(root, ws_info, share_project, *, check_run_id="", adopt=False, dry_run=False, for_sync=True):
    ensure_root_owned(root, dry_run=dry_run)
    wdir = Path(remote_path(ws_info["workdir"], "workdir"))
    validate_remote_workdir(root, wdir)
    run_exists = bool(check_run_id and (root / "runs" / identifier(check_run_id, "run_id")).exists())
    if run_exists:
        return {
            "workdir": str(wdir),
            "shared_mounted": list(share_project),
            "run_exists": True,
            "workspace_exists": wdir.is_dir(),
            "active_runs_warning": [],
        }

    managed = bool(ws_info.get("managed", True))
    expected = {
        "project_id": ws_info["project_id"],
        "root_commit": ws_info.get("root_commit", ""),
        "managed": managed,
        "workspace_id": ws_info["workspace_id"] if managed else "",
    }
    marker = wdir / WORKSPACE_MARKER
    if for_sync:
        if marker.is_file():
            found = read_json(marker)
            if not isinstance(found, dict):
                raise ValueError(f"Corrupt workspace marker at {marker}")
            mismatch = (
                found.get("project_id") != expected["project_id"]
                or found.get("root_commit", "") != expected["root_commit"]
                or (managed and found.get("workspace_id") != expected["workspace_id"])
            )
            if mismatch:
                f_rc = f"@{found.get('root_commit', '')[:8]}" if found.get("root_commit") else ""
                e_rc = f"@{expected['root_commit'][:8]}" if expected["root_commit"] else ""
                raise ValueError(
                    f"Refusing sync: {wdir} belongs to a different project/workspace "
                    f"({found.get('project_id')}/{found.get('workspace_id')}{f_rc}), expected "
                    f"({expected['project_id']}/{expected['workspace_id']}{e_rc})"
                )
        elif wdir.exists():
            if not wdir.is_dir():
                raise ValueError(f"Remote workdir exists and is not a directory: {wdir}")
            if any(wdir.iterdir()) and not adopt:
                raise ValueError(
                    f"Refusing sync into non-empty unmanaged remote directory {wdir}; "
                    f"choose a new directory, pass --no-sync, or claim it once with `remote-run sync --adopt`"
                )
            if not dry_run:
                write_json(marker, {"schema_version": 1, "created_at": now(), **expected})
        else:
            if not dry_run:
                wdir.mkdir(parents=True, exist_ok=True)
                write_json(marker, {"schema_version": 1, "created_at": now(), **expected})
    else:
        if wdir.exists() and not wdir.is_dir():
            raise ValueError(f"Remote workdir exists and is not a directory: {wdir}")
        if managed and not dry_run and (not wdir.exists() or (not marker.is_file() and not any(wdir.iterdir()))):
            wdir.mkdir(parents=True, exist_ok=True)
            write_json(marker, {"schema_version": 1, "created_at": now(), **expected})

    if share_project:
        if not ws_info.get("shared_dir"):
            raise ValueError("Cannot mount project shared directories without a managed shared_dir")
        sdir = Path(remote_path(ws_info["shared_dir"], "shared_dir"))
        if not dry_run:
            sdir.mkdir(parents=True, exist_ok=True)
        for name in share_project:
            rel = rel_path(name, "share.project")
            link_path = wdir / rel
            target_path = sdir / rel
            if dry_run:
                if not link_path.is_symlink() and link_path.exists():
                    if link_path.is_dir():
                        if any(link_path.iterdir()):
                            raise ValueError(
                                f"Cannot mount shared dir {rel!r}: non-empty real directory exists at {link_path}"
                            )
                    else:
                        raise ValueError(f"Cannot mount shared dir {rel!r}: file exists at {link_path}")
                continue
            target_path.mkdir(parents=True, exist_ok=True)
            link_path.parent.mkdir(parents=True, exist_ok=True)
            if link_path.is_symlink():
                if link_path.resolve() != target_path.resolve():
                    link_path.unlink()
                    link_path.symlink_to(target_path)
            elif link_path.exists():
                if link_path.is_dir():
                    if any(link_path.iterdir()):
                        raise ValueError(
                            f"Cannot mount shared dir {rel!r}: non-empty real directory exists at {link_path}"
                        )
                    link_path.rmdir()
                    link_path.symlink_to(target_path)
                else:
                    raise ValueError(f"Cannot mount shared dir {rel!r}: file exists at {link_path}")
            else:
                link_path.symlink_to(target_path)
    return {
        "workdir": str(wdir),
        "shared_mounted": list(share_project),
        "run_exists": False,
        "workspace_exists": wdir.is_dir(),
        "active_runs_warning": active_runs_for_workdir(root, str(wdir), persist=not dry_run),
    }


def normalize_request(root, request):
    req = dict(request)
    if req.pop("fallback_root", False):
        candidate = Path(remote_path(req["workdir"], "workdir"))
        if not candidate.is_dir():
            req["workdir"] = str(root)
    return req


def remote_submit(root, job, host, request, worker_src):
    req_to_save = normalize_request(root, request)
    if job.exists():
        if read_json(job / "request.json") != req_to_save:
            raise ValueError(f"Run ID {job.name} already belongs to a different request")
        return dict(state_of(job, root), reused=True)
    for binding in req_to_save["shared"].values():
        verify_shared_asset(binding)
    if req_to_save["python"] and not Path(req_to_save["python"]).is_file():
        raise ValueError(f"Configured Python does not exist on host: {req_to_save['python']}")
    workdir = Path(remote_path(req_to_save["workdir"], "workdir"))
    workdir.mkdir(parents=True, exist_ok=True)
    job.mkdir(parents=True)
    (job / "artifacts").mkdir()
    write_json(job / "request.json", req_to_save)
    write_json(job / "state.json", {
        "run_id": job.name, "status": "queued", "waiting_for_lock": False, "submitted_at": now(),
    })
    try:
        (job / "worker.py").write_text(worker_src)
        args = [sys.executable, "-B", str(job / "worker.py"), "_worker", str(root), str(job)]
        cmd = shlex.join(args) + " > " + shlex.quote(str(job / "run.log")) + " 2>&1"
        run(["tmux", "new-session", "-d", "-s", session_name(root, job.name), "-c", str(workdir), "--", "sh", "-c", cmd])
    except Exception as error:
        write_json(job / "state.json", {"run_id": job.name, "status": "failed", "error": str(error), "finished_at": now()})
        raise
    return dict(state_of(job, root), reused=False)


def kill_pgid(pgid):
    if not isinstance(pgid, int) or pgid <= 0:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline and pid_alive(pgid):
        time.sleep(0.15)
    if pid_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def remote_cancel(root, job):
    state = state_of(job, root)
    if state["status"] in TERMINAL:
        return state
    write_json(job / "cancel.request", {"requested_at": now()})
    pgid = state.get("child_pgid")
    wpid = state.get("worker_pid")
    if isinstance(pgid, int) and pgid > 0:
        kill_pgid(pgid)
    elif isinstance(wpid, int) and wpid > 0:
        try:
            os.kill(wpid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    subprocess.run(["tmux", "kill-session", "-t", "=" + session_name(root, job.name)], capture_output=True, check=False)
    if not pgid:
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            cur = read_json(job / "state.json")
            cpg = cur.get("child_pgid")
            if isinstance(cpg, int) and cpg > 0:
                kill_pgid(cpg)
                break
            if cur.get("status") in TERMINAL:
                break
            time.sleep(0.05)
    latest = read_json(job / "state.json")
    latest["status"] = "cancelled"
    latest.setdefault("finished_at", now())
    write_json(job / "state.json", latest)
    return state_of(job, root)


def eligible_pull_files(job, workdir, *, shared_dir="", all_artifacts=False, max_file_mb=10.0, extra_paths=(), paths_only=False):
    if not math.isfinite(max_file_mb) or max_file_mb < 0:
        raise ValueError("--max-file-mb must be a finite non-negative number")
    if paths_only and not extra_paths:
        raise ValueError("--paths-only requires at least one --path")
    job_files = []
    skipped = []
    if not paths_only:
        job_files = [n for n in ("request.json", "state.json", "run.log") if (job / n).is_file()]
        art_root = job / "artifacts"
        limit_bytes = max(0, int(max_file_mb * 1024 * 1024))
        if art_root.is_dir():
            art_resolved = art_root.resolve()
            for item in sorted(art_root.rglob("*")):
                if item.is_symlink() or not item.is_file():
                    continue
                if not item.resolve().is_relative_to(art_resolved):
                    continue
                rel = item.relative_to(job).as_posix()
                size = item.stat().st_size
                if not all_artifacts and item.suffix.lower() in CHECKPOINT_SUFFIXES:
                    skipped.append({"path": rel, "size_bytes": size, "reason": "checkpoint"})
                    continue
                if not all_artifacts and limit_bytes and size > limit_bytes:
                    skipped.append({"path": rel, "size_bytes": size, "reason": f"exceeds_{max_file_mb}mb"})
                    continue
                job_files.append(rel)
    workdir_files = []
    wroot = Path(workdir).resolve()
    sroot = Path(shared_dir).resolve() if shared_dir else None
    for raw_rel in extra_paths:
        rel = rel_path(raw_rel, "pull path")
        raw_target = wroot / rel
        if not raw_target.exists():
            raise ValueError(f"Requested workspace path missing: {raw_rel!r}")
        resolved = raw_target.resolve()
        in_work = resolved.is_relative_to(wroot)
        in_shared = bool(sroot and sroot.is_dir() and resolved.is_relative_to(sroot))
        if not (in_work or in_shared):
            raise ValueError(f"Path escapes workspace: {raw_rel!r}")
        if raw_target.is_symlink() and not resolved.is_dir():
            raise ValueError(f"Requested workspace path cannot be a file symlink: {raw_rel!r}")
        if resolved.is_dir():
            for item in sorted(resolved.rglob("*")):
                if item.is_file() and not item.is_symlink():
                    ir = item.resolve()
                    if ir.is_relative_to(wroot) or (sroot and ir.is_relative_to(sroot)):
                        workdir_files.append(f"{rel}/{item.relative_to(resolved).as_posix()}")
        elif resolved.is_file():
            workdir_files.append(rel)
    return job_files, workdir_files, skipped


def parse_gpus(raw_out):
    parsed = []
    for line in raw_out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            mem_match = re.search(r"([0-9]+)", parts[2])
            parsed.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_mib": int(mem_match.group(1)) if mem_match else None,
                "raw": line,
            })
        else:
            parsed.append({"raw": line})
    return parsed


def dir_size_bytes(path):
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def prune_runs(root, host, days, apply_del):
    cutoff = time.time() - max(0.0, days) * 86400.0
    runs_dir = root / "runs"
    pruned = []
    skipped_unmanaged = []
    reclaimed_bytes = 0
    if runs_dir.is_dir():
        for child in sorted(runs_dir.iterdir()):
            if child.is_symlink() or not child.is_dir():
                skipped_unmanaged.append({"path": str(child), "reason": "symlink_or_non_dir"})
                continue
            req_p, st_p = child / "request.json", child / "state.json"
            if not (req_p.is_file() or st_p.is_file()):
                skipped_unmanaged.append({"path": str(child), "reason": "missing_metadata"})
                continue
            has_tmux = subprocess.run(
                ["tmux", "has-session", "-t", "=" + session_name(root, child.name)], capture_output=True
            ).returncode == 0
            if has_tmux:
                continue
            raw_st = {}
            raw_req = {}
            try:
                if st_p.is_file():
                    loaded_st = read_json(st_p)
                    if isinstance(loaded_st, dict):
                        raw_st = loaded_st
            except (ValueError, KeyError, OSError):
                pass
            try:
                if req_p.is_file():
                    loaded_req = read_json(req_p)
                    if isinstance(loaded_req, dict):
                        raw_req = loaded_req
            except (ValueError, KeyError, OSError):
                pass
            if "run_id" not in raw_st and "host" not in raw_req:
                skipped_unmanaged.append({"path": str(child), "reason": "unrecognized_metadata"})
                continue
            if single_pid_alive(raw_st.get("worker_pid")) or pid_alive(raw_st.get("child_pgid")):
                continue
            mtime = child.stat().st_mtime
            if mtime > cutoff:
                continue
            try:
                st = state_of(child, root, persist=False)
                status_label = st["status"]
                if status_label not in TERMINAL:
                    continue
            except (ValueError, KeyError, OSError):
                status_label = "corrupt"
            sz = dir_size_bytes(child)
            pruned.append({"run_id": child.name, "status": status_label, "bytes": sz})
            reclaimed_bytes += sz
            if apply_del:
                shutil.rmtree(child)
    workspaces = []
    proj_dir = root / "projects"
    if proj_dir.is_dir():
        for ws in sorted(proj_dir.glob("*/workspaces/*")):
            if ws.is_dir() and not ws.is_symlink():
                workspaces.append({"path": str(ws), "bytes": dir_size_bytes(ws)})
    return {
        "host": host["name"], "days": days, "applied": apply_del,
        "runs_count": len(pruned), "reclaimed_bytes": reclaimed_bytes,
        "runs": pruned, "skipped_unmanaged": skipped_unmanaged, "workspaces": workspaces,
    }


def control(payload):
    host, action = payload["host"], payload["action"]
    root = Path(remote_path(host["root"], "root"))
    locks_dir = Path(remote_path(host["locks_dir"], "locks_dir"))

    if action == "doctor":
        missing = [t for t in ("git", "tmux", "rsync") if not shutil.which(t)]
        if missing:
            raise ValueError(f"Compute host missing required tool(s): {', '.join(missing)}")
        ensure_root_owned(root, init_root=bool(payload.get("init_root", False)))
        locks_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(locks_dir, 0o700)
        info = {
            "status": "ready",
            "host": host["name"],
            "env": payload.get("env_name", ""),
            "device": host["device"],
            "root": str(root),
        }
        if host["device"].startswith("cuda"):
            if not shutil.which("nvidia-smi"):
                raise ValueError("Host configured for CUDA but 'nvidia-smi' not found")
            gpus = run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"])
            info["gpus"] = parse_gpus(gpus)
        check_imports = list(payload.get("check_imports", []))
        check_cuda = bool(payload.get("check_cuda", False))
        py = payload.get("python", "")
        if (check_imports or check_cuda) and not py:
            raise ValueError(
                f"Cannot run --check-import/--check-cuda with env {info['env']!r} (no host Python configured); "
                "pass --env <name> with a configured python path"
            )
        if py:
            if not Path(py).is_file():
                raise ValueError(f"Configured python not found on host: {py}")
            info["python"] = py
            info["python_version"] = run([py, "--version"])
            if check_imports or check_cuda:
                for mod in check_imports:
                    identifier(mod.split(".")[0], "module")
                probe_code = (
                    "import importlib, json, sys\n"
                    "mods = json.loads(sys.argv[1])\n"
                    "check_cuda = sys.argv[2] == '1'\n"
                    "imports = {}\n"
                    "for m in mods:\n"
                    "    try:\n"
                    "        importlib.import_module(m)\n"
                    "        imports[m] = True\n"
                    "    except Exception:\n"
                    "        imports[m] = False\n"
                    "res = {'imports': imports}\n"
                    "if check_cuda:\n"
                    "    try:\n"
                    "        import torch\n"
                    "        res['cuda_probe'] = {'cuda': bool(torch.cuda.is_available()), 'count': int(torch.cuda.device_count())}\n"
                    "    except Exception as exc:\n"
                    "        res['cuda_probe'] = {'cuda': False, 'count': 0, 'error': str(exc)}\n"
                    "print(json.dumps(res))\n"
                )
                probe = json.loads(run([py, "-B", "-c", probe_code, json.dumps(check_imports), "1" if check_cuda else "0"]))
                info["checked_imports"] = check_imports
                info["imports"] = probe["imports"]
                info["missing_imports"] = [m for m in check_imports if not probe["imports"].get(m)]
                if check_cuda:
                    info["cuda_probe"] = probe["cuda_probe"]
                if info["missing_imports"]:
                    info["status"] = "missing_imports"
                    info["_exec_failed"] = True
        verified_shared = {}
        for alias, binding in payload.get("shared", {}).items():
            verified_shared[alias] = verify_shared_asset(binding)
        if verified_shared:
            info["shared"] = verified_shared
        return info

    if action == "prepare_workspace" and payload.get("dry_run"):
        return prepare_remote_workspace(
            root, payload["workspace"], payload.get("share_project", []),
            check_run_id=payload.get("run_id", ""),
            adopt=bool(payload.get("adopt", False)),
            dry_run=True,
            for_sync=bool(payload.get("for_sync", True)),
        )
    if action == "prune" and not payload.get("apply", False):
        ensure_root_owned(root, dry_run=True)
        return prune_runs(root, host, float(payload.get("days", 14.0)), False)

    ensure_root_owned(root)
    locks_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(locks_dir, 0o700)

    if action == "session":
        sdir = root / "sessions"
        sub = payload["subaction"]
        if sub == "list":
            items = []
            if sdir.is_dir():
                for p in sorted(sdir.glob("*.json")):
                    try:
                        data = read_json(p)
                        if not isinstance(data, dict):
                            raise ValueError("Corrupt session metadata")
                    except (ValueError, KeyError, OSError) as exc:
                        items.append({"session_key": p.stem, "status": "corrupt", "error": str(exc)})
                        continue
                    if payload.get("workdir") and data.get("workdir") != payload["workdir"]:
                        continue
                    items.append({"session_key": p.stem, **data})
            return {"sessions": items}
        if sub == "reset":
            key = session_key_for(payload["workdir"], payload["name"])
            sdir.mkdir(parents=True, exist_ok=True)
            os.chmod(sdir, 0o700)
            with (sdir / f"{key}.lock").open("a") as slock:
                fcntl.flock(slock, fcntl.LOCK_EX)
                target = sdir / f"{key}.json"
                existed = target.is_file()
                target.unlink(missing_ok=True)
            return {"reset": True, "name": payload["name"], "existed": existed, "workdir": payload["workdir"]}
        raise ValueError(f"Unknown session subaction: {sub}")

    if action == "status":
        wait_sec = float(payload.get("wait_sec", 0.0))
        if not math.isfinite(wait_sec) or wait_sec < 0:
            raise ValueError("wait_sec must be a finite non-negative number")
        job = root / "runs" / identifier(payload["run_id"], "run_id")
        deadline = time.monotonic() + wait_sec
        state = state_of(job, root)
        while state["status"] not in TERMINAL and time.monotonic() < deadline:
            time.sleep(min(0.5, max(0.05, deadline - time.monotonic())))
            state = state_of(job, root)
        if payload.get("include_log", True):
            state["log"] = read_log_slice(
                job / "run.log",
                offset=payload.get("offset"),
                tail_bytes=payload.get("tail_bytes", 8192),
                raw_log=payload.get("raw_log", False),
            )
        return state

    with (root / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if action == "prepare_workspace":
            return prepare_remote_workspace(
                root, payload["workspace"], payload.get("share_project", []),
                check_run_id=payload.get("run_id", ""),
                adopt=bool(payload.get("adopt", False)),
                for_sync=bool(payload.get("for_sync", True)),
            )
        if action == "list":
            runs_dir = root / "runs"
            items = []
            proj_filter = payload.get("project_id") or ""
            limit = payload.get("limit", 20)
            if runs_dir.is_dir():
                candidates = [c for c in runs_dir.iterdir() if c.is_dir()]
                candidates.sort(key=lambda c: c.stat().st_mtime, reverse=True)
                for child in candidates:
                    if not (child / "state.json").is_file() or not (child / "request.json").is_file():
                        continue
                    try:
                        st = state_of(child, root)
                    except (ValueError, KeyError, OSError) as exc:
                        if not proj_filter:
                            items.append({"run_id": child.name, "status": "corrupt", "error": str(exc)})
                        continue
                    if proj_filter and st.get("project_id") != proj_filter:
                        continue
                    items.append(st)
                    if len(items) >= limit:
                        break
            items.sort(key=lambda r: r.get("submitted_at", ""), reverse=True)
            return {"runs": items[:limit]}
        if action == "prune":
            return prune_runs(root, host, float(payload.get("days", 14.0)), bool(payload.get("apply", False)))
        job = root / "runs" / identifier(payload["run_id"], "run_id")
        if action == "submit":
            return remote_submit(root, job, host, payload["request"], payload["script"])
        if action == "cancel":
            return remote_cancel(root, job)
        state = state_of(job, root)
        if action == "pull":
            if state["status"] not in TERMINAL:
                raise ValueError(f"Run {job.name} is still {state['status']}; wait or cancel before pulling")
            job_files, workdir_files, skipped = eligible_pull_files(
                job, state["workdir"], shared_dir=state.get("shared_dir", ""),
                all_artifacts=payload.get("all_artifacts", False),
                max_file_mb=payload.get("max_file_mb", 10.0),
                extra_paths=payload.get("paths", ()),
                paths_only=bool(payload.get("paths_only", False)),
            )
            return {
                "status": state["status"], "job_files": job_files, "workdir_files": workdir_files,
                "workdir": state["workdir"], "skipped_artifacts": skipped,
            }
        raise ValueError(f"Unknown action: {action}")


def session_file(root, session_key):
    return root / "sessions" / f"{session_key}.json"


def load_session(root, session_key, fallback_cwd):
    if not session_key:
        return fallback_cwd, {}
    sdir = root / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    os.chmod(sdir, 0o700)
    with (sdir / f"{session_key}.lock").open("a") as slock:
        fcntl.flock(slock, fcntl.LOCK_SH)
        path = session_file(root, session_key)
        if not path.is_file():
            return fallback_cwd, {}
        data = read_json(path)
    cwd = data.get("cwd", fallback_cwd)
    if not Path(cwd).is_dir():
        cwd = fallback_cwd
    return cwd, dict(data.get("env", {}))


def update_session(root, session_key, session_name_user, workdir, cwd_out, env_out, baseline_env, prev_session_env):
    if not session_key or not cwd_out.is_file() or not env_out.is_file():
        return None
    new_cwd = cwd_out.read_text(errors="replace").rstrip("\r\n")
    if not new_cwd or not Path(new_cwd).is_dir():
        new_cwd = None
    env_bytes = env_out.read_bytes()
    after_env = {}
    if env_bytes:
        for entry in env_bytes.split(b"\x00"):
            if b"=" not in entry:
                continue
            k, v = entry.split(b"=", 1)
            ks = k.decode("utf-8", errors="replace")
            if ks in SKIP_SESSION_ENV or ks.startswith(SKIP_SESSION_PREFIXES):
                continue
            after_env[ks] = v.decode("utf-8", errors="replace")
    added_or_changed = {k: v for k, v in after_env.items() if baseline_env.get(k) != v} if after_env else {}
    removed = {k for k in prev_session_env if k not in after_env} if after_env else set()
    sdir = root / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    os.chmod(sdir, 0o700)
    with (sdir / f"{session_key}.lock").open("a") as slock:
        fcntl.flock(slock, fcntl.LOCK_EX)
        path = session_file(root, session_key)
        cur = read_json(path) if path.is_file() else {}
        merged = dict(cur.get("env", {}))
        for k in removed:
            merged.pop(k, None)
        merged.update(added_or_changed)
        write_json(path, {
            "name": session_name_user,
            "workdir": workdir,
            "cwd": new_cwd or cur.get("cwd", workdir),
            "env": merged,
            "updated_at": now(),
        })
        os.chmod(path, 0o600)
    return new_cwd


def worker(root, job):
    request = read_json(job / "request.json")
    state_path = job / "state.json"
    state = read_json(state_path)
    state["worker_pid"] = os.getpid()
    write_json(state_path, state)
    locks_dir = Path(remote_path(request["locks_dir"], "locks_dir"))
    locks_dir.mkdir(parents=True, exist_ok=True)
    use_lock = request["device"].startswith("cuda") and request.get("use_gpu", True) and request.get("gpu_slot")
    lease = (locks_dir / f"{request['gpu_slot']}.lock").open("a") if use_lock else nullcontext()
    proc = None

    def forward_sig(signum, _frame):
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            raise SystemExit(143)

    signal.signal(signal.SIGTERM, forward_sig)
    signal.signal(signal.SIGHUP, forward_sig)

    try:
        if use_lock:
            state.update(waiting_for_lock=True)
            write_json(state_path, state)
        with lease as handle:
            if handle is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            if (job / "cancel.request").exists():
                state.update(waiting_for_lock=False, status="cancelled")
                write_json(state_path, state)
                return
            state.update(waiting_for_lock=False)
            write_json(state_path, state)

            shared_paths = {alias: verify_shared_asset(binding) for alias, binding in request["shared"].items()}

            workdir = Path(request["workdir"])
            default_cwd = str(workdir / request["rel_subdir"]) if (
                request.get("rel_subdir") and (workdir / request["rel_subdir"]).is_dir()
            ) else str(workdir)
            session_key = request.get("session_key", "")
            cwd, session_env = load_session(root, session_key, default_cwd)

            env = dict(os.environ)
            env.update(request.get("env_static", {}))
            prepend = list(request.get("path_prepend", []))
            effective_python = request["python"]
            if request.get("auto_venv"):
                vbin = workdir / ".venv" / "bin"
                if vbin.is_dir():
                    prepend.insert(0, str(vbin))
                    env["VIRTUAL_ENV"] = str(workdir / ".venv")
                    vpy = vbin / "python"
                    if vpy.is_file():
                        effective_python = str(vpy)
            elif effective_python:
                prepend.insert(0, str(Path(effective_python).parent))
            if prepend:
                env["PATH"] = os.pathsep.join(prepend + ([env["PATH"]] if env.get("PATH") else []))

            user_cvd = "CUDA_VISIBLE_DEVICES" in request.get("env_static", {}) or "CUDA_VISIBLE_DEVICES" in request.get("env_vars", {})
            exposed_device = request["device"]
            if request["device"] == "cpu" or not request.get("use_gpu", True):
                if not user_cvd:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                exposed_device = "cpu"
            elif request["device"].startswith("cuda") and not user_cvd:
                idx = request["device"].partition(":")[2] or "0"
                env["CUDA_VISIBLE_DEVICES"] = idx
                exposed_device = "cuda:0"

            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            env.update(session_env)

            git_commit = request.get("git_commit", "")
            git_branch = request.get("git_branch", "")
            git_dirty = "1" if request.get("git_dirty") else ("0" if git_commit else "")

            placeholders = {
                "run_id": job.name,
                "host": request["host"],
                "device": exposed_device,
                "artifacts": str(job / "artifacts"),
                "code": str(workdir),
                "python": effective_python or shutil.which("python3", path=env.get("PATH", "")) or sys.executable,
                "git_commit": git_commit,
                "git_branch": git_branch,
            }
            for alias, binding in request["shared"].items():
                placeholders[f"shared.{alias}.path"] = shared_paths[alias]
                placeholders[f"shared.{alias}.version"] = binding["version"]
                placeholders[f"shared.{alias}.sha256"] = binding["sha256"]

            env.update({
                "REMOTE_RUN_ID": job.name,
                "REMOTE_RUN_HOST": request["host"],
                "REMOTE_RUN_DEVICE": exposed_device,
                "REMOTE_RUN_ARTIFACTS": str(job / "artifacts"),
                "REMOTE_RUN_CODE": str(workdir),
                "REMOTE_RUN_PYTHON": placeholders["python"],
                "REMOTE_RUN_GIT_COMMIT": git_commit,
                "REMOTE_RUN_GIT_BRANCH": git_branch,
                "REMOTE_RUN_GIT_DIRTY": git_dirty,
            })
            for alias, binding in request["shared"].items():
                prefix = "REMOTE_RUN_SHARED_" + re.sub(r"[^A-Z0-9]+", "_", alias.upper()).strip("_")
                env[f"{prefix}_PATH"] = shared_paths[alias]
                env[f"{prefix}_NAME"] = binding["name"]
                env[f"{prefix}_VERSION"] = binding["version"]
                env[f"{prefix}_SHA256"] = binding["sha256"]

            for k, v in request.get("env_vars", {}).items():
                env[k] = expand_placeholders(v, placeholders)

            if (job / "cancel.request").exists():
                state.update(status="cancelled")
                write_json(state_path, state)
                return

            cwd_out = job / ".session_cwd"
            env_out = job / ".session_env"
            if session_key:
                wrapper = (
                    "__rr_finish() { local rc=$?; pwd -P > \"$__RR_CWD_FILE\" 2>/dev/null || true; "
                    "(/usr/bin/env -0 || env -0) > \"$__RR_ENV_FILE\" 2>/dev/null || true; exit \"$rc\"; }; "
                    "trap __rr_finish EXIT; "
                )
                env["__RR_CWD_FILE"] = str(cwd_out)
                env["__RR_ENV_FILE"] = str(env_out)
            else:
                wrapper = ""

            if request.get("shell"):
                bash_argv = ["bash", "-c", wrapper + request["shell"]]
            else:
                cmd_args = list(request["command"])
                if effective_python and not session_env.get("PATH") and cmd_args and cmd_args[0] in ("python", "python3"):
                    candidate_bin = Path(effective_python).parent / cmd_args[0]
                    if not candidate_bin.is_file():
                        cmd_args[0] = effective_python
                bash_argv = ["bash", "-c", wrapper + '"$@"', "remote-run", *cmd_args]

            baseline_env = dict(env)
            proc = subprocess.Popen(bash_argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
            state.update(status="running", started_at=now(), child_pgid=proc.pid, cwd=cwd)
            write_json(state_path, state)
            rc = proc.wait()
            if (job / "cancel.request").exists():
                state.update(exit_code=rc, status="cancelled")
            else:
                state.update(exit_code=rc, status="completed" if rc == 0 else "failed")
                if rc == 0 and session_key:
                    new_cwd = update_session(
                        root, session_key, request.get("session", ""), str(workdir),
                        cwd_out, env_out, baseline_env, session_env,
                    )
                    if new_cwd:
                        state["cwd"] = new_cwd
    except SystemExit:
        state.update(waiting_for_lock=False, status="cancelled" if (job / "cancel.request").exists() else "interrupted")
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        for tmp in (job / ".session_cwd", job / ".session_env"):
            tmp.unlink(missing_ok=True)
        state["finished_at"] = now()
        write_json(state_path, state)


def build_rsync_filters(proj_cfg, ws_info):
    filters = []
    for marker in (ROOT_MARKER, WORKSPACE_MARKER):
        filters.extend(["--filter", f"P /{marker}", "--filter", f"- /{marker}"])
    protected = list(BUILTIN_PRESERVE) + proj_cfg["sync"]["preserve"] + proj_cfg["share"]["project"]
    seen_p = set()
    for p in protected:
        norm = p.strip("/")
        if not norm or norm in seen_p:
            continue
        seen_p.add(norm)
        filters.extend(["--filter", f"P /{norm}", "--filter", f"P /{norm}/**",
                        "--filter", f"- /{norm}", "--filter", f"- /{norm}/**"])
    filters.extend(["--filter", "- /.git/", "--filter", "- /.git"])
    for pat in BUILTIN_IGNORE:
        filters.extend(["--filter", f"- {pat}"])
    if proj_cfg["sync"]["use_gitignore"]:
        filters.extend(["--filter", ":- .gitignore"])
    for pat in proj_cfg["sync"]["ignore"]:
        filters.extend(["--filter", f"- {pat}"])
    return filters


def parse_rsync_itemize(stdout):
    would_update = []
    would_delete = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        code, path = parts
        if code.startswith("*deleting"):
            would_delete.append(path)
        elif code[0] in ("<", ">", "c", "h"):
            would_update.append(path)
    return would_update, would_delete


def sync_workspace(repo_root, proj_cfg, host, ws_info, *, run_id="", delete=False, dry_run=False, adopt=False):
    if adopt and delete:
        raise ValueError("Cannot combine --adopt and --delete; run --adopt first, then inspect with --dry-run --delete")
    lock_dir = Path(tempfile.gettempdir()) / "remote-run-sync-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / (sha256(f"{host['ssh']}:{ws_info['workdir']}".encode()).hexdigest()[:20] + ".lock")
    with lock_file.open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        prep = ssh_call(host, "prepare_workspace", {
            "workspace": ws_info,
            "share_project": proj_cfg["share"]["project"],
            "run_id": run_id,
            "adopt": adopt,
            "dry_run": dry_run,
        })
        if prep.get("run_exists"):
            return {"host": host["name"], "workdir": ws_info["workdir"], "skipped_for_existing_run": True,
                    "project_id": ws_info["project_id"], "workspace_id": ws_info["workspace_id"]}
        filters = build_rsync_filters(proj_cfg, ws_info)
        if dry_run and not prep.get("workspace_exists", True):
            with tempfile.TemporaryDirectory() as empty_dst:
                dry_cmd = [
                    "rsync", "-rlptz",
                    *(["--delete"] if delete else []),
                    "-ni", "--itemize-changes", "--protect-args",
                    *filters, f"{repo_root.resolve()}/", f"{empty_dst}/",
                ]
                out = run(dry_cmd, timeout=120)
        else:
            dest = f"{host['ssh']}:{ws_info['workdir']}/"
            rsync_cmd = ["rsync", "-rlptz", "--protect-args"]
            if delete:
                rsync_cmd.append("--delete")
            if dry_run:
                rsync_cmd.extend(["-ni", "--itemize-changes"])
            rsync_cmd.extend(["-e", shlex.join(SSH), *filters, f"{repo_root.resolve()}/", dest])
            out = run(rsync_cmd, timeout=120)
        res = {
            "host": host["name"], "workdir": ws_info["workdir"],
            "project_id": ws_info["project_id"], "workspace_id": ws_info["workspace_id"],
            "delete": delete, "dry_run": dry_run,
            "active_runs_warning": prep.get("active_runs_warning", []),
        }
        if dry_run:
            upd, dele = parse_rsync_itemize(out)
            res["would_update"] = upd
            res["would_delete"] = dele
        return res


def wait_for_status(host, run_id, wait_sec, *, offset=None, tail_bytes=8192, raw_log=False):
    if not math.isfinite(wait_sec) or wait_sec < 0:
        raise ValueError("--wait must be a finite non-negative number of seconds")
    res = ssh_call(
        host,
        "status",
        {
            "run_id": run_id,
            "include_log": True,
            "offset": offset,
            "tail_bytes": tail_bytes,
            "raw_log": raw_log,
            "wait_sec": wait_sec,
        },
        timeout=max(60, int(wait_sec) + 30),
    )
    res["detached"] = res["status"] not in TERMINAL
    if res["detached"]:
        res["next_action"] = (
            f"Check progress with `remote-run status {run_id} --wait 30` or cancel with `remote-run cancel {run_id}`"
        )
    return res


def handle_exec(args, host, repo_root, rel_subdir, proj_cfg):
    if args.shell and args.command:
        raise ValueError("Use either --shell CMD or positional command after '--', not both")
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not args.shell and not cmd:
        raise ValueError("Provide a command after '--' or pass --shell CMD")
    if not math.isfinite(args.wait) or args.wait < 0:
        raise ValueError("--wait must be a finite non-negative number of seconds")

    if getattr(args, "_no_project_mode", False):
        explicit_dir = args.remote_dir or proj_cfg["remote_dir"]
        if explicit_dir:
            ws_info = {
                "project_id": "", "workspace_id": "", "managed": False,
                "workdir": remote_path(explicit_dir, "remote_dir"), "shared_dir": "", "rel_subdir": "",
                "in_git": False, "branch": "", "root_commit": "",
                "git_commit": "", "git_branch": "", "git_dirty": False,
            }
        elif args.session:
            wdir = str(Path(host["root"]) / "projects" / "_adhoc" / "workspaces" / "default")
            ws_info = {
                "project_id": "_adhoc", "workspace_id": "default", "managed": True,
                "workdir": wdir, "shared_dir": "", "rel_subdir": "",
                "in_git": False, "branch": "", "root_commit": "",
                "git_commit": "", "git_branch": "", "git_dirty": False,
            }
        else:
            ws_info = {
                "project_id": "", "workspace_id": "", "managed": True,
                "workdir": host["root"], "shared_dir": "", "rel_subdir": "",
                "in_git": False, "branch": "", "root_commit": "",
                "git_commit": "", "git_branch": "", "git_dirty": False,
            }
    else:
        ws_info = resolve_workspace(
            repo_root, proj_cfg, host,
            rel_subdir=rel_subdir, in_git=args._in_git,
            workspace_override=args.workspace or "", remote_dir_override=args.remote_dir or "",
            ignore_share_project=bool(args.no_sync),
        )
    env_name = args.env or proj_cfg["env"] or host["default_env"]
    env_cfg = resolve_env(host, env_name)
    shared_specs = list(proj_cfg["share"]["host"]) + list(args.shared)
    shared_bindings = resolve_shared_bindings(host, shared_specs)
    env_vars = dict(proj_cfg["env_vars"])
    for item in args.env_var:
        if "=" not in item:
            raise ValueError(f"--env-var must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        env_vars[k] = v

    run_id = args.run_id or datetime.now(timezone.utc).strftime("rr-%Y%m%dT%H%M%S-") + uuid4().hex[:8]
    fallback_root = False
    active_warning = []
    if not args.no_sync:
        sync_res = sync_workspace(repo_root, proj_cfg, host, ws_info, run_id=run_id)
        active_warning = sync_res.get("active_runs_warning", [])
    elif args.session or args.remote_dir or not ws_info["managed"]:
        prep = ssh_call(host, "prepare_workspace", {
            "workspace": ws_info,
            "share_project": proj_cfg["share"]["project"] if ws_info["managed"] else [],
            "run_id": run_id,
            "for_sync": False,
        })
        active_warning = prep.get("active_runs_warning", [])
    else:
        fallback_root = True

    session_name_user = identifier(args.session, "session") if args.session else ""
    session_key = session_key_for(ws_info["workdir"], session_name_user) if session_name_user else ""

    request = {
        "host": host["name"],
        "device": host["device"],
        "physical_device": host["device"],
        "gpu_slot": host["gpu_slot"],
        "locks_dir": host["locks_dir"],
        "use_gpu": not args.no_gpu,
        "env_name": env_cfg["name"],
        "python": env_cfg["python"],
        "auto_venv": env_cfg["auto_venv"],
        "path_prepend": env_cfg["path_prepend"],
        "env_static": env_cfg["env"],
        "env_vars": env_vars,
        "shared": shared_bindings,
        "workdir": ws_info["workdir"],
        "shared_dir": ws_info["shared_dir"],
        "rel_subdir": rel_subdir,
        "fallback_root": fallback_root,
        "project_id": ws_info["project_id"],
        "workspace_id": ws_info["workspace_id"],
        "git_commit": ws_info.get("git_commit", ""),
        "git_branch": ws_info.get("git_branch", ""),
        "git_dirty": bool(ws_info.get("git_dirty", False)),
        "session": session_name_user,
        "session_key": session_key,
        "command": [] if args.shell else cmd,
        "shell": args.shell or "",
    }
    receipt = ssh_call(host, "submit", {"run_id": run_id, "request": request})
    if args.wait == 0:
        receipt["detached"] = receipt["status"] not in TERMINAL
        if receipt["detached"]:
            receipt["next_action"] = (
                f"Check progress with `remote-run status {run_id} --wait 30` or cancel with `remote-run cancel {run_id}`"
            )
        if active_warning:
            receipt["active_runs_warning"] = active_warning
        if receipt.get("status") in ("failed", "interrupted", "cancelled"):
            receipt["_exec_failed"] = True
        return receipt
    out = wait_for_status(host, run_id, args.wait, tail_bytes=args.tail_bytes, raw_log=args.raw_log)
    out["reused"] = receipt.get("reused", False)
    if active_warning:
        out["active_runs_warning"] = active_warning
    if out.get("status") in ("failed", "interrupted", "cancelled"):
        out["_exec_failed"] = True
    return out


def handle_pull(args, host):
    dest = Path(os.path.abspath(args.destination))
    if dest == Path("/") or dest == Path.home():
        raise ValueError(f"Refusing dangerous pull destination: {dest}")
    cwd_res = Path.cwd().resolve()
    dest_check = dest.resolve() if dest.exists() else dest
    if cwd_res == dest_check or cwd_res.is_relative_to(dest_check):
        raise ValueError(f"Refusing to pull into or above current working directory: {dest}")
    if (dest.exists() or dest.is_symlink()) and not args.force:
        raise ValueError(f"Destination already exists (use a new directory or --force): {dest}")
    if not math.isfinite(args.max_file_mb) or args.max_file_mb < 0:
        raise ValueError("--max-file-mb must be a finite non-negative number")
    paths_only = bool(getattr(args, "paths_only", False))
    if paths_only and not args.path:
        raise ValueError("--paths-only requires at least one --path")
    meta = ssh_call(host, "pull", {
        "run_id": args.run_id, "all_artifacts": args.all_artifacts,
        "max_file_mb": args.max_file_mb, "paths": args.path, "paths_only": paths_only,
    })
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".remote-run-pull-", dir=dest.parent) as stage:
        contents = Path(stage) / "contents"
        contents.mkdir()
        if meta["job_files"]:
            run([
                "rsync", "-aR", "--protect-args", "--files-from=-", "--from0", "-e", shlex.join(SSH),
                f"{host['ssh']}:{host['root']}/runs/{args.run_id}/", str(contents),
            ], stdin="\0".join(meta["job_files"]), timeout=300)
        if meta["workdir_files"]:
            wdest = contents / "workspace"
            wdest.mkdir(parents=True, exist_ok=True)
            run([
                "rsync", "-aRk", "--protect-args", "--files-from=-", "--from0", "-e", shlex.join(SSH),
                f"{host['ssh']}:{meta['workdir']}/", str(wdest),
            ], stdin="\0".join(meta["workdir_files"]), timeout=300)
        if dest.is_symlink():
            dest.unlink()
        if dest.exists():
            run(["rsync", "-a", f"{contents}/", f"{dest}/"], timeout=120)
        else:
            contents.rename(dest)
    return {"run_id": args.run_id, "destination": str(dest), **meta}


def infer_cwd_project_host():
    try:
        repo_root, _, _ = resolve_repo_context(Path.cwd())
        return load_project_config(repo_root)
    except (ValueError, RuntimeError):
        return None


def cli():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                        help=f"Host config path (default: $REMOTE_RUN_CONFIG or {DEFAULT_CONFIG})")
    common.add_argument("--host", default=argparse.SUPPRESS,
                        help="Host name from hosts.toml (default: project .remote-run.toml host or default_host)")
    common.add_argument("--pretty", action="store_true", default=argparse.SUPPRESS,
                        help="Pretty-print JSON output")

    parser = argparse.ArgumentParser(prog="remote-run", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("hosts", parents=[common], help="List configured hosts, environments, and host-level shared assets")

    cfg_p = sub.add_parser("config", parents=[common], help="Validate configs and print effective workspace/sync settings")
    cfg_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Local project path (default: cwd, auto-normalized to git root)")
    cfg_p.add_argument("--remote-dir", default="", help="Explicit absolute remote workspace path override")
    cfg_p.add_argument("--workspace", default="", help="Explicit remote workspace ID override")

    doc_p = sub.add_parser("doctor", parents=[common], help="Verify SSH, git/tmux/rsync, GPU, Python env, and shared assets")
    doc_p.add_argument("--repo", type=Path, default=None, help="Local project path to infer host, env, and [share].host from .remote-run.toml")
    doc_p.add_argument("--env", default="", help="Environment name from host config (or 'venv' / 'none')")
    doc_p.add_argument("--shared", action="append", default=[], help="Host shared asset name to verify (repeatable)")
    doc_p.add_argument("--check-import", action="append", default=[], help="Python module to test importing (repeatable)")
    doc_p.add_argument("--check-cuda", action="store_true", help="Probe torch.cuda.is_available() in the selected Python env")
    doc_p.add_argument("--init-root", action="store_true", help="Claim an existing non-empty remote root directory as owned by remote-run")

    sync_p = sub.add_parser("sync", parents=[common], help="One-way additive sync from local repository to remote workspace")
    sync_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Local project path (default: cwd, auto-normalized to git root)")
    sync_p.add_argument("--remote-dir", default="", help="Explicit absolute remote workspace path override")
    sync_p.add_argument("--workspace", default="", help="Explicit remote workspace ID override")
    sync_p.add_argument("--delete", action="store_true", help="Explicitly delete remote files absent locally (respects preserve protections)")
    sync_p.add_argument("--dry-run", action="store_true", help="Preview files that would be updated or deleted without modifying remote")
    sync_p.add_argument("--adopt", action="store_true", help="Claim a pre-existing non-empty remote workspace directory once")

    ex_p = sub.add_parser("exec", parents=[common], help="Sync (optional) and run command on remote host")
    ex_p.add_argument("--repo", type=Path, default=None, help="Local project path (default: cwd, auto-normalized to git root)")
    ex_p.add_argument("--remote-dir", default="", help="Explicit absolute remote workspace path override")
    ex_p.add_argument("--workspace", default="", help="Explicit remote workspace ID override")
    ex_p.add_argument("--no-sync", action="store_true", help="Skip rsync upload before executing")
    ex_p.add_argument("--no-project", action="store_true", help="Do not infer .remote-run.toml from --repo/cwd")
    ex_p.add_argument("--session", default="", help="Opt-in persistent shell session name (persists cwd and env diffs on rc=0)")
    ex_p.add_argument("--shell", default="", help="Run a compound bash command string (do not combine with positional command)")
    ex_p.add_argument("--wait", type=float, default=15.0, help="Seconds to wait before detaching to background (0=immediate, default: 15)")
    ex_p.add_argument("--no-gpu", action="store_true", help="Skip GPU slot lock (useful for quick inspection like nvidia-smi)")
    ex_p.add_argument("--env", default="", help="Host environment name, 'venv', or 'none'")
    ex_p.add_argument("--shared", action="append", default=[], help="Bind host shared asset as NAME or ALIAS=NAME (repeatable)")
    ex_p.add_argument("--env-var", action="append", default=[], help="Extra KEY=VALUE env var (supports {artifacts}/{shared.*}, repeatable)")
    ex_p.add_argument("--run-id", type=identifier, help="Explicit run ID for idempotent retries")
    ex_p.add_argument("--tail-bytes", type=int, default=8192, help="Max trailing log bytes to return (default: 8192)")
    ex_p.add_argument("--raw-log", action="store_true", help="Return raw log bytes without collapsing tqdm carriage returns")
    ex_p.add_argument("--print-log", action="store_true", help="Print decoded log text to stdout and metadata JSON to stderr")
    ex_p.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute after '--' (e.g. -- python train.py)")

    st_p = sub.add_parser("status", parents=[common], help="Inspect a run status/log, or list recent runs")
    st_p.add_argument("run_id", nargs="?", type=identifier, help="Run ID to inspect (omit to list recent runs)")
    st_p.add_argument("--wait", type=float, default=0.0, help="Seconds to wait for terminal status (default: 0)")
    st_p.add_argument("--offset", type=int, default=None, help="Byte offset to read log from")
    st_p.add_argument("--tail-bytes", type=int, default=8192, help="Max log bytes to return (default: 8192)")
    st_p.add_argument("--raw-log", action="store_true", help="Disable carriage-return folding")
    st_p.add_argument("--print-log", action="store_true", help="Print decoded log text to stdout and metadata JSON to stderr")
    st_p.add_argument("--limit", type=int, default=20, help="Max recent runs to list when run_id is omitted (default: 20)")
    st_p.add_argument("--project", default="", help="Filter listed runs by project_id")
    st_p.add_argument("--repo", type=Path, default=None, help="Filter listed runs to the project of this local repo")

    ca_p = sub.add_parser("cancel", parents=[common], help="Cancel a queued or running job cleanly without leaking GPU VRAM")
    ca_p.add_argument("run_id", type=identifier, help="Run ID to cancel")

    pl_p = sub.add_parser("pull", parents=[common], help="Pull logs, metadata, and eligible artifacts to a local directory")
    pl_p.add_argument("run_id", type=identifier, help="Terminal run ID to pull from")
    pl_p.add_argument("destination", type=Path, help="Local destination directory (must not exist unless --force)")
    pl_p.add_argument("--all-artifacts", action="store_true", help="Include checkpoint files and artifacts exceeding --max-file-mb")
    pl_p.add_argument("--max-file-mb", type=float, default=10.0, help="Max artifact file size in MB when --all-artifacts is unset (default: 10)")
    pl_p.add_argument("--path", action="append", default=[], help="Relative path inside remote workspace to pull into DEST/workspace/ (repeatable)")
    pl_p.add_argument("--paths-only", action="store_true", help="Pull only --path workspace files without re-fetching job logs/artifacts")
    pl_p.add_argument("--force", action="store_true", help="Merge pulled files into an existing destination directory without deleting other files")

    se_p = sub.add_parser("session", parents=[common], help="List or reset opt-in persistent shell sessions")
    se_sub = se_p.add_subparsers(dest="session_action", required=True)
    sl_p = se_sub.add_parser("list", parents=[common], help="List persistent sessions on the host")
    sl_p.add_argument("--repo", type=Path, default=None, help="Filter sessions to this local project workspace")
    sl_p.add_argument("--remote-dir", default="", help="Filter sessions to explicit remote directory")
    sl_p.add_argument("--workspace", default="", help="Workspace ID override when filtering with --repo")
    sr_p = se_sub.add_parser("reset", parents=[common], help="Reset (delete) a named persistent session for the workspace")
    sr_p.add_argument("name", type=identifier, help="Session name to reset")
    sr_p.add_argument("--repo", type=Path, default=None, help="Local project path (default: cwd unless --remote-dir is set)")
    sr_p.add_argument("--remote-dir", default="", help="Explicit remote directory override")
    sr_p.add_argument("--workspace", default="", help="Workspace ID override")

    pr_p = sub.add_parser("prune", parents=[common], help="Inspect or remove old terminal run directories on the remote host")
    pr_p.add_argument("--days", type=float, default=14.0, help="Remove terminal runs older than this many days (default: 14)")
    pr_p.add_argument("--apply", action="store_true", help="Actually delete old terminal run directories (default is dry-run)")

    args = parser.parse_args()
    config_path = getattr(args, "config", Path(os.environ.get("REMOTE_RUN_CONFIG") or DEFAULT_CONFIG))
    cli_host = getattr(args, "host", None)
    pretty = bool(getattr(args, "pretty", False))

    if args.action == "hosts":
        default_host, hosts = load_hosts_config(config_path)
        return {"default_host": default_host, "config_path": str(config_path), "hosts": hosts}, pretty, False

    if args.action in ("config", "sync", "exec"):
        args._no_project_mode = False
        if args.action == "exec" and args.no_project:
            if not args.no_sync:
                raise ValueError("--no-project can only be used with --no-sync")
            repo_root, rel_subdir, in_git = Path.cwd(), "", False
            proj_cfg = empty_project_config()
            args._no_project_mode = True
        elif (
            args.action == "exec"
            and args.no_sync
            and args.repo is None
            and (args.remote_dir or Path.cwd().resolve() in (Path("/"), Path.home().resolve()))
        ):
            repo_root, rel_subdir, in_git = Path.cwd(), "", False
            cwd_proj = infer_cwd_project_host()
            proj_cfg = empty_project_config()
            if cwd_proj and cwd_proj["host"]:
                proj_cfg["host"] = cwd_proj["host"]
            args._no_project_mode = True
        else:
            repo_arg = args.repo if args.repo is not None else Path.cwd()
            repo_root, rel_subdir, in_git = resolve_repo_context(repo_arg)
            proj_cfg = load_project_config(repo_root)
        args._in_git = in_git
        host = select_host(config_path, cli_host or proj_cfg["host"] or None)
        if args.action == "config":
            ws_info = resolve_workspace(
                repo_root, proj_cfg, host,
                rel_subdir=rel_subdir, in_git=in_git,
                workspace_override=args.workspace or "", remote_dir_override=args.remote_dir or "",
            )
            env_name = proj_cfg["env"] or host["default_env"]
            resolve_env(host, env_name)
            resolve_shared_bindings(host, proj_cfg["share"]["host"])
            return {
                "config_path": str(config_path),
                "repo_root": str(repo_root),
                "host": host["name"],
                "device": host["device"],
                "env": env_name,
                "workspace": ws_info,
                "project_config": proj_cfg,
                "effective_ignore": [".git/", *BUILTIN_IGNORE, *(["<from .gitignore>"] if proj_cfg["sync"]["use_gitignore"] else []), *proj_cfg["sync"]["ignore"]],
                "effective_preserve": sorted(set(BUILTIN_PRESERVE + proj_cfg["sync"]["preserve"] + proj_cfg["share"]["project"])),
            }, pretty, False
        if args.action == "sync":
            ws_info = resolve_workspace(
                repo_root, proj_cfg, host,
                rel_subdir=rel_subdir, in_git=in_git,
                workspace_override=args.workspace or "", remote_dir_override=args.remote_dir or "",
            )
            return sync_workspace(
                repo_root, proj_cfg, host, ws_info,
                delete=args.delete, dry_run=args.dry_run, adopt=args.adopt,
            ), pretty, False
        return handle_exec(args, host, repo_root, rel_subdir, proj_cfg), pretty, bool(args.print_log)

    cwd_proj = infer_cwd_project_host()
    host = select_host(config_path, cli_host or (cwd_proj["host"] if cwd_proj else None) or None)

    if args.action == "doctor":
        if args.repo is not None:
            rr_root, _, _ = resolve_repo_context(args.repo)
            doc_proj = load_project_config(rr_root)
            host = select_host(config_path, cli_host or doc_proj["host"] or None)
            use_proj_defaults = True
        else:
            doc_proj = cwd_proj
            use_proj_defaults = bool(doc_proj and (not cli_host or not doc_proj["host"] or cli_host == doc_proj["host"]))
        env_name = args.env or (doc_proj["env"] if use_proj_defaults else "") or host["default_env"]
        env_cfg = resolve_env(host, env_name)
        shared_specs = (list(doc_proj["share"]["host"]) if use_proj_defaults else []) + list(args.shared)
        shared = resolve_shared_bindings(host, shared_specs)
        return ssh_call(host, "doctor", {
            "env_name": env_cfg["name"],
            "python": env_cfg["python"],
            "check_imports": args.check_import,
            "check_cuda": args.check_cuda,
            "shared": shared,
            "init_root": args.init_root,
        }), pretty, False
    if args.action == "status":
        if not args.run_id:
            proj_filter = args.project
            if args.repo is not None:
                rr_root, r_sub, r_git = resolve_repo_context(args.repo)
                pc = load_project_config(rr_root)
                host = select_host(config_path, cli_host or pc["host"] or None)
                if not proj_filter:
                    proj_filter = resolve_workspace(rr_root, pc, host, rel_subdir=r_sub, in_git=r_git)["project_id"]
            return ssh_call(host, "list", {"limit": args.limit, "project_id": proj_filter}), pretty, False
        return wait_for_status(host, args.run_id, args.wait, offset=args.offset,
                               tail_bytes=args.tail_bytes, raw_log=args.raw_log), pretty, bool(args.print_log)
    if args.action == "cancel":
        return ssh_call(host, "cancel", {"run_id": args.run_id}), pretty, False
    if args.action == "pull":
        return handle_pull(args, host), pretty, False
    if args.action == "prune":
        if not math.isfinite(args.days) or args.days < 0:
            raise ValueError("--days must be a non-negative finite number")
        return ssh_call(host, "prune", {"days": args.days, "apply": args.apply}), pretty, False
    if args.action == "session":
        wdir = ""
        if args.remote_dir:
            wdir = remote_path(args.remote_dir, "remote_dir")
            if args.repo is not None:
                rr_root, _, _ = resolve_repo_context(args.repo)
                pc = load_project_config(rr_root)
                host = select_host(config_path, cli_host or pc["host"] or None)
        elif args.session_action == "reset" or args.repo is not None:
            repo_arg = args.repo if args.repo is not None else Path.cwd()
            rr_root, r_sub, r_git = resolve_repo_context(repo_arg)
            pc = load_project_config(rr_root)
            host = select_host(config_path, cli_host or pc["host"] or None)
            wdir = resolve_workspace(
                rr_root, pc, host, rel_subdir=r_sub, in_git=r_git,
                workspace_override=args.workspace or "",
            )["workdir"]
        payload = {"subaction": args.session_action, "workdir": wdir}
        if args.session_action == "reset":
            payload["name"] = args.name
        return ssh_call(host, "session", payload), pretty, False
    raise ValueError(f"Unsupported action: {args.action}")


if "PAYLOAD" in globals():
    try:
        print(json.dumps(control(PAYLOAD), ensure_ascii=False))
    except (ValueError, KeyError, OSError, RuntimeError, OverflowError, subprocess.TimeoutExpired) as error:
        print(f"remote-run: {error}", file=sys.stderr)
        raise SystemExit(1)
elif __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["_worker"]:
            worker(Path(sys.argv[2]), Path(sys.argv[3]))
        else:
            out, pretty, print_log = cli()
            exec_failed = isinstance(out, dict) and bool(out.pop("_exec_failed", False))
            if print_log and isinstance(out, dict) and "log" in out:
                log_obj = out.pop("log")
                if log_obj.get("text"):
                    print(log_obj["text"])
                meta = {**out, "log_bytes": log_obj.get("total_bytes"), "truncated": log_obj.get("truncated")}
                print(json.dumps(meta, ensure_ascii=False, indent=2 if pretty else None), file=sys.stderr)
            else:
                print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
            if exec_failed:
                raise SystemExit(2)
    except (ValueError, KeyError, OSError, RuntimeError, OverflowError, subprocess.TimeoutExpired, tomllib.TOMLDecodeError) as error:
        print(f"remote-run: {error}", file=sys.stderr)
        raise SystemExit(1)
