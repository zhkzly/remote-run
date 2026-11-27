#!/usr/bin/env python3
"""Standalone zero-dependency remote worker and shared protocol primitives for remote-run v2."""
from __future__ import annotations

import base64
from contextlib import ExitStack, contextmanager, suppress
from datetime import datetime, timezone
import errno
import fcntl
from fnmatch import fnmatchcase
import functools
import grp
import hashlib
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

PROTOCOL_VERSION = 2
SCHEMA_VERSION = 2
CONTROL_REPLY = "__REMOTE_RUN_REPLY__:"
DEFAULT_REMOTE_ROOT = "~/.local/share/remote-run-v2"

ROOT_MARKER = ".remote-run-root.json"
WORKSPACE_MARKER = ".remote-run-workspace.json"
SYNCED_MANIFEST = ".remote-run-synced.json"
RUN_MARKER = ".remote-run-job.json"
CONTROL_ROOT_FILES = {ROOT_MARKER, WORKSPACE_MARKER, SYNCED_MANIFEST, RUN_MARKER}

TERMINAL = {"completed", "failed", "interrupted", "cancelled"}
POSIX_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
RESERVED_ENV_KEYS = {"PATH", "BASH_ENV"}
SKIP_SSH_BASELINE_ENV = {
    "_", "PWD", "OLDPWD", "SHLVL", "BASH_EXECUTION_STRING", "PPID",
    "PATH", "BASH_ENV",
}
SKIP_SESSION_ENV = SKIP_SSH_BASELINE_ENV | {"CUDA_VISIBLE_DEVICES"}
SKIP_SESSION_PREFIXES = ("TMUX", "REMOTE_RUN_", "__RR_")


# === Shared Pure Validators & Pathname Matching ===

def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identifier(value: object, label: str = "name") -> str:
    if not isinstance(value, str) or not IDENT_RE.fullmatch(value):
        raise ValueError(
            f"Invalid {label} {value!r}; use 1-64 letters, numbers, dots, hyphens or underscores"
        )
    return value


def posix_env_name(value: object, label: str = "environment variable") -> str:
    if not isinstance(value, str) or not POSIX_ENV_RE.fullmatch(value):
        raise ValueError(f"Invalid {label} name {value!r}; must match [A-Za-z_][A-Za-z0-9_]*")
    return value


def check_env_map(
    mapping: object,
    label: str,
    *,
    reject_reserved: bool = True,
    gpu_managed: bool = False,
) -> dict[str, str]:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a mapping of environment variable names to strings")
    out: dict[str, str] = {}
    for k, v in mapping.items():
        name = posix_env_name(k, label)
        if not isinstance(v, str) or "\0" in v:
            raise ValueError(f"{label}.{name} must be a string without NUL")
        if reject_reserved:
            if name in RESERVED_ENV_KEYS or name.startswith("REMOTE_RUN_"):
                raise ValueError(
                    f"{label} cannot set reserved variable {name!r} "
                    f"(use 'path' for PATH; REMOTE_RUN_* and BASH_ENV are reserved)"
                )
            if gpu_managed and name == "CUDA_VISIBLE_DEVICES":
                raise ValueError(
                    f"{label} cannot set CUDA_VISIBLE_DEVICES on a GPU-managed host; use --gpus / --no-gpu"
                )
        out[name] = v
    return out


def remote_path(value: object, label: str = "path") -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError(f"Expected remote {label} starting with '/' or '~/': {value!r}")
    if ":" in value and label in ("path", "path entry"):
        raise ValueError(f"Remote {label} cannot contain ':': {value!r}")
    if value.startswith("~/"):
        sub = value[2:]
        p = PurePosixPath(sub)
        if not sub or p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Expected safe remote {label} under '~/': {value!r}")
        norm = p.as_posix().strip("/")
        if not norm or norm == ".":
            raise ValueError(f"Refusing bare home directory '~/' as remote {label}")
        return f"~/{norm}"
    if value.startswith("~"):
        raise ValueError(
            f"Use '~/subdir' (not bare {value!r}) or an absolute path like '/opt/bin' for remote {label}"
        )
    if not value.startswith("/") or value.startswith("//") or ".." in PurePosixPath(value).parts:
        raise ValueError(f"Expected absolute remote {label} (or '~/subdir') without '..': {value!r}")
    norm = str(PurePosixPath(value))
    if norm == "/":
        raise ValueError(f"Refusing filesystem root '/' as remote {label}")
    return norm


def expand_remote_path(value: str, label: str = "path") -> str:
    norm = remote_path(value, label)
    if norm.startswith("~/"):
        return str(Path.home() / norm[2:])
    return norm


def join_remote_path(base: str, *parts: str) -> str:
    clean_base = remote_path(base, "path").rstrip("/")
    tail = "/".join(p.strip("/") for p in parts if p.strip("/"))
    return f"{clean_base}/{tail}" if tail else clean_base


def rel_path(value: object, label: str = "relative path") -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid {label}: {value!r}")
    if "\\" in value:
        raise ValueError(f"Expected POSIX relative {label} without backslashes: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or value in (".", "./"):
        raise ValueError(f"Expected safe relative {label}: {value!r}")
    norm = p.as_posix().strip("/")
    if not norm or norm == ".":
        raise ValueError(f"Invalid {label}: {value!r}")
    return norm


def validate_path_entry(value: object, label: str, *, allow_relative: bool = False) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value) or ":" in value:
        raise ValueError(f"Invalid {label} {value!r}; must be non-empty without ':' or control chars")
    if value.startswith("/") or value.startswith("~"):
        return remote_path(value, label)
    if allow_relative:
        return rel_path(value, label)
    raise ValueError(f"Expected absolute or '~/...' {label}: {value!r}")


def validate_pattern(value: object, label: str = "pattern") -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid {label}: {value!r}")
    if value.startswith("/") or value.startswith("!") or "\\" in value:
        raise ValueError(
            f"Invalid {label} {value!r}; use root-relative POSIX patterns without leading '/', '!' or '\\'"
        )
    dir_only = value.endswith("/")
    body = value.rstrip("/")
    if not body:
        raise ValueError(f"Invalid {label}: {value!r}")
    parts = body.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid {label} {value!r}; empty segments, '.' and '..' are not allowed")
    for part in parts:
        if "**" in part and part != "**":
            raise ValueError(f"Invalid {label} {value!r}; '**' must be a complete path segment")
    return body + ("/" if dir_only else "")


def _match_segments(path_parts: tuple[str, ...], pat_parts: tuple[str, ...]) -> bool:
    memo: dict[tuple[int, int], bool] = {}

    def dp(i: int, j: int) -> bool:
        key = (i, j)
        if key in memo:
            return memo[key]
        if j == len(pat_parts):
            res = i == len(path_parts)
        elif pat_parts[j] == "**":
            res = dp(i, j + 1) or (i < len(path_parts) and dp(i + 1, j))
        elif i < len(path_parts) and fnmatchcase(path_parts[i], pat_parts[j]):
            res = dp(i + 1, j + 1)
        else:
            res = False
        memo[key] = res
        return res

    return dp(0, 0)


def match_pattern(rel: str, pattern: str) -> bool:
    """Match root-relative POSIX path against validated v2 glob pattern."""
    dir_only = pattern.endswith("/")
    pat_parts = tuple(pattern.rstrip("/").split("/"))
    path_parts = tuple(rel.strip("/").split("/"))
    if not path_parts or path_parts == ("",):
        return False
    if dir_only:
        # Matches the directory itself or any path beneath a matching directory prefix.
        for end in range(1, len(path_parts) + 1):
            if _match_segments(path_parts[:end], pat_parts):
                return True
        return False
    return _match_segments(path_parts, pat_parts)


def validate_gpu_indices(values: object, label: str = "gpus") -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list of non-negative integers")
    out: list[int] = []
    seen: set[int] = set()
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label} must contain non-negative integers, got {item!r}")
        if item in seen:
            raise ValueError(f"{label} contains duplicate GPU index: {item}")
        seen.add(item)
        out.append(item)
    return out


# === Logging & UTF-8-Safe Slicing ===

def fold_cr(text: str) -> str:
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


def _utf8_expected_len(lead_byte: int) -> int:
    if lead_byte < 0x80:
        return 1
    if 0xC2 <= lead_byte <= 0xDF:
        return 2
    if 0xE0 <= lead_byte <= 0xEF:
        return 3
    if 0xF0 <= lead_byte <= 0xF4:
        return 4
    return 0


def _trailing_incomplete_utf8_len(data: bytes) -> int:
    """Return count (1..3) of trailing bytes that form a valid incomplete UTF-8 sequence."""
    n = len(data)
    if n == 0:
        return 0
    max_lookback = min(3, n)
    for back in range(1, max_lookback + 1):
        b = data[n - back]
        if b < 0x80:
            return 0
        if 0x80 <= b <= 0xBF:
            continue
        expected = _utf8_expected_len(b)
        if expected > back:
            # Verify all following bytes are valid continuation bytes and prefix is legal.
            tail = data[n - back:]
            if all(0x80 <= c <= 0xBF for c in tail[1:]):
                if expected == 3 and len(tail) >= 2:
                    if b == 0xE0 and tail[1] < 0xA0:
                        return 0
                    if b == 0xED and tail[1] >= 0xA0:
                        return 0
                if expected == 4 and len(tail) >= 2:
                    if b == 0xF0 and tail[1] < 0x90:
                        return 0
                    if b == 0xF4 and tail[1] >= 0x90:
                        return 0
                return back
        return 0
    return 0


def read_log_slice(
    log_path: Path,
    *,
    offset: int | None = None,
    tail_bytes: int = 8192,
    raw_log: bool = False,
    binary: bool = False,
    active: bool = False,
) -> dict:
    if tail_bytes < 0:
        raise ValueError("--tail-bytes must be non-negative")
    if offset is not None and offset < 0:
        raise ValueError("--offset must be non-negative")
    try:
        fd = os.open(log_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        result: dict = {"total_bytes": 0, "offset": 0, "next_offset": 0, "truncated": False}
        if not binary:
            result["text"] = ""
        if raw_log or binary:
            result["data"] = ""
        return result
    with os.fdopen(fd, "rb") as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Run log is not a regular file: {log_path}")
        size = info.st_size
        if offset is not None:
            start = max(0, min(offset, size))
            f.seek(start)
            chunk = f.read(tail_bytes) if tail_bytes > 0 else f.read()
        elif tail_bytes > 0 and size > tail_bytes:
            start = size - tail_bytes
            f.seek(start)
            chunk = f.read(tail_bytes)
            if not binary:
                skip = 0
                while skip < min(3, len(chunk)) and (0x80 <= chunk[skip] <= 0xBF):
                    skip += 1
                if skip:
                    start += skip
                    chunk = chunk[skip:]
        else:
            start = 0
            chunk = f.read()

        if not binary:
            more_in_file = (start + len(chunk)) < size
            if more_in_file or active:
                incomplete = _trailing_incomplete_utf8_len(chunk)
                if incomplete:
                    if incomplete < len(chunk):
                        chunk = chunk[:-incomplete]
                    elif more_in_file:
                        # Single multi-byte char exceeds tiny requested page: peek ahead safely and
                        # only consume valid continuation bytes so invalid bytes aren't eaten.
                        needed = _utf8_expected_len(chunk[0]) - len(chunk)
                        if needed > 0:
                            extra = f.read(needed)
                            cand = chunk + extra
                            valid_len = len(chunk)
                            for idx in range(len(chunk), len(cand)):
                                b = cand[idx]
                                if not (0x80 <= b <= 0xBF):
                                    break
                                if idx == 1:
                                    b0 = cand[0]
                                    if (b0 == 0xE0 and b < 0xA0) or (b0 == 0xED and b >= 0xA0):
                                        break
                                    if (b0 == 0xF0 and b < 0x90) or (b0 == 0xF4 and b >= 0x90):
                                        break
                                valid_len = idx + 1
                            chunk = cand[:valid_len]
                    elif active:
                        # Only incomplete sequence at active EOF: hold back until writer finishes codepoint.
                        chunk = b""

    next_offset = start + len(chunk)
    truncated = next_offset < size if offset is not None else (start > 0 or next_offset < size)
    result: dict = {
        "total_bytes": size,
        "offset": start,
        "next_offset": next_offset,
        "truncated": truncated,
    }
    if not binary:
        decoded = chunk.decode("utf-8", errors="replace")
        result["text"] = decoded if raw_log else fold_cr(decoded)
    if raw_log or binary:
        result["data"] = base64.b64encode(chunk).decode("ascii")
    return result


# === Security, Filesystem & Process Primitives ===

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_private_text(path: Path, value: str) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    write_private_text(path, json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n")


@functools.lru_cache(maxsize=16)
def is_single_user_group(gid: int, euid: int) -> bool:
    try:
        pw = pwd.getpwuid(euid)
        gr = grp.getgrgid(gid)
    except (KeyError, OSError):
        return False
    if pw.pw_gid != gid or gr.gr_name != pw.pw_name:
        return False
    if any(member != pw.pw_name for member in gr.gr_mem):
        return False
    try:
        entries = pwd.getpwall()
    except OSError:
        return False
    return all(entry.pw_uid == euid for entry in entries if entry.pw_gid == gid)


def check_directory_chain(path: Path, label: str, *, trusted: bool = False) -> None:
    euid = os.geteuid()
    admin_uid = Path(path.anchor).stat().st_uid if trusted else None
    private_parent = False
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} includes a symlink or non-directory: {part}")
        if trusted and not private_parent:
            sticky = bool(info.st_mode & stat.S_ISVTX)
            bad_owner = info.st_uid not in (0, admin_uid, euid)
            world_writable = bool(info.st_mode & 0o002) and not sticky
            group_writable = bool(info.st_mode & 0o020) and not sticky and not (
                info.st_uid == euid and is_single_user_group(info.st_gid, euid)
            )
            if bad_owner or world_writable or group_writable:
                raise ValueError(
                    f"{label} includes a directory writable by another user: {part} "
                    f"(fix permissions or choose a root under your home directory)"
                )
        private_parent |= (
            info.st_uid == euid
            and not (info.st_mode & 0o001)
            and (not (info.st_mode & 0o010) or is_single_user_group(info.st_gid, euid))
        )


def private_directory(path: Path, *, create: bool = True, tighten: bool = True) -> bool:
    check_directory_chain(path, "Private directory", trusted=True)
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        if create:
            raise
        return False
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid():
            raise ValueError(f"Unsafe directory ownership or permissions: {path}")
        if info.st_mode & 0o077:
            if create and tighten:
                os.fchmod(fd, 0o700)
            elif not tighten:
                raise ValueError(f"Directory must be private (0700): {path}")
        return True
    finally:
        os.close(fd)


def open_lock(path: Path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
        or info.st_nlink != 1
    ):
        os.close(fd)
        raise ValueError(f"Unsafe lock file: {path}")
    return os.fdopen(fd, "r+b")


def run_cmd(argv: list[str], *, cwd=None, env=None, stdin=None, timeout: float = 30, trim: bool = True) -> str:
    completed = subprocess.run(
        argv, cwd=cwd, env=env, input=stdin, capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{argv[0]} exited {completed.returncode}: {detail[-1500:]}")
    return completed.stdout.strip() if trim else completed.stdout


def session_name(root: Path, run_id: str) -> str:
    return "rr-" + sha256((str(root) + "/" + run_id).encode()).hexdigest()[:20]


def session_key_for(workdir: str, name: str) -> str:
    return sha256(f"{workdir}::{name}".encode()).hexdigest()[:20]


def pid_alive(pgid) -> bool:
    if type(pgid) is not int or pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True


def single_pid_alive(pid) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True


@functools.lru_cache(maxsize=1)
def boot_identity() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def _read_proc_stat_fields(stat_path: Path) -> list[str]:
    """Parse fields after '(comm) ' from Linux /proc/*/stat using raw bytes (comm may be truncated mid-UTF-8)."""
    raw = stat_path.read_bytes()
    tail = raw.rpartition(b") ")[2]
    if not tail:
        return []
    return tail.decode("ascii", errors="strict").split()


def process_identity(pid) -> list[str] | None:
    if type(pid) is not int or pid < 1 or not boot_identity():
        return None
    try:
        fields = _read_proc_stat_fields(Path(f"/proc/{pid}/stat"))
        return [boot_identity(), fields[19], fields[2], fields[3]]
    except (IndexError, OSError, UnicodeError):
        return None


def owns_process(pid, expected, *, group: bool = False) -> bool:
    identity = process_identity(pid)
    return (
        identity is not None
        and identity == expected
        and (not group or identity[2:] == [str(pid), str(pid)])
    )


def run_process_alive(state: dict) -> bool:
    worker_pid, pgid = state.get("worker_pid"), state.get("child_pgid")
    worker = (
        owns_process(worker_pid, state["worker_identity"])
        if "worker_identity" in state
        else single_pid_alive(worker_pid)
    )
    group = (
        owns_process(pgid, state["child_identity"], group=True)
        if "child_identity" in state
        else pid_alive(pgid)
    )
    return worker or group


# === Component 2: Remote Path & Workspace Safety ===

def validate_remote_root(root: Path) -> Path:
    r = root.resolve()
    home = Path.home().resolve()
    if r == Path("/") or r == home or home.is_relative_to(r):
        raise ValueError(
            f"Refusing unsafe remote root {r} (cannot be '/', remote $HOME, or an ancestor of $HOME)"
        )
    return r


def validate_remote_workdir(root: Path, wdir: Path) -> Path:
    r = validate_remote_root(root)
    w = wdir.resolve()
    home = Path.home().resolve()
    if w == Path("/") or w == home or home.is_relative_to(w):
        raise ValueError(
            f"Refusing unsafe remote workdir {w} (cannot be '/', remote $HOME, or an ancestor of $HOME)"
        )
    if w == r or r.is_relative_to(w):
        raise ValueError(f"Refusing remote workdir {w}: cannot equal or contain remote root {r}")
    for reserved in ("runs", "locks", "sessions", "workspace-locks", "retired-runs", "pull-stages"):
        if w.is_relative_to(r / reserved) or w.is_relative_to((r / reserved).resolve()):
            raise ValueError(f"Refusing remote workdir {w}: inside reserved control dir {r / reserved}")
    proj_base = r / "projects"
    if w.is_relative_to(proj_base):
        rel_parts = w.relative_to(proj_base).parts
        if len(rel_parts) >= 2 and rel_parts[1] == "shared":
            raise ValueError(f"Refusing remote workdir {w}: inside project shared dir")
        if len(rel_parts) != 3 or rel_parts[1] != "workspaces":
            raise ValueError(
                f"Refusing remote workdir {w}: paths under projects/ must be projects/<project>/workspaces/<workspace>"
            )
    return w


def validate_locks_dir(root: Path, locks_dir: Path, workdir: Path | None = None) -> None:
    if locks_dir == root / "locks":
        check_directory_chain(locks_dir, "GPU locks directory", trusted=True)
        return
    validate_remote_root(locks_dir)
    path = locks_dir.resolve()
    reserved = (
        root / name
        for name in ("runs", "sessions", "projects", "workspace-locks", "retired-runs", "pull-stages", "adhoc")
    )
    if root.resolve().is_relative_to(path) or any(path.is_relative_to(d.resolve()) for d in reserved):
        raise ValueError(f"GPU locks directory overlaps remote state: {locks_dir}")
    if workdir and (path.is_relative_to(workdir.resolve()) or workdir.resolve().is_relative_to(path)):
        raise ValueError(f"GPU locks directory overlaps the workdir: {locks_dir}")
    private_directory(locks_dir, create=False, tighten=False)


def ensure_root_owned(root: Path, *, dry_run: bool = False) -> str:
    """Validate ownership of root. Returns 'ready' or 'pending' (when dry_run and not yet initialized)."""
    validate_remote_root(root)
    check_directory_chain(root, "Remote root", trusted=True)
    marker = root / ROOT_MARKER
    if marker.exists() or marker.is_symlink():
        info = marker.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o022
            or info.st_nlink != 1
            or not isinstance(data := read_json(marker), dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != SCHEMA_VERSION
            or not isinstance(data.get("created_at"), str)
        ):
            raise ValueError(f"Invalid v2 ownership marker: {marker}")
        return "ready"
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"Configured remote root is not a directory: {root}")
        if any(root.iterdir()):
            raise ValueError(
                f"Refusing to use non-empty unmanaged remote root {root}; "
                f"choose a clean v2 root directory (default: {DEFAULT_REMOTE_ROOT})"
            )
    if dry_run:
        return "pending"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json(marker, {"schema_version": SCHEMA_VERSION, "created_at": now()})
    return "ready"


# === Component 3: Trusted Tools, Layered PATH & Executable Probes ===

def _is_searchable_dir(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.X_OK)


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def build_trusted_host_path(ssh_base_path: str, host_paths: list[str]) -> str:
    """Trusted host PATH for control tools (bash/rsync/tmux/nvidia-smi): host.path -> ssh_base_path.
    Validates each host_paths entry is a trusted non-symlink directory chain before using it.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in host_paths:
        exp = expand_remote_path(raw, "host.path")
        p = Path(exp)
        if p.exists():
            check_directory_chain(p, "host.path", trusted=True)
        if exp not in seen:
            seen.add(exp)
            ordered.append(exp)
    for entry in ssh_base_path.split(os.pathsep):
        if os.path.isabs(entry) and entry not in seen:
            seen.add(entry)
            ordered.append(entry)
    return os.pathsep.join(ordered)


def resolve_path_layers(
    layers: list[dict],
    ssh_base_path: str,
    *,
    workdir: Path | None = None,
    strict: bool = True,
) -> tuple[str, list[dict]]:
    """Resolve layered PATH entries left-to-right (highest priority first).

    Each item in `layers` is {"source": str, "entry": str, "allow_pending": bool}.
    Explicit absolute/tilde entries must be searchable directories when strict=True.
    Project-relative entries missing on disk are reported as 'pending' and kept in PATH.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    report: list[dict] = []

    for item in layers:
        source = item["source"]
        raw = item["entry"]
        allow_pending = bool(item.get("allow_pending", False))
        if raw.startswith("/") or raw.startswith("~"):
            resolved = expand_remote_path(raw, f"{source}.path")
            is_rel = False
        else:
            if workdir is None:
                raise ValueError(f"Relative path entry {raw!r} from {source} requires a project workspace")
            rel = rel_path(raw, f"{source}.path")
            resolved = str(workdir / rel)
            is_rel = True

        p = Path(resolved)
        if p.exists():
            if _is_searchable_dir(p):
                status = "ok"
            else:
                status = "unusable"
                if strict:
                    raise ValueError(
                        f"Configured PATH entry from {source} is not a searchable directory: {resolved}"
                    )
        elif allow_pending and is_rel:
            status = "pending"
        else:
            status = "missing"
            if strict:
                raise ValueError(f"Configured PATH entry from {source} does not exist on host: {resolved}")

        duplicate = resolved in seen
        if not duplicate:
            seen.add(resolved)
            ordered.append(resolved)
        report.append({
            "source": source,
            "entry": raw,
            "resolved": resolved,
            "status": status,
            "duplicate": duplicate,
        })

    for entry in ssh_base_path.split(os.pathsep):
        if not os.path.isabs(entry):
            continue
        norm = str(PurePosixPath(entry))
        duplicate = norm in seen
        if not duplicate:
            seen.add(norm)
            ordered.append(norm)
        report.append({
            "source": "ssh_base",
            "entry": entry,
            "resolved": norm,
            "status": "ok" if _is_searchable_dir(Path(norm)) else "missing",
            "duplicate": duplicate,
        })

    return os.pathsep.join(ordered), report


def probe_executable(spec: str, effective_path: str, *, cwd: Path | None = None) -> dict:
    """Read-only filesystem probe for bare names (PATH lookup) or literal paths (contains '/')."""
    if not isinstance(spec, str) or not spec or any(ord(c) < 32 for c in spec):
        raise ValueError(f"Invalid command probe {spec!r}")
    if "/" in spec or spec.startswith("~"):
        if spec.startswith("~"):
            expanded = Path(expand_remote_path(spec, "command"))
            kind = "literal_path"
        elif spec.startswith("/"):
            expanded = Path(PurePosixPath(spec))
            if ".." in PurePosixPath(spec).parts:
                raise ValueError(f"Command path cannot contain '..': {spec!r}")
            kind = "literal_path"
        else:
            if cwd is None:
                return {
                    "input": spec, "kind": "relative_path", "resolved": None,
                    "ok": False, "reason": "relative_command_requires_workdir",
                }
            rel = rel_path(spec, "command")
            expanded = cwd / rel
            kind = "relative_path"
        if not expanded.exists():
            return {"input": spec, "kind": kind, "resolved": str(expanded), "ok": False, "reason": "not_found"}
        if not expanded.is_file():
            return {"input": spec, "kind": kind, "resolved": str(expanded), "ok": False, "reason": "not_regular_file"}
        if not os.access(expanded, os.X_OK):
            return {"input": spec, "kind": kind, "resolved": str(expanded), "ok": False, "reason": "not_executable"}
        return {"input": spec, "kind": kind, "resolved": str(expanded), "ok": True, "reason": "ok"}

    # Bare executable name
    for directory in effective_path.split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / spec
        if _is_executable_file(candidate):
            return {"input": spec, "kind": "bare_name", "resolved": str(candidate), "ok": True, "reason": "ok"}
    return {"input": spec, "kind": "bare_name", "resolved": None, "ok": False, "reason": "not_found_in_path"}


@functools.lru_cache(maxsize=1)
def rsync_supports_protect_args(rsync_bin: str = "rsync") -> bool:
    completed = subprocess.run([rsync_bin, "--help"], capture_output=True, text=True)
    help_text = completed.stdout + completed.stderr
    return "--protect-args" in help_text or "--secluded-args" in help_text


def remote_rsync_info(trusted_path: str) -> dict | None:
    path = shutil.which("rsync", path=trusted_path)
    return {"path": path, "secluded": rsync_supports_protect_args(path)} if path else None


def trusted_tool(name: str, trusted_path: str) -> str:
    found = shutil.which(name, path=trusted_path)
    if not found:
        raise ValueError(f"Required tool {name!r} not found in trusted host PATH")
    return found


# === Component 4: Optional NVIDIA GPU Inventory & UUID-Ordered Locking ===

def query_nvidia_gpus(trusted_path: str) -> list[dict]:
    smi = shutil.which("nvidia-smi", path=trusted_path)
    if not smi:
        raise ValueError("Host has [gpu] configured but 'nvidia-smi' was not found in trusted host PATH")
    raw = run_cmd(
        [smi, "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"],
        timeout=10,
    )
    gpus: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].isdigit() and parts[1]:
            mem_match = re.search(r"([0-9]+)", parts[3])
            gpus.append({
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_mib": int(mem_match.group(1)) if mem_match else None,
            })
        else:
            raise ValueError(f"Unexpected nvidia-smi output line: {line!r}")
    return gpus


def resolve_gpu_allocation(
    requested_indices: list[int],
    configured_devices: list[int],
    trusted_path: str,
) -> tuple[list[int], list[str], list[str]]:
    """Validate requested indices against host inventory and return (indices, uuids_in_order, sorted_lock_keys)."""
    if not requested_indices:
        return [], [], []
    if not configured_devices:
        raise ValueError("Host does not declare [gpu].devices; cannot allocate GPUs")
    for idx in requested_indices:
        if idx not in configured_devices:
            raise ValueError(
                f"Requested GPU index {idx} is not in host [gpu].devices {configured_devices}"
            )
    inventory = {g["index"]: g for g in query_nvidia_gpus(trusted_path)}
    uuids_in_order: list[str] = []
    for idx in requested_indices:
        if idx not in inventory:
            raise ValueError(
                f"Configured GPU index {idx} not reported by nvidia-smi (available: {sorted(inventory)})"
            )
        uuids_in_order.append(inventory[idx]["uuid"])
    lock_keys = sorted(
        f"gpu-{re.sub(r'[^A-Za-z0-9_.-]+', '_', u)}" for u in set(uuids_in_order)
    )
    return list(requested_indices), uuids_in_order, lock_keys


# === Component 2 (Remote): Workspace Prep, Shared Links & Manifest Reconciliation ===

def read_synced_manifest(wdir: Path) -> dict:
    mpath = wdir / SYNCED_MANIFEST
    if mpath.is_symlink() or not mpath.is_file():
        return {}
    try:
        data = read_json(mpath)
        files = data.get("files") if isinstance(data, dict) and data.get("schema_version") == SCHEMA_VERSION else None
        return files if isinstance(files, dict) else {}
    except (ValueError, OSError):
        return {}


@contextmanager
def workspace_file(workdir: Path, rel: str):
    parts = PurePosixPath(rel).parts
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        directory = os.open(workdir, flags)
        stack.callback(os.close, directory)
        for part in parts[:-1]:
            try:
                directory = os.open(part, flags, dir_fd=directory)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(f"Symlink parent in workspace path: {rel}") from exc
                raise
            stack.callback(os.close, directory)
        yield directory, parts[-1]


def file_version(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size]


def file_snapshot(directory: int, name: str) -> dict:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"Refusing a symlink in the sync manifest: {name}") from exc
        raise
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Sync manifest path is not a regular file: {name}")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        version = file_version(info)
        if version != file_version(os.fstat(fd)):
            raise ValueError(f"File changed while checking sync ownership: {name}")
    return {"version": version, "sha256": digest}


def remove_verified_file(directory: int, name: str, expected: dict, path: Path) -> bool:
    holding = ".remote-run-gc-" + uuid4().hex
    os.mkdir(holding, mode=0o700, dir_fd=directory)
    try:
        fd = os.open(holding, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            os.rename(name, "file", src_dir_fd=directory, dst_dir_fd=fd)
            try:
                actual = file_snapshot(fd, "file")
            except (OSError, ValueError):
                actual = None
            matched = actual and actual["sha256"] == expected["sha256"] and (
                actual["version"][:3] + actual["version"][4:]
                == expected["version"][:3] + expected["version"][4:]
            )
            if not matched:
                try:
                    os.link("file", name, src_dir_fd=fd, dst_dir_fd=directory, follow_symlinks=False)
                except OSError as exc:
                    recovery = path.parent / holding / "file"
                    raise RuntimeError(f"File changed during cleanup. Recover it from {recovery}") from exc
            os.unlink("file", dir_fd=fd)
            return bool(matched)
        finally:
            os.close(fd)
    finally:
        with suppress(OSError):
            os.rmdir(holding, dir_fd=directory)


def active_runs_for_workdir(root: Path, workdir: str, trusted_path: str) -> list[str]:
    runs_dir = root / "runs"
    if not private_directory(runs_dir, create=False):
        return []
    active = []
    target = Path(workdir).resolve()
    for child in runs_dir.iterdir():
        st_path = child / "state.json"
        if not child.is_dir() or not st_path.is_file() or not (child / "request.json").is_file():
            continue
        try:
            raw_st = read_json(st_path)
            raw_status = raw_st.get("status") if isinstance(raw_st, dict) else None
            if not isinstance(raw_status, str) or raw_status in TERMINAL:
                continue
            st = state_of(child, root, trusted_path)
        except (ValueError, KeyError, TypeError, OSError):
            continue
        other = Path(st["workdir"]).resolve()
        if st["status"] not in TERMINAL and (other.is_relative_to(target) or target.is_relative_to(other)):
            active.append(st["run_id"])
    return active


def _read_workspace_marker(wdir: Path) -> dict | None:
    """Read and strictly validate .remote-run-workspace.json anchored to wdir."""
    marker = wdir / WORKSPACE_MARKER
    if not (marker.exists() or marker.is_symlink()):
        return None
    info = marker.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
        or info.st_nlink != 1
    ):
        raise ValueError(f"Unsafe workspace marker: {marker}")
    found = read_json(marker)
    if not isinstance(found, dict) or found.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Corrupt or incompatible workspace marker at {marker}")
    return found


def _validate_workspace_owner(wdir: Path, found: dict, expected: dict) -> None:
    managed = bool(expected.get("managed", True))
    mismatch = (
        found.get("project_id") != expected["project_id"]
        or (
            found.get("root_commit", "")
            and expected["root_commit"]
            and found["root_commit"] != expected["root_commit"]
        )
        or (managed and found.get("workspace_id") != expected["workspace_id"])
    )
    if mismatch:
        raise ValueError(
            f"Refusing workspace access: {wdir} belongs to a different project/workspace "
            f"({found.get('project_id')}/{found.get('workspace_id')}), expected "
            f"({expected['project_id']}/{expected['workspace_id']})"
        )


def _validated_registered_shares(raw_reg: object, sdir_raw: str) -> dict[str, str]:
    if not isinstance(raw_reg, dict):
        return {}
    out: dict[str, str] = {}
    sdir = Path(sdir_raw) if sdir_raw else None
    for k, v in raw_reg.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        try:
            rel = rel_path(k, "registered shared_link")
            target = remote_path(v, "registered shared_link target")
        except ValueError:
            continue
        if sdir is not None and str(sdir / rel) == target:
            out[rel] = target
    return out


def prepare_remote_workspace(
    root: Path,
    ws_info: dict,
    shared_dirs: list[str],
    *,
    trusted_path: str,
    check_run_id: str = "",
    dry_run: bool = False,
    for_sync: bool = True,
    create_if_missing: bool = True,
    manifest_parents: list[str] = (),
    locks_dir: Path | None = None,
) -> dict:
    wdir = Path(expand_remote_path(ws_info["workdir"], "workdir"))
    validate_remote_workdir(root, wdir)
    if locks_dir is not None:
        validate_locks_dir(root, locks_dir, wdir)
    check_directory_chain(wdir, "Remote workdir", trusted=for_sync)
    ensure_root_owned(root, dry_run=dry_run)
    sdir_raw = expand_remote_path(ws_info["shared_dir"], "shared_dir") if ws_info.get("shared_dir") else ""

    run_exists = bool(check_run_id and (root / "runs" / identifier(check_run_id, "run_id")).exists())
    if run_exists:
        return {
            "root": str(root),
            "workdir": str(wdir),
            "shared_dir": sdir_raw,
            "shared_mounted": list(shared_dirs),
            "run_exists": True,
            "workspace_exists": wdir.is_dir(),
            "active_runs_warning": [],
            "prev_synced_files": [],
        }

    # Check active runs BEFORE any filesystem mutation so a rejected sync never touches mounts or markers.
    active_runs = active_runs_for_workdir(root, str(wdir), trusted_path)
    if for_sync and not dry_run and active_runs:
        raise ValueError(f"Cannot sync a workspace used by active runs: {', '.join(active_runs)}")

    managed = bool(ws_info.get("managed", True))
    expected = {
        "project_id": ws_info["project_id"],
        "root_commit": ws_info.get("root_commit", ""),
        "managed": managed,
        "workspace_id": ws_info["workspace_id"] if managed else "",
    }
    marker = wdir / WORKSPACE_MARKER
    found = _read_workspace_marker(wdir)
    registered_shares: dict[str, str] = {}
    if found is not None:
        _validate_workspace_owner(wdir, found, expected)
        registered_shares = _validated_registered_shares(found.get("shared_links", {}), sdir_raw)
        if not for_sync and managed:
            # Project --no-sync strictly requires a completed sync flag (a pre-existing manifest alone is insufficient).
            if found.get("sync_completed") is not True:
                raise ValueError(
                    f"Managed workspace {wdir} has not completed an initial sync; run `remote-run sync` first"
                )
        elif for_sync and not dry_run and found.get("sync_completed") is True:
            # Mark sync in-progress until reconcile_synced_manifest completes cleanly!
            cur = dict(found)
            cur["sync_completed"] = False
            write_json(marker, cur)
    elif for_sync or (managed and create_if_missing):
        if wdir.exists():
            if not wdir.is_dir():
                raise ValueError(f"Remote workdir exists and is not a directory: {wdir}")
            if any(wdir.iterdir()):
                raise ValueError(
                    f"Refusing to use non-empty unmanaged remote directory {wdir}; "
                    "choose a fresh workspace ID or clean v2 root"
                )
        if not dry_run:
            wdir.mkdir(parents=True, exist_ok=True)
            write_json(marker, {
                "schema_version": SCHEMA_VERSION, "created_at": now(),
                **expected, "shared_links": {}, "sync_completed": False,
            })
    else:
        if managed and not wdir.exists():
            raise ValueError(
                f"No remote workspace at {wdir}. Sync it first or use --no-project --no-sync --remote-dir PATH."
            )
        if managed and found is None:
            raise ValueError(f"Refusing unowned managed workspace without marker: {wdir}")
        if wdir.exists() and not wdir.is_dir():
            raise ValueError(f"Remote workdir exists and is not a directory: {wdir}")

    # Verify receiver containment for all incoming manifest parent directories BEFORE rsync runs.
    if for_sync and manifest_parents and wdir.is_dir():
        for raw_parent in manifest_parents:
            p_rel = rel_path(raw_parent, "manifest parent")
            cur = wdir
            for part in PurePosixPath(p_rel).parts:
                cur = cur / part
                try:
                    st_p = cur.lstat()
                except FileNotFoundError:
                    break
                if not stat.S_ISDIR(st_p.st_mode):
                    raise ValueError(
                        f"Refusing sync: receiver parent path {cur} is a symlink or non-directory"
                    )

    desired_shares: dict[str, str] = {}
    if shared_dirs:
        if not sdir_raw:
            raise ValueError("Cannot mount project shared directories without a managed shared_dir")
        sdir = Path(sdir_raw)
        check_directory_chain(sdir, "Project shared directory", trusted=True)
        # Validate all shared_dirs upfront before any mutation (prevents partial mounts).
        planned: list[tuple[str, Path, Path]] = []
        for name in shared_dirs:
            rel = rel_path(name, "workspace.shared")
            link_path = wdir / rel
            target_path = sdir / rel
            check_directory_chain(link_path.parent, "Project shared mount parent", trusted=True)
            check_directory_chain(target_path, "Project shared directory", trusted=True)
            expected_target = str(target_path)
            desired_shares[rel] = expected_target
            if link_path.is_symlink():
                actual_target = os.readlink(link_path)
                is_owned = registered_shares.get(rel) == actual_target
                if not is_owned or link_path.resolve() != target_path.resolve():
                    raise ValueError(f"Refusing to reuse unowned or mismatched symlink at {link_path}")
            elif not for_sync:
                raise ValueError(
                    f"Shared directory {rel!r} is not mounted in {wdir}; run `remote-run sync` to update mounts"
                )
            elif link_path.exists():
                if link_path.is_dir():
                    if any(link_path.iterdir()):
                        raise ValueError(
                            f"Cannot mount shared dir {rel!r}: non-empty directory exists at {link_path}"
                        )
                else:
                    raise ValueError(f"Cannot mount shared dir {rel!r}: file exists at {link_path}")
            planned.append((rel, link_path, target_path))

        # Only mutate shared mounts during sync (never during --no-sync while active runs may read workspace).
        if for_sync and not dry_run:
            sdir.mkdir(parents=True, exist_ok=True)
            if marker.is_file():
                cur = _read_workspace_marker(wdir) or {}
                cur["shared_links"] = {**registered_shares, **desired_shares}
                write_json(marker, cur)
            for rel, link_path, target_path in planned:
                target_path.mkdir(parents=True, exist_ok=True)
                link_path.parent.mkdir(parents=True, exist_ok=True)
                with workspace_file(wdir, rel) as (parent_fd, leaf):
                    try:
                        st_leaf = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        os.symlink(str(target_path), leaf, dir_fd=parent_fd)
                    else:
                        if stat.S_ISLNK(st_leaf.st_mode):
                            if os.readlink(leaf, dir_fd=parent_fd) != str(target_path):
                                raise ValueError(f"Refusing to replace changed symlink at {link_path}")
                        elif stat.S_ISDIR(st_leaf.st_mode):
                            os.rmdir(leaf, dir_fd=parent_fd)
                            os.symlink(str(target_path), leaf, dir_fd=parent_fd)
                        else:
                            raise ValueError(f"Cannot mount shared dir {rel!r}: file exists at {link_path}")

    # Unlink previously registered workspace shares removed from config (only when syncing, descriptor-anchored).
    if for_sync and not dry_run and marker.is_file():
        for old_rel, old_target in list(registered_shares.items()):
            if old_rel not in desired_shares:
                try:
                    with workspace_file(wdir, old_rel) as (parent_fd, leaf):
                        st_leaf = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                        if stat.S_ISLNK(st_leaf.st_mode) and os.readlink(leaf, dir_fd=parent_fd) == old_target:
                            os.unlink(leaf, dir_fd=parent_fd)
                except (ValueError, FileNotFoundError, NotADirectoryError):
                    pass
        cur = _read_workspace_marker(wdir) or {}
        if cur.get("shared_links") != desired_shares:
            cur["shared_links"] = desired_shares
            write_json(marker, cur)

    prev_synced = sorted(read_synced_manifest(wdir)) if (for_sync and wdir.is_dir()) else []
    return {
        "root": str(root),
        "workdir": str(wdir),
        "shared_dir": sdir_raw,
        "shared_mounted": list(desired_shares),
        "run_exists": False,
        "workspace_exists": wdir.is_dir(),
        "active_runs_warning": active_runs,
        "prev_synced_files": prev_synced,
    }


def reconcile_synced_manifest(
    root: Path,
    workdir_raw: str,
    uploaded_files: dict[str, str],
    prune_candidates: list[str],
    protected_prefixes: list[str],
    *,
    uploaded_symlinks: dict[str, str] | None = None,
    dry_run: bool = False,
) -> dict:
    wdir = Path(expand_remote_path(workdir_raw, "workdir"))
    validate_remote_workdir(root, wdir)
    check_directory_chain(wdir, "Remote workdir", trusted=True)
    if _read_workspace_marker(wdir) is None:
        return {"pruned_ghost_files": []}
    owned = read_synced_manifest(wdir)
    prot = {p.strip("/") for p in protected_prefixes if isinstance(p, str) and p.strip("/")}
    markers = set(CONTROL_ROOT_FILES)
    up_links = uploaded_symlinks or {}

    def unprotected(rel: str) -> bool:
        return rel not in markers and not any(rel == p or rel.startswith(p + "/") for p in prot)

    pruned: list[str] = []
    for raw in sorted(set(prune_candidates) & owned.keys() - set(uploaded_files) - set(up_links)):
        try:
            rel = rel_path(raw, "stale file")
            if not unprotected(rel):
                continue
            with workspace_file(wdir, rel) as (directory, name):
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                expected = owned[raw]
                if isinstance(expected, dict) and isinstance(expected.get("symlink"), str):
                    ver_ok = (
                        isinstance(expected.get("version"), list)
                        and file_version(info) == expected["version"]
                    )
                    if stat.S_ISLNK(info.st_mode) and ver_ok and os.readlink(name, dir_fd=directory) == expected["symlink"]:
                        if dry_run:
                            pruned.append(rel)
                        else:
                            os.unlink(name, dir_fd=directory)
                            pruned.append(rel)
                elif (
                    stat.S_ISREG(info.st_mode)
                    and isinstance(expected, dict)
                    and isinstance(expected.get("sha256"), str)
                    and file_version(info) == expected.get("version")
                ):
                    if dry_run:
                        snap = file_snapshot(directory, name)
                        if snap["sha256"] == expected["sha256"]:
                            pruned.append(rel)
                    elif remove_verified_file(directory, name, expected, wdir / rel):
                        pruned.append(rel)
            owned.pop(raw, None)
        except (ValueError, FileNotFoundError, NotADirectoryError):
            owned.pop(raw, None)
            continue

    if not dry_run:
        for raw, digest in uploaded_files.items():
            try:
                rel = rel_path(raw, "synced file")
                if unprotected(rel):
                    with workspace_file(wdir, rel) as (directory, name):
                        snapshot = file_snapshot(directory, name)
                        if snapshot["sha256"] == digest:
                            owned[rel] = snapshot
            except (ValueError, FileNotFoundError, NotADirectoryError):
                continue
        for raw, target in up_links.items():
            try:
                rel = rel_path(raw, "synced symlink")
                if unprotected(rel):
                    with workspace_file(wdir, rel) as (directory, name):
                        st_l = os.stat(name, dir_fd=directory, follow_symlinks=False)
                        if stat.S_ISLNK(st_l.st_mode) and os.readlink(name, dir_fd=directory) == target:
                            owned[rel] = {"symlink": target, "version": file_version(st_l)}
            except (ValueError, FileNotFoundError, NotADirectoryError):
                continue
        write_json(wdir / SYNCED_MANIFEST, {
            "schema_version": SCHEMA_VERSION,
            "updated_at": now(),
            "files": owned,
        })
        cur_marker = _read_workspace_marker(wdir)
        if cur_marker is not None and not cur_marker.get("sync_completed"):
            cur_marker["sync_completed"] = True
            write_json(wdir / WORKSPACE_MARKER, cur_marker)
    return {"pruned_ghost_files": pruned}


# === Component 3 & 4 (Remote): Run State, Queue Inspection, Submit, Cancel ===

def _valid_lock_keys(raw) -> set[str]:
    if not isinstance(raw, list) or not all(isinstance(s, str) and s for s in raw):
        return set()
    return set(raw)


def _resolved_locks_dir(req: dict) -> Path | None:
    raw = req.get("locks_dir")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return Path(expand_remote_path(raw, "locks_dir")).resolve()
    except (ValueError, OSError, RuntimeError):
        return None


def queue_holder_runs(root: Path, job: Path, req: dict) -> list[str]:
    try:
        wanted = _valid_lock_keys(req.get("gpu_lock_keys"))
        target_locks = _resolved_locks_dir(req)
        if not wanted or target_locks is None:
            return []
        runs_dir = root / "runs"
        if not runs_dir.is_dir():
            return []
        children = sorted(runs_dir.iterdir())
    except OSError:
        return []
    holders: list[str] = []
    for child in children:
        if child.name == job.name:
            continue
        try:
            if child.is_symlink() or not child.is_dir():
                continue
            st_p = child / "state.json"
            req_p = child / "request.json"
            if st_p.is_symlink() or req_p.is_symlink() or not st_p.is_file() or not req_p.is_file():
                continue
            other_req = read_json(req_p)
            if not isinstance(other_req, dict):
                continue
            other_locks = _resolved_locks_dir(other_req)
            if other_locks is None or other_locks != target_locks:
                continue
            if not (wanted & _valid_lock_keys(other_req.get("gpu_lock_keys"))):
                continue
            other_st = read_json(st_p)
            if not isinstance(other_st, dict) or other_st.get("status") in TERMINAL:
                continue
            if not run_process_alive(other_st):
                continue
            # Include running runs and queued runs that already acquired partial locks (holding >=1 lock).
            if other_st.get("status") == "running" or other_st.get("held_gpu_locks"):
                holders.append(child.name)
        except (ValueError, KeyError, TypeError, OverflowError, RuntimeError, OSError):
            continue
    return holders


def state_of(job: Path, root: Path, trusted_path: str) -> dict:
    state_path = job / "state.json"
    request_path = job / "request.json"
    if job.is_symlink() or state_path.is_symlink() or request_path.is_symlink():
        raise ValueError(f"Unsafe run metadata: {job}")
    if not state_path.is_file():
        raise ValueError(f"No such run: {job.name}")
    state = read_json(state_path)
    if not isinstance(state, dict):
        raise ValueError(f"Corrupt run metadata for {job.name}")
    t_session = session_name(root, job.name)
    if state["status"] not in TERMINAL and not run_process_alive(state):
        submitted = state.get("submitted_at")
        starting = state["status"] == "queued" and submitted and (
            datetime.now(timezone.utc) - datetime.fromisoformat(submitted)
        ).total_seconds() < 5
        tmux_bin = shutil.which("tmux", path=trusted_path)
        has_tmux = starting or (
            tmux_bin is not None
            and subprocess.run([tmux_bin, "has-session", "-t", "=" + t_session], capture_output=True, timeout=5).returncode == 0
        )
        if not has_tmux:
            state = read_json(state_path)
            if not isinstance(state, dict):
                raise ValueError(f"Corrupt run metadata for {job.name}")
            if state["status"] not in TERMINAL and not run_process_alive(state):
                state["status"] = "cancelled" if (job / "cancel.request").exists() else "interrupted"
                state.setdefault("finished_at", now())
    req = read_json(request_path)
    if not isinstance(req, dict):
        raise ValueError(f"Corrupt run metadata for {job.name}")
    elapsed = None
    if state.get("started_at"):
        t0 = datetime.fromisoformat(state["started_at"])
        t1 = datetime.fromisoformat(state["finished_at"]) if state.get("finished_at") else datetime.now(timezone.utc)
        elapsed = round(max(0.0, (t1 - t0).total_seconds()), 3)
    out = {
        **state,
        "host": req["host"],
        "gpus": req.get("gpus", []),
        "gpu_uuids": req.get("gpu_uuids", []),
        "project_id": req.get("project_id", ""),
        "workspace_id": req.get("workspace_id", ""),
        "workdir": req["workdir"],
        "shared_dir": req.get("shared_dir", ""),
        "session": req.get("session", ""),
        "tmux_session": t_session,
        "elapsed_sec": elapsed,
        "git_commit": req.get("git_commit", ""),
        "git_branch": req.get("git_branch", ""),
        "git_dirty": req.get("git_dirty", False),
    }
    if out.get("status") == "queued" and out.get("waiting_for_lock"):
        holders = queue_holder_runs(root, job, req)
        if holders:
            out["blocked_by_runs"] = holders
    return out


def wait_job_terminal(
    job: Path,
    root: Path,
    wait_sec: float,
    trusted_path: str,
    *,
    include_log: bool = True,
    offset: int | None = None,
    tail_bytes: int = 8192,
    raw_log: bool = False,
    wake_on_output: bool = False,
    binary_log: bool = False,
) -> dict:
    deadline = time.monotonic() + wait_sec
    log_path = job / "run.log"
    state = state_of(job, root, trusted_path)
    while state["status"] not in TERMINAL and time.monotonic() < deadline:
        if wake_on_output and offset is not None and log_path.is_file() and log_path.stat().st_size > offset:
            break
        interval = 0.15 if wake_on_output else 0.5
        time.sleep(min(interval, max(0.05, deadline - time.monotonic())))
        state = state_of(job, root, trusted_path)
    if include_log:
        state["log"] = read_log_slice(
            log_path,
            offset=offset,
            tail_bytes=tail_bytes,
            raw_log=raw_log,
            binary=binary_log,
            active=state["status"] not in TERMINAL,
        )
    return state


def _static_request_fields(host_cfg: dict, request: dict) -> dict:
    """Normalize deterministic request fields WITHOUT invoking live nvidia-smi or PATH resolution."""
    req = dict(request)
    req.pop("require_workdir", None)
    req.pop("reuse_only", None)
    gpu_cfg = host_cfg.get("gpu") or {}
    req["gpu_managed"] = bool(gpu_cfg)
    req["gpus"] = validate_gpu_indices(req.get("gpus", []), "gpus")
    if gpu_cfg:
        locks_raw = gpu_cfg.get("locks_dir") or join_remote_path(host_cfg["root"], "locks")
        req["locks_dir"] = expand_remote_path(locks_raw, "gpu.locks_dir")
    else:
        req["locks_dir"] = ""
    if req.get("shared_dir"):
        req["shared_dir"] = expand_remote_path(req["shared_dir"], "shared_dir")
    req["workdir"] = expand_remote_path(req["workdir"], "workdir")
    # Compute session_key AFTER expanding workdir so '~/' and absolute paths always hash identically!
    sess_name = req.get("session", "")
    req["session_key"] = session_key_for(req["workdir"], identifier(sess_name, "session")) if sess_name else ""
    req["config_env"] = check_env_map(
        req.get("config_env", {}), "config_env", reject_reserved=True, gpu_managed=req["gpu_managed"],
    )
    req["invocation_env"] = check_env_map(
        req.get("invocation_env", {}), "invocation_env", reject_reserved=True, gpu_managed=req["gpu_managed"],
    )
    return req


def normalize_request(root: Path, host_cfg: dict, request: dict, trusted_path: str, ssh_base_env: dict) -> dict:
    req = _static_request_fields(host_cfg, request)
    gpu_cfg = host_cfg.get("gpu") or {}
    configured_devices = validate_gpu_indices(gpu_cfg.get("devices", []), "host.gpu.devices") if gpu_cfg else []
    if req["gpus"]:
        indices, uuids, lock_keys = resolve_gpu_allocation(req["gpus"], configured_devices, trusted_path)
        req["gpus"] = indices
        req["gpu_uuids"] = uuids
        req["gpu_lock_keys"] = lock_keys
    else:
        req["gpu_uuids"] = []
        req["gpu_lock_keys"] = []

    workdir = Path(req["workdir"])
    ssh_base_path = ssh_base_env.get("PATH", "")
    effective_path, _ = resolve_path_layers(req.get("path_layers", []), ssh_base_path, workdir=workdir, strict=True)
    req["effective_path"] = effective_path
    req["trusted_path"] = trusted_path
    req["ssh_base_env"] = {
        k: v for k, v in ssh_base_env.items()
        if k not in SKIP_SSH_BASELINE_ENV and not k.startswith(SKIP_SESSION_PREFIXES)
    }
    return req


def _canonical_idempotency_view(req: dict) -> dict:
    """Compare semantic request fields without transient SSH baseline, retry flags, or resolved UUID churn."""
    drop = {"ssh_base_env", "trusted_path", "effective_path", "gpu_uuids", "gpu_lock_keys", "reuse_only"}
    return {k: v for k, v in req.items() if k not in drop}


def remote_submit(
    root: Path,
    job: Path,
    host_cfg: dict,
    request: dict,
    worker_src: str,
    trusted_path: str,
    ssh_base_env: dict,
) -> dict:
    if job.is_symlink() or (job / "request.json").is_symlink():
        raise ValueError(f"Unsafe run metadata: {job}")
    # Check idempotency against saved request BEFORE live GPU/PATH probes so retries succeed even if env changed.
    if job.exists():
        static_req = _static_request_fields(host_cfg, request)
        existing = read_json(job / "request.json")
        if _canonical_idempotency_view(existing) != _canonical_idempotency_view(static_req):
            raise ValueError(f"Run ID {job.name} already belongs to a different request")
        return dict(state_of(job, root, trusted_path), reused=True)
    if request.get("reuse_only"):
        raise ValueError(
            f"Run ID {job.name} was removed while reusing an existing run; retry sync and submit"
        )

    req_to_save = normalize_request(root, host_cfg, request, trusted_path, ssh_base_env)
    workdir = Path(remote_path(req_to_save["workdir"], "workdir"))
    if request.get("require_workdir") and not workdir.is_dir():
        raise ValueError(
            f"No remote workspace at {workdir}. Sync it first or use --no-project --no-sync --remote-dir PATH."
        )
    if request.get("adhoc_existing_dir") and not workdir.is_dir():
        raise ValueError(f"Ad-hoc --remote-dir does not exist on host: {workdir}")
    validate_remote_workdir(root, workdir)
    check_directory_chain(workdir, "Remote workdir", trusted=not bool(request.get("adhoc_existing_dir")))
    if req_to_save.get("project_id") and req_to_save.get("workspace_id"):
        # Managed project workspace MUST already own a valid matching marker with sync_completed=True.
        found_marker = _read_workspace_marker(workdir)
        if found_marker is None:
            raise ValueError(f"Managed workspace {workdir} is missing ownership marker; sync it first")
        _validate_workspace_owner(workdir, found_marker, {
            "project_id": req_to_save["project_id"],
            "workspace_id": req_to_save["workspace_id"],
            "root_commit": req_to_save.get("root_commit", ""),
            "managed": True,
        })
        if found_marker.get("sync_completed") is not True:
            raise ValueError(
                f"Managed workspace {workdir} has not completed a sync (sync_completed=false); run `remote-run sync` first"
            )
    if req_to_save["locks_dir"]:
        validate_locks_dir(root, Path(req_to_save["locks_dir"]), workdir)

    tmux_bin = trusted_tool("tmux", trusted_path)
    bash_bin = trusted_tool("bash", trusted_path)
    if not request.get("adhoc_existing_dir") and not request.get("require_workdir") and not req_to_save.get("project_id"):
        workdir.mkdir(parents=True, exist_ok=True)
    runs_dir = root / "runs"
    private_directory(runs_dir)
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=runs_dir))
    try:
        (pending / "artifacts").mkdir()
        write_json(pending / RUN_MARKER, {"schema_version": SCHEMA_VERSION, "run_id": job.name})
        write_json(pending / "request.json", req_to_save)
        write_json(pending / "state.json", {
            "run_id": job.name, "status": "queued", "waiting_for_lock": False, "submitted_at": now(),
        })
        write_private_text(pending / "worker.py", worker_src)
        pending.rename(job)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    t_sess = session_name(root, job.name)
    try:
        # Use isolated Python (-I -B) and trusted bash (--noprofile --norc) with BASH_ENV unset.
        args = [os.path.realpath(sys.executable), "-I", "-B", str(job / "worker.py"), "_worker", str(root), str(job)]
        cmd = "exec " + shlex.join(args) + " > " + shlex.quote(str(job / "run.log")) + " 2>&1"
        tmux_env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
        tmux_env["PATH"] = trusted_path
        run_cmd(
            [tmux_bin, "new-session", "-d", "-s", t_sess, "-c", str(workdir), "--", bash_bin, "--noprofile", "--norc", "-c", cmd],
            env=tmux_env,
        )
    except Exception as error:
        # Reconcile before marking failed: if tmux/worker actually started despite error/timeout, do NOT overwrite!
        has_tmux = subprocess.run([tmux_bin, "has-session", "-t", "=" + t_sess], capture_output=True, timeout=5).returncode == 0
        with open_lock(job / "launch.lock") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            cur = read_json(job / "state.json")
            if not has_tmux and cur.get("status") not in TERMINAL and not run_process_alive(cur):
                cur.update(status="failed", error=str(error), finished_at=now())
                write_json(job / "state.json", cur)
        raise
    return dict(state_of(job, root, trusted_path), reused=False)


def _has_other_pgid_members(pgid: int, exclude_pid: int) -> bool:
    """Return True if any live (non-zombie) process OR thread other than exclude_pid belongs to process group pgid."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return False
    target = str(pgid)
    for entry in proc_root.iterdir():
        name = entry.name
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == exclude_pid:
            continue
        try:
            fields = _read_proc_stat_fields(entry / "stat")
            # fields[0] is state ('Z' = zombie, 'X'/'x' = dead), fields[2] is pgrp
            if len(fields) < 3 or fields[2] != target:
                continue
            if fields[0] not in ("Z", "X", "x"):
                return True
            # Multi-threaded process whose thread-group leader exited into zombie state while
            # sibling threads in /proc/<pid>/task/* are still alive!
            task_dir = entry / "task"
            if task_dir.is_dir():
                for tid_entry in task_dir.iterdir():
                    if not tid_entry.name.isdigit():
                        continue
                    try:
                        t_fields = _read_proc_stat_fields(tid_entry / "stat")
                        if t_fields and t_fields[0] not in ("Z", "X", "x"):
                            return True
                    except (OSError, IndexError, UnicodeError):
                        continue
        except (OSError, IndexError, UnicodeError):
            continue
    return False


def _kill_pinned_pgid(pgid: int, leader_pid: int, *, include_leader: bool = False) -> bool:
    """Signal a process group whose leader is still un-reaped (PID/PGID cannot be reused)."""
    exclude = -1 if include_leader else leader_pid
    with suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline and _has_other_pgid_members(pgid, exclude):
        time.sleep(0.05)
    if _has_other_pgid_members(pgid, exclude):
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        kill_deadline = time.monotonic() + 1.5
        while time.monotonic() < kill_deadline and _has_other_pgid_members(pgid, exclude):
            time.sleep(0.05)
    return not _has_other_pgid_members(pgid, exclude)


def _pgid_any_alive(pgid: int) -> bool:
    if type(pgid) is not int or pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True


def kill_pgid(pgid: int, expected) -> bool:
    """Signal verified process group and return True once no members remain alive."""
    if not owns_process(pgid, expected, group=True):
        return not _pgid_any_alive(pgid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline and _pgid_any_alive(pgid):
        time.sleep(0.1)
    if _pgid_any_alive(pgid) and owns_process(pgid, expected, group=True):
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        kill_deadline = time.monotonic() + 1.5
        while time.monotonic() < kill_deadline and _pgid_any_alive(pgid):
            time.sleep(0.05)
    return not _pgid_any_alive(pgid)


def remote_cancel(root: Path, job: Path, trusted_path: str) -> dict:
    state = state_of(job, root, trusted_path)
    if state["status"] in TERMINAL:
        return state
    with open_lock(job / "launch.lock") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest = read_json(job / "state.json")
        if latest["status"] in TERMINAL:
            return state_of(job, root, trusted_path)
        write_json(job / "cancel.request", {"requested_at": now()})
        pgid = latest.get("child_pgid")
        wpid = latest.get("worker_pid")
        if owns_process(pgid, latest.get("child_identity"), group=True):
            kill_pgid(pgid, latest["child_identity"])
        elif owns_process(wpid, latest.get("worker_identity")):
            with suppress(ProcessLookupError):
                os.kill(wpid, signal.SIGTERM)
    tmux_bin = shutil.which("tmux", path=trusted_path)
    if tmux_bin:
        subprocess.run(
            [tmux_bin, "kill-session", "-t", "=" + session_name(root, job.name)],
            capture_output=True, check=False, timeout=5,
        )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        latest = read_json(job / "state.json")
        if latest["status"] in TERMINAL or not run_process_alive(latest):
            break
        time.sleep(0.05)
    if latest["status"] not in TERMINAL and not run_process_alive(latest):
        latest.update(status="cancelled", finished_at=now())
        write_json(job / "state.json", latest)
    return {**state_of(job, root, trusted_path), "cancellation_requested": True}


# === Component 2 (Remote): Type-Neutral Pull Selection & Prune ===

def eligible_pull_files(
    job: Path,
    workdir: str,
    *,
    shared_dir: str = "",
    max_file_mib: int = 10,
    exclude_patterns: list[str] = (),
    explicit_artifacts: list[str] = (),
    extra_paths: list[str] = (),
    paths_only: bool = False,
    remote_root: Path | None = None,
    locks_dir: Path | None = None,
) -> tuple[list[str], list[str], list[dict]]:
    if isinstance(max_file_mib, bool) or not isinstance(max_file_mib, int) or max_file_mib < 0:
        raise ValueError("max_file_mib must be a non-negative integer")
    if paths_only and not extra_paths:
        raise ValueError("--paths-only requires at least one --path")
    if paths_only and explicit_artifacts:
        raise ValueError("--paths-only cannot be combined with --artifact")
    validated_excludes = [validate_pattern(p, "pull.exclude") for p in exclude_patterns]

    job_files: list[str] = []
    skipped: list[dict] = []
    if not paths_only:
        job_files = [n for n in ("state.json", "run.log") if not (job / n).is_symlink() and (job / n).is_file()]
        art_root = job / "artifacts"
        if explicit_artifacts:
            if not art_root.is_dir() or art_root.is_symlink():
                raise ValueError(f"Run has no artifacts directory: {job.name}")
            art_resolved = art_root.resolve()
            seen_art: set[str] = set()
            for raw_art in explicit_artifacts:
                rel_art = rel_path(raw_art, "artifact")
                target = art_root / rel_art
                if target.is_symlink() or not target.is_file():
                    raise ValueError(f"Requested artifact missing or not a regular file: {raw_art!r}")
                if not target.resolve().is_relative_to(art_resolved):
                    raise ValueError(f"Requested artifact escapes artifacts directory: {raw_art!r}")
                rel_job = f"artifacts/{rel_art}"
                if rel_job not in seen_art:
                    seen_art.add(rel_job)
                    job_files.append(rel_job)
        elif art_root.is_dir() and not art_root.is_symlink():
            art_resolved = art_root.resolve()
            limit_bytes = max_file_mib * 1024 * 1024
            for item in sorted(art_root.rglob("*")):
                if item.is_symlink() or not item.is_file():
                    continue
                if not item.resolve().is_relative_to(art_resolved):
                    continue
                rel_art = item.relative_to(art_root).as_posix()
                rel_job = f"artifacts/{rel_art}"
                size = item.stat().st_size
                matched_pat = next((pat for pat in validated_excludes if match_pattern(rel_art, pat)), None)
                if matched_pat is not None:
                    skipped.append({"path": rel_job, "size_bytes": size, "reason": f"excluded:{matched_pat}"})
                    continue
                if limit_bytes > 0 and size > limit_bytes:
                    skipped.append({"path": rel_job, "size_bytes": size, "reason": f"exceeds_{max_file_mib}mib"})
                    continue
                job_files.append(rel_job)

    workdir_files: list[str] = []
    if extra_paths:
        wroot_raw = Path(workdir)
        check_directory_chain(wroot_raw, "Pull workspace root", trusted=False)
        wroot = wroot_raw.resolve()
        sroot = None
        if shared_dir:
            sroot_raw = Path(shared_dir)
            if sroot_raw.exists() or sroot_raw.is_symlink():
                check_directory_chain(sroot_raw, "Pull shared root", trusted=False)
                sroot = sroot_raw.resolve()
        registered_shares: dict[str, str] = {}
        marker_data = _read_workspace_marker(wroot)
        if marker_data is not None:
            registered_shares = _validated_registered_shares(marker_data.get("shared_links", {}), str(sroot) if sroot else "")

        remote_controls = [
            (remote_root / name).resolve()
            for name in ("runs", "sessions", "locks", "retired-runs", "workspace-locks", "pull-stages", "submit.lock", ROOT_MARKER)
        ] if remote_root else []
        if locks_dir is not None:
            remote_controls.append(locks_dir.resolve())

        def private_path(path: Path) -> bool:
            return any(path.is_relative_to(control_path) for control_path in remote_controls)

        def walk_pinned_dir(base_dir: Path, rel_prefix: str, allowed_root: Path) -> None:
            dir_fd = os.open(base_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                stack_dirs: list[tuple[int, str]] = [(dir_fd, rel_prefix)]
                while stack_dirs:
                    cur_fd, cur_rel = stack_dirs.pop()
                    try:
                        entries = sorted(os.listdir(cur_fd))
                        sub_dirs: list[tuple[int, str]] = []
                        for name in entries:
                            st_ent = os.stat(name, dir_fd=cur_fd, follow_symlinks=False)
                            child_rel = f"{cur_rel}/{name}" if cur_rel else name
                            if stat.S_ISLNK(st_ent.st_mode):
                                continue
                            if stat.S_ISDIR(st_ent.st_mode):
                                sub_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
                                sub_dirs.append((sub_fd, child_rel))
                            elif stat.S_ISREG(st_ent.st_mode):
                                candidate_res = (allowed_root / child_rel).resolve() if allowed_root == wroot else (base_dir / Path(child_rel).relative_to(rel_prefix)).resolve()
                                if not private_path(candidate_res) and candidate_res.is_relative_to(allowed_root):
                                    workdir_files.append(child_rel)
                        for pair in reversed(sub_dirs):
                            stack_dirs.append(pair)
                    finally:
                        os.close(cur_fd)
            except BaseException:
                raise

        for raw_rel in extra_paths:
            rel = rel_path(raw_rel, "pull path")
            # Check if rel enters a registered workspace share link (pinning to shared root without following arbitrary symlinks).
            matched_share = next(
                (s_rel for s_rel in registered_shares if rel == s_rel or rel.startswith(s_rel + "/")),
                None,
            )
            if matched_share is not None and sroot is not None:
                # Verify the share link itself is still the registered symlink.
                with workspace_file(wroot, matched_share) as (p_fd, leaf):
                    st_lnk = os.stat(leaf, dir_fd=p_fd, follow_symlinks=False)
                    if not stat.S_ISLNK(st_lnk.st_mode) or os.readlink(leaf, dir_fd=p_fd) != registered_shares[matched_share]:
                        raise ValueError(f"Unowned or modified shared link for pull path: {raw_rel!r}")
                sub_rel = rel[len(matched_share):].lstrip("/")
                base_target = (sroot / matched_share / sub_rel) if sub_rel else (sroot / matched_share)
                # Ensure no symlink hops under sroot.
                with workspace_file(sroot, f"{matched_share}/{sub_rel}" if sub_rel else matched_share) as (s_fd, s_leaf):
                    st_t = os.stat(s_leaf, dir_fd=s_fd, follow_symlinks=False)
                    if stat.S_ISLNK(st_t.st_mode):
                        raise ValueError(f"Requested workspace path cannot be a symlink: {raw_rel!r}")
                if not base_target.exists():
                    raise ValueError(f"Requested workspace path missing: {raw_rel!r}")
                resolved = base_target.resolve()
                if private_path(resolved) or not resolved.is_relative_to(sroot):
                    raise ValueError(f"Path escapes workspace shared root: {raw_rel!r}")
                if stat.S_ISDIR(st_t.st_mode):
                    walk_pinned_dir(base_target, rel, sroot)
                elif stat.S_ISREG(st_t.st_mode):
                    workdir_files.append(rel)
            else:
                try:
                    with workspace_file(wroot, rel) as (p_fd, leaf):
                        st_t = os.stat(leaf, dir_fd=p_fd, follow_symlinks=False)
                except FileNotFoundError as exc:
                    raise ValueError(f"Requested workspace path missing: {raw_rel!r}") from exc
                if stat.S_ISLNK(st_t.st_mode):
                    raise ValueError(f"Requested workspace path cannot be an arbitrary symlink: {raw_rel!r}")
                target = wroot / rel
                resolved = target.resolve()
                if private_path(resolved) or not resolved.is_relative_to(wroot):
                    raise ValueError(f"Path escapes workspace: {raw_rel!r}")
                if stat.S_ISDIR(st_t.st_mode):
                    walk_pinned_dir(target, rel, wroot)
                elif stat.S_ISREG(st_t.st_mode):
                    workdir_files.append(rel)
    return job_files, workdir_files, skipped


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            with suppress(OSError):
                total += item.stat().st_size
    return total


def prune_runs(root: Path, host_name: str, days: float, apply_del: bool, trusted_path: str, *, verbose: bool = False) -> dict:
    cutoff = time.time() - max(0.0, days) * 86400.0
    runs_dir = root / "runs"
    pruned: list[dict] = []
    skipped_unmanaged: list[dict] = []
    skipped_by_reason: dict[str, int] = {}
    reclaimed_bytes = 0

    def record_skip(path: Path, reason: str) -> None:
        skipped_unmanaged.append({"path": str(path), "reason": reason})
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1

    if private_directory(runs_dir, create=False):
        tmux_bin = shutil.which("tmux", path=trusted_path)
        live_sessions: set[str] = set()
        if tmux_bin:
            tsess_proc = subprocess.run(
                [tmux_bin, "list-sessions", "-F", "#{session_name}"],
                capture_output=True, text=True, timeout=5,
            )
            if tsess_proc.returncode == 0 and isinstance(tsess_proc.stdout, str):
                live_sessions = set(tsess_proc.stdout.splitlines())
        for child in sorted(runs_dir.iterdir()):
            try:
                original = child.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(original.st_mode) or original.st_uid != os.geteuid():
                record_skip(child, "symlink_or_non_dir")
                continue
            mark_p = child / RUN_MARKER
            try:
                mark_info = mark_p.lstat()
                marked = (
                    stat.S_ISREG(mark_info.st_mode)
                    and mark_info.st_uid == os.geteuid()
                    and mark_info.st_nlink == 1
                    and not (mark_info.st_mode & 0o022)
                    and read_json(mark_p) == {"schema_version": SCHEMA_VERSION, "run_id": child.name}
                )
            except (ValueError, OSError):
                marked = False
            if not marked:
                record_skip(child, "unrecognized_metadata")
                continue
            req_p, st_p = child / "request.json", child / "state.json"
            if req_p.is_symlink() or st_p.is_symlink() or not (req_p.is_file() and st_p.is_file()):
                record_skip(child, "missing_metadata")
                continue
            if session_name(root, child.name) in live_sessions:
                continue
            raw_st, raw_req = {}, {}
            with suppress(ValueError, KeyError, OSError):
                loaded_st = read_json(st_p)
                if isinstance(loaded_st, dict):
                    raw_st = loaded_st
            with suppress(ValueError, KeyError, OSError):
                loaded_req = read_json(req_p)
                if isinstance(loaded_req, dict):
                    raw_req = loaded_req
            if raw_st.get("run_id") != child.name or not isinstance(raw_req.get("host"), str) or not raw_req["host"]:
                record_skip(child, "unrecognized_metadata")
                continue
            if single_pid_alive(raw_st.get("worker_pid")) or pid_alive(raw_st.get("child_pgid")):
                continue
            mtime = original.st_mtime
            if mtime > cutoff:
                continue
            raw_status = raw_st.get("status")
            if isinstance(raw_status, str) and raw_status in TERMINAL:
                status_label = raw_status
            elif isinstance(raw_status, str) and raw_status:
                if (time.time() - mtime) < 5.0:
                    continue
                try:
                    st = state_of(child, root, trusted_path)
                    status_label = st["status"]
                    if not isinstance(status_label, str) or status_label not in TERMINAL:
                        continue
                except (ValueError, KeyError, TypeError, OSError):
                    record_skip(child, "unconfirmed_state")
                    continue
            else:
                record_skip(child, "unrecognized_metadata")
                continue
            sz = dir_size_bytes(child)
            if apply_del:
                retired = root / "retired-runs"
                with open_lock(root / "submit.lock") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fresh = child.lstat()
                    except (BlockingIOError, FileNotFoundError):
                        record_skip(child, "busy_or_missing")
                        continue
                    if (fresh.st_dev, fresh.st_ino, fresh.st_ctime_ns) != (
                        original.st_dev, original.st_ino, original.st_ctime_ns,
                    ):
                        record_skip(child, "changed_during_prune")
                        continue
                    if raw_st.get("status") not in TERMINAL:
                        try:
                            if state_of(child, root, trusted_path)["status"] not in TERMINAL:
                                continue
                        except (ValueError, KeyError, OSError):
                            continue
                    private_directory(retired)
                    removed = retired / uuid4().hex
                    child.rename(removed)
                shutil.rmtree(removed)
            pruned.append({"run_id": child.name, "status": status_label, "bytes": sz})
            reclaimed_bytes += sz

    workspaces = []
    proj_dir = root / "projects"
    if proj_dir.is_dir():
        for ws in sorted(proj_dir.glob("*/workspaces/*")):
            if ws.is_dir() and not ws.is_symlink():
                workspaces.append({"path": str(ws)})

    sample_limit = len(skipped_unmanaged) if verbose else 20
    return {
        "host": host_name,
        "days": days,
        "applied": apply_del,
        "runs_count": len(pruned),
        "reclaimed_bytes": reclaimed_bytes,
        "runs": pruned,
        "skipped_unmanaged_count": len(skipped_unmanaged),
        "skipped_by_reason": skipped_by_reason,
        "skipped_unmanaged": skipped_unmanaged[:sample_limit],
        "skipped_unmanaged_truncated": len(skipped_unmanaged) > sample_limit,
        "workspaces": workspaces,
    }


# === Component 3 (Remote): Opt-In Persistent Session & Worker Runtime ===

def session_file(root: Path, session_key: str) -> Path:
    return root / "sessions" / f"{session_key}.json"


def load_session(root: Path, session_key: str, fallback_cwd: str) -> tuple[str, dict[str, str], set[str]]:
    if not session_key:
        return fallback_cwd, {}, set()
    path = session_file(root, session_key)
    if path.is_symlink():
        raise ValueError("Symlink session metadata")
    if not path.is_file():
        return fallback_cwd, {}, set()
    data = read_json(path)
    cwd = data.get("cwd", fallback_cwd)
    if not Path(cwd).is_dir():
        cwd = fallback_cwd
    raw_env = {
        k: v for k, v in dict(data.get("env", {})).items()
        if k not in SKIP_SESSION_ENV and not k.startswith(SKIP_SESSION_PREFIXES)
    }
    raw_unset = {
        k for k in set(data.get("unset", []))
        if k not in SKIP_SESSION_ENV and not k.startswith(SKIP_SESSION_PREFIXES)
    }
    return cwd, raw_env, raw_unset


def update_session(
    root: Path,
    session_key: str,
    session_name_user: str,
    workdir: str,
    cwd_out: Path,
    env_out: Path,
    baseline_env: dict[str, str],
    prev_session_env: dict[str, str],
    prev_unset: set[str],
) -> str | None:
    if not session_key or not cwd_out.is_file() or not env_out.is_file():
        return None
    new_cwd = os.fsdecode(cwd_out.read_bytes().removesuffix(b"\n"))
    if not new_cwd or not Path(new_cwd).is_dir():
        new_cwd = None
    env_bytes = env_out.read_bytes()
    after_env: dict[str, str] = {}
    for entry in env_bytes.split(b"\x00"):
        if b"=" in entry:
            k, v = entry.split(b"=", 1)
            after_env[os.fsdecode(k)] = os.fsdecode(v)
    if not after_env:
        return None
    meaningful = lambda key: key not in SKIP_SESSION_ENV and not key.startswith(SKIP_SESSION_PREFIXES)
    changed = {k: v for k, v in after_env.items() if meaningful(k) and baseline_env.get(k) != v}
    removed = {k for k in baseline_env.keys() - after_env.keys() if meaningful(k)}
    merged = {k: v for k, v in prev_session_env.items() if k not in removed}
    merged.update(changed)
    write_json(session_file(root, session_key), {
        "name": session_name_user,
        "workdir": workdir,
        "cwd": new_cwd or workdir,
        "env": merged,
        "unset": sorted((prev_unset | removed) - changed.keys()),
        "updated_at": now(),
    })
    return new_cwd


def worker(root: Path, job: Path) -> None:
    request = read_json(job / "request.json")
    state_path = job / "state.json"
    state = read_json(state_path)
    gpu_managed = bool(request.get("gpu_managed", False))
    lock_keys = list(request.get("gpu_lock_keys") or [])
    locks_dir = Path(request["locks_dir"]) if request.get("locks_dir") else None
    if lock_keys:
        if locks_dir is None:
            raise ValueError("GPU locks_dir missing for GPU job")
        private_directory(locks_dir, tighten=locks_dir == root / "locks")
    proc = None
    leader_reaped = False

    def forward_sig(signum, _frame):
        # Never reap inside signal handler, and verify identity before signaling so we never signal a reaped/reused PGID.
        if proc is not None and not leader_reaped and owns_process(proc.pid, state.get("child_identity"), group=True):
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        elif proc is None:
            raise SystemExit(143)

    signal.signal(signal.SIGTERM, forward_sig)
    signal.signal(signal.SIGHUP, forward_sig)

    try:
        with open_lock(job / "launch.lock") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            if (job / "cancel.request").exists():
                state.update(status="cancelled", waiting_for_lock=False)
                return
            state["worker_pid"] = os.getpid()
            state["worker_identity"] = process_identity(os.getpid())
            write_json(state_path, state)

        with ExitStack() as stack:
            session_key = request.get("session_key", "")
            if session_key:
                identifier(session_key, "session key")
                session_dir = root / "sessions"
                private_directory(session_dir)
                session_lock = stack.enter_context(open_lock(session_dir / f"{session_key}.lock"))
                fcntl.flock(session_lock, fcntl.LOCK_EX)
            held_locks: list[str] = []
            if lock_keys:
                state.update(waiting_for_lock=True, held_gpu_locks=[])
                write_json(state_path, state)
            for key in lock_keys:
                assert locks_dir is not None
                handle = stack.enter_context(open_lock(locks_dir / f"{key}.lock"))
                fcntl.flock(handle, fcntl.LOCK_EX)
                held_locks.append(key)
                state.update(held_gpu_locks=list(held_locks))
                write_json(state_path, state)
            if (job / "cancel.request").exists():
                state.update(waiting_for_lock=False, status="cancelled")
                write_json(state_path, state)
                return
            state.update(waiting_for_lock=False)
            write_json(state_path, state)

            workdir = Path(request["workdir"])
            default_cwd = (
                str(workdir / request["rel_subdir"])
                if (request.get("rel_subdir") and (workdir / request["rel_subdir"]).is_dir())
                else str(workdir)
            )
            cwd, session_env, session_unset = load_session(root, session_key, default_cwd)

            # Strict literal env assembly with reserved-key enforcement at worker boundary too:
            # SSH baseline -> config layers -> saved session deltas -> invocation --env -> tool-owned metadata & GPU mask
            config_env = check_env_map(
                request.get("config_env", {}), "config_env", reject_reserved=True, gpu_managed=gpu_managed,
            )
            invocation_env = check_env_map(
                request.get("invocation_env", {}), "invocation_env", reject_reserved=True, gpu_managed=gpu_managed,
            )
            env: dict[str, str] = {
                k: v for k, v in dict(request.get("ssh_base_env", {})).items()
                if k not in RESERVED_ENV_KEYS and not k.startswith("REMOTE_RUN_")
            }
            env.update(config_env)
            env.update(session_env)
            for key in session_unset:
                env.pop(key, None)
            env.update(invocation_env)
            env.pop("BASH_ENV", None)

            env["PATH"] = request["effective_path"]
            env["REMOTE_RUN_ID"] = job.name
            env["REMOTE_RUN_HOST"] = request["host"]
            env["REMOTE_RUN_WORKSPACE"] = str(workdir)
            env["REMOTE_RUN_ARTIFACTS"] = str(job / "artifacts")
            if gpu_managed:
                if request.get("gpu_uuids"):
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(request["gpu_uuids"])
                else:
                    env["CUDA_VISIBLE_DEVICES"] = ""

            trusted_path = request.get("trusted_path", "")
            bash_bin = trusted_tool("bash", trusted_path)
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
                launch_argv = [bash_bin, "--noprofile", "--norc", "-c", wrapper + request["shell"]]
            else:
                cmd_args = list(request["argv"])
                if not cmd_args:
                    raise ValueError("Request has neither 'shell' nor non-empty 'argv'")
                probe = probe_executable(cmd_args[0], env["PATH"], cwd=Path(cwd))
                if not probe["ok"] or not probe["resolved"]:
                    raise ValueError(
                        f"Executable {cmd_args[0]!r} not found or not executable ({probe['reason']})"
                    )
                # Preserve exact resolved executable path & original argv[1:] without mutating args.
                resolved_argv = [probe["resolved"], *cmd_args[1:]]
                if session_key:
                    launch_argv = [bash_bin, "--noprofile", "--norc", "-c", wrapper + '"$@"', "remote-run", *resolved_argv]
                else:
                    launch_argv = resolved_argv

            baseline_env = dict(env)
            # Keep launch + wait + PGID drain inside one try/finally INSIDE the ExitStack so even a
            # post-launch state-write error kills/drains the entire process group BEFORE releasing GPU locks!
            rc = None
            try:
                with open_lock(job / "launch.lock") as guard:
                    fcntl.flock(guard, fcntl.LOCK_EX)
                    if (job / "cancel.request").exists():
                        state.update(status="cancelled")
                        return
                    proc = subprocess.Popen(
                        launch_argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, start_new_session=True,
                    )
                    state.update(
                        status="running", started_at=now(), child_pgid=proc.pid,
                        child_identity=process_identity(proc.pid), cwd=cwd,
                    )
                    write_json(state_path, state)
                if hasattr(os, "waitid") and hasattr(os, "P_PID") and hasattr(os, "WNOWAIT"):
                    while True:
                        try:
                            os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
                            break
                        except InterruptedError:
                            continue
                    if _has_other_pgid_members(proc.pid, proc.pid):
                        if (job / "cancel.request").exists():
                            _kill_pinned_pgid(proc.pid, proc.pid)
                        else:
                            drain_deadline = time.monotonic() + 2.0
                            while time.monotonic() < drain_deadline and _has_other_pgid_members(proc.pid, proc.pid):
                                time.sleep(0.05)
                            if _has_other_pgid_members(proc.pid, proc.pid) and lock_keys:
                                _kill_pinned_pgid(proc.pid, proc.pid)
                    if lock_keys:
                        while _has_other_pgid_members(proc.pid, proc.pid):
                            with suppress(ProcessLookupError):
                                os.killpg(proc.pid, signal.SIGKILL)
                            time.sleep(0.1)
                leader_reaped = True
                rc = proc.wait()
            finally:
                if proc is not None and not leader_reaped:
                    # Leader (or a sibling thread inside leader) may still be alive and SIGTERM-resistant.
                    # Keep the leader UN-REAPED via waitid(WNOWAIT) while killing/draining the PGID so the
                    # kernel CANNOT recycle proc.pid as a new PGID before every member & thread is gone!
                    _kill_pinned_pgid(proc.pid, proc.pid, include_leader=True)
                    if lock_keys:
                        has_waitid = hasattr(os, "waitid") and hasattr(os, "P_PID") and hasattr(os, "WNOWAIT") and hasattr(os, "WNOHANG")
                        leader_exited = False
                        while True:
                            with suppress(ProcessLookupError):
                                os.killpg(proc.pid, signal.SIGKILL)
                            if has_waitid and not leader_exited:
                                try:
                                    res = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
                                    leader_exited = res is not None
                                except ChildProcessError:
                                    leader_exited = True
                                except InterruptedError:
                                    continue
                            elif not has_waitid:
                                leader_exited = not _has_other_pgid_members(proc.pid, -1)
                            if leader_exited and not _has_other_pgid_members(proc.pid, proc.pid):
                                break
                            time.sleep(0.05)
                        leader_reaped = True
                        with suppress(OSError):
                            proc.wait(timeout=5)
                    else:
                        with suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGKILL)
                        leader_reaped = True
                        with suppress(OSError):
                            proc.wait(timeout=5)
            if (job / "cancel.request").exists():
                state.update(exit_code=rc, status="cancelled")
            else:
                state.update(exit_code=rc, status="completed" if rc == 0 else "failed")
                if rc == 0 and session_key:
                    new_cwd = update_session(
                        root, session_key, request.get("session", ""), str(workdir),
                        cwd_out, env_out, baseline_env, session_env, session_unset,
                    )
                    if new_cwd:
                        state["cwd"] = new_cwd
    except SystemExit:
        if proc is not None:
            kill_pgid(proc.pid, state.get("child_identity"))
        state.update(waiting_for_lock=False, status="cancelled" if (job / "cancel.request").exists() else "interrupted")
    except Exception as error:
        if proc is not None:
            kill_pgid(proc.pid, state.get("child_identity"))
        state.update(waiting_for_lock=False, status="failed", error=str(error))
        raise
    finally:
        for tmp in (job / ".session_cwd", job / ".session_env"):
            tmp.unlink(missing_ok=True)
        state.pop("held_gpu_locks", None)
        state["finished_at"] = now()
        write_json(state_path, state)


# === RPC Control Plane (Versioned Envelope, Read-Only Doctor) ===

def reply_line(value: dict) -> str:
    envelope = {"protocol_version": PROTOCOL_VERSION, "data": value}
    return CONTROL_REPLY + json.dumps(envelope, allow_nan=False)


def parse_reply(output: str) -> dict:
    replies = [line[len(CONTROL_REPLY):] for line in output.splitlines() if line.startswith(CONTROL_REPLY)]
    if len(replies) != 1:
        raise ValueError(
            "Could not read the remote worker reply. Check SSH shell startup output and remote Python 3.11+."
        )
    envelope = json.loads(replies[0])
    if not isinstance(envelope, dict) or envelope.get("protocol_version") != PROTOCOL_VERSION or "data" not in envelope:
        raise ValueError("Incompatible remote worker protocol reply")
    return envelope["data"]


def _stage_pinned_regular_file(src_root: Path, rel: str, dst_root: Path, *, max_bytes: int = 0) -> None:
    """Copy a verified regular file using O_NOFOLLOW descriptor traversal into a private pull stage.
    If max_bytes > 0, enforce the byte cap directly on the pinned file descriptor (preventing post-stat growth).
    """
    with workspace_file(src_root, rel) as (parent_fd, leaf):
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        with os.fdopen(fd, "rb") as src_stream:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Pull path changed to non-regular file: {rel!r}")
            if max_bytes > 0 and info.st_size > max_bytes:
                raise ValueError(f"Artifact {rel!r} grew beyond max_file_mib cap ({max_bytes} bytes)")
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            with dst.open("wb") as out_stream:
                while True:
                    chunk = src_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if max_bytes > 0 and copied > max_bytes:
                        raise ValueError(f"Artifact {rel!r} exceeded max_file_mib cap during staging")
                    out_stream.write(chunk)
            with suppress(OSError):
                os.chmod(dst, stat.S_IMODE(info.st_mode))
                os.utime(dst, ns=(info.st_atime_ns, info.st_mtime_ns))


PULL_STAGE_MARKER = ".remote-run-pull-stage.json"


def _is_owned_pull_stage(child: Path) -> bool:
    if not child.name.startswith("stage-"):
        return False
    try:
        st = child.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
            return False
        mp = child / PULL_STAGE_MARKER
        mst = mp.lstat()
        if not stat.S_ISREG(mst.st_mode) or mst.st_uid != os.geteuid() or mst.st_mode & 0o022 or mst.st_nlink != 1:
            return False
        return read_json(mp) == {"schema_version": SCHEMA_VERSION, "stage_id": child.name}
    except (ValueError, OSError):
        return False


def _cleanup_stale_pull_stages(root: Path, max_age_sec: float = 3600.0) -> None:
    stage_root = root / "pull-stages"
    if not stage_root.is_dir() or stage_root.is_symlink():
        return
    cutoff = time.time() - max_age_sec
    for child in list(stage_root.iterdir()):
        if not _is_owned_pull_stage(child):
            continue
        with suppress(OSError):
            if child.lstat().st_mtime < cutoff:
                shutil.rmtree(child)


def _create_pull_stage(
    root: Path,
    job: Path,
    wroot: Path,
    sroot: Path | None,
    job_files: list[str],
    ws_regular_files: list[str],
    shared_files: list[str],
    *,
    artifact_max_bytes: int = 0,
    explicit_artifacts: bool = False,
) -> tuple[str, list[str]]:
    """Stage all validated regular files from pinned descriptors into a private 0700 stage dir."""
    stage_root = root / "pull-stages"
    private_directory(stage_root)
    _cleanup_stale_pull_stages(root)
    stage_dir = Path(tempfile.mkdtemp(prefix="stage-", dir=stage_root))
    staged_rels: list[str] = []
    try:
        os.chmod(stage_dir, 0o700)
        write_json(stage_dir / PULL_STAGE_MARKER, {"schema_version": SCHEMA_VERSION, "stage_id": stage_dir.name})
        for rel in job_files:
            cap = artifact_max_bytes if (rel.startswith("artifacts/") and not explicit_artifacts) else 0
            _stage_pinned_regular_file(job, rel, stage_dir, max_bytes=cap)
            staged_rels.append(rel)
        ws_dst = stage_dir / "workspace"
        for rel in ws_regular_files:
            _stage_pinned_regular_file(wroot, rel, ws_dst)
            staged_rels.append(f"workspace/{rel}")
        if shared_files:
            if sroot is None:
                raise ValueError("Missing shared_dir for staged shared pull")
            for rel in shared_files:
                _stage_pinned_regular_file(sroot, rel, ws_dst)
                staged_rels.append(f"workspace/{rel}")
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    return stage_dir.name, staged_rels


def _remove_pull_stage(root: Path, stage_id: str) -> dict:
    if not isinstance(stage_id, str) or not re.fullmatch(r"stage-[A-Za-z0-9_.-]+", stage_id):
        raise ValueError(f"Invalid pull stage_id: {stage_id!r}")
    stage_root = root / "pull-stages"
    if not private_directory(stage_root, create=False):
        return {"cleaned": False}
    target = stage_root / stage_id
    if not _is_owned_pull_stage(target):
        return {"cleaned": False}
    shutil.rmtree(target)
    return {"cleaned": True}


def control(payload: dict) -> dict | None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {payload.get('protocol_version')!r}")
    host_cfg = payload["host"]
    action = payload["action"]
    ssh_base_env = dict(os.environ)
    ssh_base_path = ssh_base_env.get("PATH", "")
    trusted_path = build_trusted_host_path(ssh_base_path, list(host_cfg.get("path", [])))
    root = Path(expand_remote_path(host_cfg["root"], "root"))
    gpu_cfg = host_cfg.get("gpu") or {}
    locks_dir = (
        Path(expand_remote_path(gpu_cfg.get("locks_dir") or join_remote_path(host_cfg["root"], "locks"), "gpu.locks_dir"))
        if gpu_cfg else None
    )

    if action == "workspace_lease":
        ensure_root_owned(root)
        wdir = Path(expand_remote_path(payload["workdir"], "workdir")).resolve()
        validate_remote_workdir(root, wdir)
        lock_dir = root / "workspace-locks"
        private_directory(lock_dir)
        with ExitStack() as held:
            for ancestor in (*reversed(wdir.parents), wdir):
                name = sha256(os.fsencode(ancestor)).hexdigest() + ".lock"
                lock = held.enter_context(open_lock(lock_dir / name))
                mode = fcntl.LOCK_EX if ancestor == wdir else fcntl.LOCK_SH
                while True:
                    try:
                        fcntl.flock(lock, mode | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            return None
            print(reply_line({"locked": True}), flush=True)
            sys.stdin.read(1)
        return None

    if action == "probe_init":
        validate_remote_root(root)
        tools = {t: shutil.which(t, path=trusted_path) for t in ("bash", "tmux", "rsync")}
        missing_tools = [t for t, found in tools.items() if not found]
        gpus = []
        if shutil.which("nvidia-smi", path=trusted_path):
            with suppress(RuntimeError, ValueError, OSError):
                gpus = query_nvidia_gpus(trusted_path)
        return {
            "status": "ok",
            "tools": tools,
            "missing_tools": missing_tools,
            "gpus": gpus,
            "resolved_root": str(root),
            "python_version": sys.version.split()[0],
        }

    if action == "doctor":
        # Strictly read-only, zero arbitrary code execution, zero filesystem mutations.
        tools = {t: shutil.which(t, path=trusted_path) for t in ("bash", "tmux", "rsync")}
        missing_tools = [t for t, found in tools.items() if not found]
        root_state = ensure_root_owned(root, dry_run=True)
        workdir_raw = payload.get("workdir")
        workdir = Path(expand_remote_path(workdir_raw, "workdir")) if workdir_raw else None
        if locks_dir is not None:
            validate_locks_dir(root, locks_dir, workdir)

        workspace_state = "none"
        if workdir is not None:
            validate_remote_workdir(root, workdir)
            check_directory_chain(workdir, "Remote workdir", trusted=False)
            marker = workdir / WORKSPACE_MARKER
            if marker.exists() or marker.is_symlink():
                _read_workspace_marker(workdir)
                workspace_state = "ready"
            elif workdir.exists():
                if not workdir.is_dir():
                    raise ValueError(f"Remote workdir exists and is not a directory: {workdir}")
                workspace_state = "unmanaged_nonempty" if any(workdir.iterdir()) else "pending"
            else:
                workspace_state = "pending"

        effective_path, path_report = resolve_path_layers(
            payload.get("path_layers", []), ssh_base_path, workdir=workdir, strict=False,
        )
        bad_paths = [p for p in path_report if p["source"] != "ssh_base" and p["status"] in ("missing", "unusable")]

        cmd_cwd = None
        if workdir is not None:
            rel_sub = payload.get("rel_subdir", "")
            cand = (workdir / rel_sub) if rel_sub else workdir
            cmd_cwd = cand if cand.is_dir() else workdir
            # If an opt-in session is specified on doctor, match worker's saved session cwd!
            sess_name = payload.get("session", "")
            if sess_name:
                s_key = session_key_for(str(workdir), identifier(sess_name, "session"))
                sess_cwd, _, _ = load_session(root, s_key, str(cmd_cwd))
                cmd_cwd = Path(sess_cwd)
        cmd_results = [
            probe_executable(spec, effective_path, cwd=cmd_cwd)
            for spec in payload.get("check_cmds", [])
        ]
        failed_cmds = [r["input"] for r in cmd_results if not r["ok"]]

        gpu_report = None
        gpu_error = None
        if gpu_cfg:
            configured = validate_gpu_indices(gpu_cfg.get("devices", []), "host.gpu.devices")
            try:
                observed = query_nvidia_gpus(trusted_path)
                by_idx = {g["index"]: g for g in observed}
                missing_idx = [i for i in configured if i not in by_idx]
                gpu_report = {
                    "configured_devices": configured,
                    "observed": observed,
                    "missing_devices": missing_idx,
                    "locks_dir": str(locks_dir),
                }
                if missing_idx:
                    gpu_error = f"Configured GPU index(es) not found: {missing_idx}"
            except (ValueError, RuntimeError, OSError) as exc:
                gpu_error = str(exc)

        ok = not missing_tools and not bad_paths and not failed_cmds and not gpu_error and workspace_state != "unmanaged_nonempty"
        result = {
            "status": "ready" if ok else "checks_failed",
            "host": host_cfg["name"],
            "root": str(root),
            "root_state": root_state,
            "workspace": str(workdir) if workdir else "",
            "workspace_state": workspace_state,
            "tools": tools,
            "missing_tools": missing_tools,
            "rsync": remote_rsync_info(trusted_path),
            "effective_path": effective_path,
            "path_layers": path_report,
            "commands": cmd_results,
            "failed_commands": failed_cmds,
            "gpu": gpu_report,
        }
        if gpu_error:
            result["gpu_error"] = gpu_error
        if not ok:
            result["_exec_failed"] = True
        return result

    if action == "prepare_workspace" and payload.get("dry_run"):
        info = prepare_remote_workspace(
            root, payload["workspace"], payload.get("shared_dirs", []),
            trusted_path=trusted_path, check_run_id=payload.get("run_id", ""),
            dry_run=True, for_sync=bool(payload.get("for_sync", True)),
            create_if_missing=bool(payload.get("create_if_missing", True)),
            manifest_parents=list(payload.get("manifest_parents") or ()),
            locks_dir=locks_dir,
        )
        info["rsync"] = remote_rsync_info(trusted_path)
        return info

    if action == "reconcile_synced" and payload.get("dry_run"):
        ensure_root_owned(root, dry_run=True)
        return reconcile_synced_manifest(
            root, payload["workdir"], {}, payload["prune_candidates"],
            payload["protected_prefixes"], dry_run=True,
        )

    if action == "prune" and not payload.get("apply", False):
        ensure_root_owned(root, dry_run=True)
        return prune_runs(
            root, host_cfg["name"], float(payload.get("days", 14.0)), False,
            trusted_path, verbose=bool(payload.get("verbose", False)),
        )

    read_only = action in ("status", "list") or (action == "session" and payload.get("subaction") == "list")
    ensure_root_owned(root, dry_run=read_only)
    for control_dir in ("runs", "locks", "sessions", "workspace-locks", "pull-stages"):
        private_directory(root / control_dir, create=False)

    if action == "pull_cleanup":
        return _remove_pull_stage(root, payload["stage_id"])

    if action == "prune":
        return prune_runs(
            root, host_cfg["name"], float(payload.get("days", 14.0)), True,
            trusted_path, verbose=bool(payload.get("verbose", False)),
        )

    if action == "session":
        sdir = root / "sessions"
        sub = payload["subaction"]
        wdir_filter = expand_remote_path(payload["workdir"], "workdir") if payload.get("workdir") else ""
        if sub == "list":
            items = []
            if sdir.is_dir():
                for p in sorted(sdir.glob("*.json")):
                    try:
                        if p.is_symlink():
                            raise ValueError("Symlink session metadata")
                        data = read_json(p)
                        if not isinstance(data, dict):
                            raise ValueError("Corrupt session metadata")
                    except (ValueError, KeyError, OSError) as exc:
                        items.append({"session_key": p.stem, "status": "corrupt", "error": str(exc)})
                        continue
                    if wdir_filter and data.get("workdir") != wdir_filter:
                        continue
                    items.append({"session_key": p.stem, **data, "env": {key: "<redacted>" for key in data.get("env", {})}})
            return {"sessions": items}
        if sub == "reset":
            key = session_key_for(wdir_filter, payload["name"])
            private_directory(sdir)
            with open_lock(sdir / f"{key}.lock") as slock:
                try:
                    fcntl.flock(slock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError(f"Session {payload['name']!r} is active. Retry after it finishes.") from exc
                target = sdir / f"{key}.json"
                existed = target.is_file()
                target.unlink(missing_ok=True)
            return {"reset": True, "name": payload["name"], "existed": existed, "workdir": wdir_filter}
        raise ValueError(f"Unknown session subaction: {sub}")

    if action == "status":
        wait_sec = float(payload.get("wait_sec", 0.0))
        if not math.isfinite(wait_sec) or wait_sec < 0:
            raise ValueError("wait_sec must be a finite non-negative number")
        job = root / "runs" / identifier(payload["run_id"], "run_id")
        return wait_job_terminal(
            job, root, wait_sec, trusted_path,
            include_log=payload.get("include_log", True),
            offset=payload.get("offset"),
            tail_bytes=payload.get("tail_bytes", 8192),
            raw_log=payload.get("raw_log", False),
            wake_on_output=bool(payload.get("wake_on_output", False)),
            binary_log=bool(payload.get("binary_log", False)),
        )

    if action == "submit":
        wait_sec = float(payload.get("wait_sec", 0.0))
        if not math.isfinite(wait_sec) or wait_sec < 0:
            raise ValueError("wait_sec must be a finite non-negative number")
        job = root / "runs" / identifier(payload["run_id"], "run_id")
        with open_lock(root / "submit.lock") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = remote_submit(
                root, job, host_cfg, payload["request"], payload["script"], trusted_path, ssh_base_env,
            )
        if wait_sec > 0:
            st = wait_job_terminal(
                job, root, wait_sec, trusted_path,
                include_log=True,
                tail_bytes=payload.get("tail_bytes", 8192),
                raw_log=payload.get("raw_log", False),
            )
            st["reused"] = receipt.get("reused", False)
            return st
        return receipt

    with ExitStack() as stack:
        if not read_only:
            lock = stack.enter_context(open_lock(root / "submit.lock"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        if action == "prepare_workspace":
            info = prepare_remote_workspace(
                root, payload["workspace"], payload.get("shared_dirs", []),
                trusted_path=trusted_path, check_run_id=payload.get("run_id", ""),
                for_sync=bool(payload.get("for_sync", True)),
                create_if_missing=bool(payload.get("create_if_missing", True)),
                manifest_parents=list(payload.get("manifest_parents") or ()),
                locks_dir=locks_dir,
            )
            if payload.get("for_sync", True):
                info["rsync"] = remote_rsync_info(trusted_path)
            return info
        if action == "reconcile_synced":
            return reconcile_synced_manifest(
                root,
                payload["workdir"],
                payload.get("uploaded_files", {}),
                payload.get("prune_candidates", []),
                payload.get("protected_prefixes", []),
                uploaded_symlinks=payload.get("uploaded_symlinks"),
            )
        if action == "list":
            runs_dir = root / "runs"
            items = []
            proj_filter = payload.get("project_id") or ""
            limit = int(payload.get("limit", 20))
            if runs_dir.is_dir():
                candidates = [c for c in runs_dir.iterdir() if c.is_dir()]
                candidates.sort(key=lambda c: c.stat().st_mtime, reverse=True)
                for child in candidates:
                    if not (child / "state.json").is_file() or not (child / "request.json").is_file():
                        continue
                    try:
                        st = state_of(child, root, trusted_path)
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

        job = root / "runs" / identifier(payload["run_id"], "run_id")
        if action == "cancel":
            return remote_cancel(root, job, trusted_path)
        state = state_of(job, root, trusted_path)
        if action == "pull":
            if state["status"] not in TERMINAL:
                raise ValueError(f"Run {job.name} is still {state['status']}; wait or cancel before pulling")
            req = read_json(job / "request.json")
            pull_policy = req.get("pull_policy") or {}
            eff_max_mib = payload["max_file_mib"] if payload.get("max_file_mib") is not None else pull_policy.get("max_file_mib", 10)
            eff_excludes = list(pull_policy.get("exclude", []))
            job_files, workdir_files, skipped = eligible_pull_files(
                job, state["workdir"],
                shared_dir=state.get("shared_dir", ""),
                max_file_mib=eff_max_mib,
                exclude_patterns=eff_excludes,
                explicit_artifacts=list(payload.get("artifacts") or ()),
                extra_paths=list(payload.get("paths") or ()),
                paths_only=bool(payload.get("paths_only", False)),
                remote_root=root,
                locks_dir=locks_dir,
            )
            # Split workdir_files into pinned regular workspace files vs registered shared-dir files
            # so rsync NEVER needs -k (--copy-dirlinks) and cannot follow replacement symlinks!
            ws_regular_files: list[str] = []
            shared_files: list[str] = []
            wroot = Path(state["workdir"])
            sroot_str = state.get("shared_dir", "")
            marker_data = _read_workspace_marker(wroot) if wroot.is_dir() else None
            reg_shares = _validated_registered_shares(marker_data.get("shared_links", {}), sroot_str) if marker_data else {}
            for wf in workdir_files:
                if any(wf == s or wf.startswith(s + "/") for s in reg_shares):
                    shared_files.append(wf)
                else:
                    ws_regular_files.append(wf)
            # Probe remote rsync BEFORE creating pull stage so a failing probe never orphans a stage.
            rsync_info = remote_rsync_info(trusted_path)
            explicit_arts = list(payload.get("artifacts") or ())
            stage_id, staged_files = _create_pull_stage(
                root, job, wroot, Path(sroot_str) if sroot_str else None,
                job_files, ws_regular_files, shared_files,
                artifact_max_bytes=eff_max_mib * 1024 * 1024 if eff_max_mib > 0 else 0,
                explicit_artifacts=bool(explicit_arts),
            )
            return {
                "status": state["status"],
                "root": str(root),
                "stage_id": stage_id,
                "stage_dir": str(root / "pull-stages" / stage_id),
                "staged_files": staged_files,
                "job_files": job_files,
                "workdir_files": workdir_files,
                "workdir": state["workdir"],
                "skipped_artifacts": skipped,
                "rsync": rsync_info,
            }
        raise ValueError(f"Unknown action: {action}")


if "PAYLOAD" in globals():
    try:
        response = control(PAYLOAD)
        if PAYLOAD.get("action") != "workspace_lease":
            print(reply_line(response), flush=True)
    except (ValueError, KeyError, OSError, RuntimeError, OverflowError, subprocess.TimeoutExpired) as error:
        print(f"remote-run: {error}", file=sys.stderr)
        raise SystemExit(1)
elif __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["_worker"] and len(sys.argv) == 4:
            worker(Path(sys.argv[2]), Path(sys.argv[3]))
        else:
            raise ValueError("remote_worker.py is an internal remote runtime entrypoint")
    except (ValueError, KeyError, OSError, RuntimeError, OverflowError, subprocess.TimeoutExpired) as error:
        print(f"remote-run: {error}", file=sys.stderr)
        raise SystemExit(1)
