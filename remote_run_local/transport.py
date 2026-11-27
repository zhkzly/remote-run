"""Component 1: OpenSSH transport, ControlMaster multiplexing, auth diagnostics, and worker RPC."""
from __future__ import annotations

from contextlib import contextmanager, suppress
import json
import os
from pathlib import Path
import select
import shlex
import socket
import subprocess
import tempfile
import time

import remote_worker
from remote_worker import PROTOCOL_VERSION, parse_reply, private_directory, reply_line

SSH_BASE = [
    "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=7",
    "-o", "ConnectionAttempts=1", "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
    "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
    "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
]


def local_cache_dir(name: str) -> Path:
    return Path.home().resolve() / ".cache" / "remote-run" / name


def build_ssh_argv(host: dict, *, allow_control_create: bool = True) -> list[str]:
    argv = list(SSH_BASE)
    if allow_control_create and os.environ.get("REMOTE_RUN_SSH_CONTROL", "1") != "0":
        ctl_dir = local_cache_dir("ssh")
        if len(os.fsencode(ctl_dir)) + 61 < 104:
            private_directory(ctl_dir)
            ctl_path = str(ctl_dir).replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
            argv.extend([
                "-o", "ControlMaster=auto",
                "-o", "ControlPersist=60",
                "-o", f'ControlPath="{ctl_path}/cm-%C"',
            ])
    if argv == SSH_BASE:
        argv.extend(["-o", "ControlMaster=no", "-o", "ControlPath=none"])
    if host.get("port"):
        argv.extend(["-p", str(int(host["port"]))])
    return argv


def _unix_socket_connectable(sock_path: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(sock_path)
        return True
    except OSError:
        return False


def _ssh_auth_hint(argv: list[str], returncode: int, detail: str) -> str:
    if not argv or Path(argv[0]).name != "ssh" or returncode != 255:
        return ""
    lower = detail.lower()
    if "permission denied" not in lower or "publickey" not in lower:
        return ""
    sock = os.environ.get("SSH_AUTH_SOCK", "")
    if not sock:
        return (
            " [Hint: SSH_AUTH_SOCK is unset in this process. If running via an MCP client such as Codex, "
            "add `env_vars = [\"SSH_AUTH_SOCK\"]` to `[mcp_servers.remote-run]` in ~/.codex/config.toml, "
            "or configure an unencrypted IdentityFile in ~/.ssh/config.]"
        )
    if not os.path.exists(sock) or not _unix_socket_connectable(sock):
        return (
            " [Hint: SSH_AUTH_SOCK points to a missing or unreachable socket. Refresh your ssh-agent or "
            "configure IdentityFile in ~/.ssh/config.]"
        )
    return ""


def run_local(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin: str | bytes | None = None,
    timeout: float = 30,
    trim: bool = True,
    lease=None,
) -> str:
    text_mode = not isinstance(stdin, bytes)
    if lease is None:
        with subprocess.Popen(
            argv, cwd=cwd, env=env, text=text_mode,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        ) as proc:
            try:
                out, errors = proc.communicate(input=stdin, timeout=timeout)
            except BrokenPipeError:
                out, errors = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
            completed = subprocess.CompletedProcess(argv, proc.returncode, out, errors)
    else:
        lease()
        deadline = time.monotonic() + timeout
        with subprocess.Popen(
            argv, cwd=cwd, env=env, text=text_mode,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        ) as proc:
            try:
                while True:
                    lease()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    try:
                        out, errors = proc.communicate(input=stdin, timeout=min(remaining, 0.2))
                        break
                    except BrokenPipeError:
                        stdin = None
                        out, errors = proc.communicate(timeout=max(0.1, deadline - time.monotonic()))
                        break
                    except subprocess.TimeoutExpired:
                        stdin = None
                lease()
            finally:
                if proc.poll() is None:
                    proc.kill()
            completed = subprocess.CompletedProcess(argv, proc.returncode, out, errors)
    if completed.returncode:
        err_s = completed.stderr if isinstance(completed.stderr, str) else completed.stderr.decode(errors="replace")
        out_s = completed.stdout if isinstance(completed.stdout, str) else completed.stdout.decode(errors="replace")
        detail = err_s.strip() or out_s.strip()
        hint = _ssh_auth_hint(argv, completed.returncode, detail)
        raise RuntimeError(f"{argv[0]} exited {completed.returncode}: {detail[-1500:]}{hint}")
    out_text = completed.stdout if isinstance(completed.stdout, str) else completed.stdout.decode(errors="replace")
    return out_text.strip() if trim else out_text


def worker_source() -> str:
    return Path(remote_worker.__file__).read_text(encoding="utf-8")


def control_command(host: dict, action: str, data: dict, *, allow_control_create: bool = True) -> tuple[list[str], str]:
    src = worker_source()
    host_wire = {key: host[key] for key in ("name", "root", "path", "gpu") if key in host}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "action": action,
        "script": src,
        "host": host_wire,
        **data,
    }
    boot = (
        "import json,sys;"
        "p=json.loads(sys.stdin.readline());"
        "exec(compile(p['script'],'remote_worker.py','exec'),{'PAYLOAD':p})"
    )
    # Use isolated Python (-I -B) on remote host so remote user site/PYTHONPATH cannot hijack bootstrap.
    cmd = shlex.join(["python3", "-I", "-B", "-c", boot])
    return [*build_ssh_argv(host, allow_control_create=allow_control_create), host["ssh"], cmd], json.dumps(payload, allow_nan=False) + "\n"


def ssh_call(
    host: dict,
    action: str,
    data: dict,
    timeout: float = 60,
    *,
    lease=None,
    read_only_transport: bool = False,
) -> dict:
    argv, payload = control_command(host, action, data, allow_control_create=not read_only_transport)
    return parse_reply(run_local(argv, stdin=payload, timeout=timeout, lease=lease))


@contextmanager
def workspace_lease(host: dict, workdir: str):
    argv, payload = control_command(host, "workspace_lease", {"workdir": workdir})
    with tempfile.TemporaryFile() as stderr, subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
    ) as process:
        try:
            assert process.stdin is not None and process.stdout is not None
            with suppress(BrokenPipeError):
                process.stdin.write(payload.encode("utf-8"))
                process.stdin.flush()
            pending = b""
            deadline = time.monotonic() + 60
            expected = reply_line({"locked": True}).encode("utf-8")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                    raise ValueError(f"Timed out waiting for remote workspace lock: {workdir}")
                data = os.read(process.stdout.fileno(), 8192)
                if not data:
                    stderr.seek(0, os.SEEK_END)
                    stderr.seek(max(0, stderr.tell() - 1500))
                    detail = stderr.read().decode(errors="replace").strip()
                    rc = process.poll()
                    hint = _ssh_auth_hint(argv, rc if rc is not None else 255, detail)
                    raise ValueError(f"Could not lock remote workspace: {detail}{hint}")
                pending += data
                if len(pending) > 1024 * 1024:
                    raise ValueError("Unexpected output while locking remote workspace")
                lines = pending.split(b"\n")
                pending = lines.pop()
                if any(line == expected for line in lines):
                    def verify_lease() -> None:
                        if process.poll() is not None:
                            raise RuntimeError(f"Remote workspace lock connection lost: {workdir}")

                    yield verify_lease
                    return
        finally:
            if process.stdin is not None:
                with suppress(BrokenPipeError):
                    process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
