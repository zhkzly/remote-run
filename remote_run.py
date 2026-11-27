#!/usr/bin/env python3
"""Universal cross-project remote Linux execution, safe manifest sync, and persistent sessions (v2)."""
from __future__ import annotations

import argparse
import base64
import codecs
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from remote_worker import DEFAULT_REMOTE_ROOT, identifier
from remote_run_local.config import default_hosts_config_path
from remote_run_local import service as svc


def parse_gpu_csv(raw: str) -> list[int]:
    if not raw or not raw.strip():
        raise ValueError("--gpus requires a comma-separated list of non-negative GPU indices (e.g. '0' or '0,1')")
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p.isdigit():
            raise ValueError(f"Invalid GPU index {p!r} in --gpus {raw!r}; expected non-negative integers")
        out.append(int(p))
    return out


def parse_env_items(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _run_service_rpc() -> int:
    """Internal typed-JSON entrypoint invoked by remote_run_mcp.py as a cancellable subprocess."""
    try:
        req = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid service RPC JSON: {exc}") from exc
    if not isinstance(req, dict):
        raise ValueError("Service RPC payload must be a JSON object")
    op = req.get("op")
    if not isinstance(op, str) or not op:
        raise ValueError("Service RPC payload requires string 'op'")
    params = req.get("params")
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        raise ValueError("Service RPC 'params' must be a JSON object")
    out = svc.dispatch_operation(op, params)
    exec_failed = isinstance(out, dict) and bool(out.pop("_exec_failed", False))
    print(json.dumps(out, ensure_ascii=False))
    return 2 if exec_failed else 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", type=Path, default=argparse.SUPPRESS,
        help=f"Host config path (default: $REMOTE_RUN_CONFIG or {default_hosts_config_path()})",
    )
    common.add_argument(
        "--host", default=argparse.SUPPRESS,
        help="Host alias from hosts.toml (default: binding or default_host)",
    )
    common.add_argument(
        "--pretty", action="store_true", default=argparse.SUPPRESS,
        help="Pretty-print JSON output",
    )

    parser = argparse.ArgumentParser(prog="remote-run", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("hosts", parents=[common], help="List configured remote hosts")

    init_p = sub.add_parser("init", parents=[common], help="Probe an SSH target and add it to hosts.toml")
    init_p.add_argument("target", help="SSH alias from ~/.ssh/config or user@host")
    init_p.add_argument("--name", default="", help="Host alias in hosts.toml (default: derived from target)")
    init_p.add_argument("--port", type=int, default=None, help="Optional SSH port (1-65535)")
    init_p.add_argument("--root", default="", help=f"Remote state root (default: {DEFAULT_REMOTE_ROOT})")
    init_p.add_argument("--default", action="store_true", help="Set as default_host in hosts.toml")
    init_p.add_argument("--force", action="store_true", help="Overwrite an existing host entry")

    bind_p = sub.add_parser("bind", parents=[common], help="Bind a local repo or checkout to a remote host/env")
    bind_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Local project directory (default: cwd)")
    bind_p.add_argument("--scope", choices=("checkout", "repo"), default="checkout",
                        help="Binding scope: 'repo' (shared by all worktrees) or 'checkout' (this worktree only)")
    bind_p.add_argument("--workspace", default="", help="Pin workspace ID (checkout scope only)")
    bind_p.add_argument("--gpus", default="", help="Default GPU indices for this binding, e.g. '0' or '0,1'")
    bind_p.add_argument("--clear-gpus", action="store_true", help="Clear bound default GPUs (back to CPU default)")
    bind_p.add_argument("--note", default=None, help="Short human/AI note about preinstalled remote tools")
    bind_p.add_argument("--path", action="append", default=None, help="Remote PATH entry for this project/host (repeatable)")
    bind_p.add_argument("--env", action="append", default=[], help="KEY=VALUE env var for this project/host (repeatable)")
    bind_p.add_argument("--unset-env", action="append", default=[], help="Remove an env key from binding (repeatable)")
    bind_p.add_argument("--replace", action="store_true", help="Replace existing binding instead of merging")

    unbind_p = sub.add_parser("unbind", parents=[common], help="Remove a private repo or checkout binding")
    unbind_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Local project directory (default: cwd)")
    unbind_p.add_argument("--scope", choices=("checkout", "repo"), default="checkout")

    cfg_p = sub.add_parser("config", parents=[common], help="Validate configs and inspect effective merged settings")
    cfg_p.add_argument("--repo", type=Path, default=None, help="Local project path (default: cwd unless --no-project)")
    cfg_p.add_argument("--no-project", action="store_true", help="Inspect host-only configuration without project context")
    cfg_p.add_argument("--workspace", default="", help="Workspace ID override")
    cfg_p.add_argument("--gpus", default="", help="GPU indices override (e.g. '0,1')")
    cfg_p.add_argument("--no-gpu", action="store_true", help="Force CPU execution without GPU allocation")
    cfg_p.add_argument("--path", action="append", default=[], help="Extra PATH entry preview (repeatable)")
    cfg_p.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE preview (repeatable)")
    cfg_p.add_argument("--max-untracked-mib", type=int, default=None, help="Override untracked upload budget (MiB)")

    doc_p = sub.add_parser("doctor", parents=[common], help="Read-only check of SSH, tools, PATH layers, commands and GPUs")
    doc_p.add_argument("--repo", type=Path, default=None, help="Local project path (default: cwd unless --no-project)")
    doc_p.add_argument("--no-project", action="store_true", help="Check host-only readiness without project context")
    doc_p.add_argument("--workspace", default="", help="Workspace ID override")
    doc_p.add_argument("--path", action="append", default=[], help="Extra PATH entry to include in probe (repeatable)")
    doc_p.add_argument("--check-cmd", action="append", default=[], help="Bare executable or literal path to verify (repeatable)")
    doc_p.add_argument("--session", default="", help="Include saved session cwd when checking relative commands")

    sync_p = sub.add_parser("sync", parents=[common], help="One-way manifest-tracked sync to remote workspace")
    sync_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Local project directory (default: cwd)")
    sync_p.add_argument("--workspace", default="", help="Workspace ID override")
    sync_p.add_argument("--dry-run", action="store_true", help="Preview manifest, budget, updates and ghost cleanup")
    sync_p.add_argument("--max-untracked-mib", type=int, default=None, help="Override untracked upload budget (MiB, 0=unlimited)")

    ex_p = sub.add_parser("exec", parents=[common], help="Sync (optional) and run command or shell snippet remotely")
    ex_p.add_argument("--repo", type=Path, default=None, help="Local project directory (default: cwd unless --no-project)")
    ex_p.add_argument("--workspace", default="", help="Workspace ID override")
    ex_p.add_argument("--remote-dir", default="", help="Existing remote directory (only with --no-project --no-sync)")
    ex_p.add_argument("--no-sync", action="store_true", help="Skip upload before executing")
    ex_p.add_argument("--no-project", action="store_true", help="Run ad-hoc without project context (requires --no-sync)")
    ex_p.add_argument("--session", default="", help="Opt-in persistent shell session name")
    ex_p.add_argument("--shell", default="", help="Run an explicit bash string (mutually exclusive with '-- CMD')")
    ex_p.add_argument("--wait", type=float, default=15.0, help="Seconds to wait before detaching (0=async, default 15)")
    ex_p.add_argument("--follow", action="store_true", help="Stream remote output live until completion")
    ex_p.add_argument("--gpus", default="", help="Allocate NVIDIA GPU indices, e.g. '0' or '0,1' (default: binding or CPU)")
    ex_p.add_argument("--no-gpu", action="store_true", help="Force CPU execution with no GPU locks")
    ex_p.add_argument("--path", action="append", default=[], help="Prepend remote or workspace-relative PATH dir (repeatable)")
    ex_p.add_argument("--env", action="append", default=[], help="Extra literal KEY=VALUE env var (repeatable)")
    ex_p.add_argument("--run-id", type=identifier, default=None, help="Explicit run ID for idempotent retries")
    ex_p.add_argument("--tail-bytes", type=int, default=8192, help="Max trailing log bytes to return (default 8192)")
    ex_p.add_argument("--raw-log", action="store_true", help="Disable carriage-return folding and include base64 bytes")
    ex_p.add_argument("--print-log", action="store_true", help="Print log text to stdout and metadata JSON to stderr")
    ex_p.add_argument("--max-untracked-mib", type=int, default=None, help="Override untracked upload budget (MiB)")
    ex_p.add_argument("command", nargs=argparse.REMAINDER, help="Command argv after '--' (e.g. -- cargo test)")

    st_p = sub.add_parser("status", parents=[common], help="Inspect run status/log, or list recent runs")
    st_p.add_argument("run_id", nargs="?", type=identifier, help="Run ID to inspect (omit to list recent runs)")
    st_p.add_argument("--repo", type=Path, default=None, help="Resolve host / filter listed runs by this local repo")
    st_p.add_argument("--project", default="", help="Filter listed runs by project_id")
    st_p.add_argument("--wait", type=float, default=0.0, help="Seconds to wait for terminal status (default 0)")
    st_p.add_argument("--follow", action="store_true", help="Stream live output until completion")
    st_p.add_argument("--offset", type=int, default=None, help="Byte offset to read log from")
    st_p.add_argument("--tail-bytes", type=int, default=8192, help="Max log bytes to return (default 8192)")
    st_p.add_argument("--raw-log", action="store_true", help="Disable carriage-return folding")
    st_p.add_argument("--print-log", action="store_true", help="Print decoded log text to stdout and metadata to stderr")
    st_p.add_argument("--limit", type=int, default=20, help="Max recent runs to list (default 20)")

    ca_p = sub.add_parser("cancel", parents=[common], help="Cancel a running or queued job")
    ca_p.add_argument("run_id", type=identifier, help="Run ID to cancel")
    ca_p.add_argument("--repo", type=Path, default=None, help="Optional repo to infer bound host")

    pl_p = sub.add_parser("pull", parents=[common], help="Pull logs, metadata, eligible artifacts, or workspace paths")
    pl_p.add_argument("run_id", type=identifier, help="Terminal run ID to pull from")
    pl_p.add_argument("destination", type=Path, help="Local destination directory (must not exist unless --force)")
    pl_p.add_argument("--repo", type=Path, default=None, help="Optional repo to infer bound host")
    pl_p.add_argument("--max-file-mib", type=int, default=None, help="Override recorded artifact max size (MiB, 0=unlimited)")
    pl_p.add_argument("--artifact", action="append", default=[], help="Exact artifact relative path to pull (bypasses filters)")
    pl_p.add_argument("--path", action="append", default=[], help="Relative workspace path to pull into DEST/workspace/")
    pl_p.add_argument("--paths-only", action="store_true", help="Pull only --path workspace files")
    pl_p.add_argument("--force", action="store_true", help="Atomically merge into existing destination directory")

    se_p = sub.add_parser("session", parents=[common], help="List or reset opt-in persistent shell sessions")
    se_sub = se_p.add_subparsers(dest="session_action", required=True)
    sl_p = se_sub.add_parser("list", parents=[common], help="List persistent sessions on the host")
    sl_p.add_argument("--repo", type=Path, default=None, help="Filter sessions to this project workspace")
    sl_p.add_argument("--remote-dir", default="", help="Filter sessions to explicit remote directory")
    sl_p.add_argument("--workspace", default="", help="Workspace ID override")
    sr_p = se_sub.add_parser("reset", parents=[common], help="Reset a named persistent session")
    sr_p.add_argument("name", type=identifier, help="Session name to reset")
    sr_p.add_argument("--repo", type=Path, default=None, help="Local project path (default: cwd unless --remote-dir)")
    sr_p.add_argument("--remote-dir", default="", help="Explicit remote directory override")
    sr_p.add_argument("--workspace", default="", help="Workspace ID override")

    pr_p = sub.add_parser("prune", parents=[common], help="Inspect or remove old terminal run directories")
    pr_p.add_argument("--repo", type=Path, default=None, help="Optional repo to infer bound host")
    pr_p.add_argument("--days", type=float, default=14.0, help="Remove terminal runs older than this many days")
    pr_p.add_argument("--apply", action="store_true", help="Actually delete old terminal runs (default dry-run)")
    pr_p.add_argument("--verbose", action="store_true", help="Show all skipped_unmanaged entries instead of capping at 20")
    return parser


def _default_repo(explicit: Path | None, no_project: bool) -> Path | None:
    if no_project:
        return None
    if explicit is not None:
        return explicit
    return Path.cwd()


def cli(argv: list[str] | None = None) -> tuple[dict, bool, bool]:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = getattr(args, "config", None)
    cli_host = getattr(args, "host", None)
    pretty = bool(getattr(args, "pretty", False))
    act = args.action

    if act == "hosts":
        return svc.op_hosts(config=config_path), pretty, False
    if act == "init":
        return svc.op_init(
            args.target, name=args.name, port=args.port, root=args.root,
            default=args.default, force=args.force, config=config_path,
        ), pretty, False
    if act == "bind":
        return svc.op_bind(
            args.repo, scope=args.scope, host=cli_host or "", workspace=args.workspace,
            gpus=parse_gpu_csv(args.gpus) if args.gpus else None,
            clear_gpus=args.clear_gpus, note=args.note, paths=args.path,
            env_vars=parse_env_items(args.env) if args.env else None,
            unset_env=args.unset_env, replace=args.replace, config=config_path,
        ), pretty, False
    if act == "unbind":
        return svc.op_unbind(args.repo, scope=args.scope), pretty, False
    if act == "config":
        return svc.op_config(
            repo=_default_repo(args.repo, args.no_project), host=cli_host, workspace=args.workspace,
            gpus=parse_gpu_csv(args.gpus) if args.gpus else None, no_gpu=args.no_gpu,
            paths=args.path, env_vars=parse_env_items(args.env),
            max_untracked_mib=args.max_untracked_mib, config=config_path,
        ), pretty, False
    if act == "doctor":
        return svc.op_doctor(
            repo=_default_repo(args.repo, args.no_project), host=cli_host, workspace=args.workspace,
            paths=args.path, check_cmds=args.check_cmd, session=args.session, config=config_path,
        ), pretty, False
    if act == "sync":
        return svc.op_sync(
            args.repo, host=cli_host, workspace=args.workspace,
            dry_run=args.dry_run, max_untracked_mib=args.max_untracked_mib, config=config_path,
        ), pretty, False
    if act == "exec":
        cmd = list(args.command)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        out = svc.op_exec(
            repo=_default_repo(args.repo, args.no_project),
            host=cli_host, workspace=args.workspace, remote_dir=args.remote_dir,
            argv=cmd if cmd else None, shell=args.shell,
            no_sync=args.no_sync, no_project=args.no_project,
            session=args.session, wait=args.wait, follow=args.follow,
            gpus=parse_gpu_csv(args.gpus) if args.gpus else None, no_gpu=args.no_gpu,
            paths=args.path, env_vars=parse_env_items(args.env),
            run_id=args.run_id or "", tail_bytes=args.tail_bytes, raw_log=args.raw_log,
            max_untracked_mib=args.max_untracked_mib, config=config_path, emit_banner=True,
        )
        return out, pretty, bool(args.print_log or args.follow)
    if act == "status":
        if not args.run_id and args.print_log:
            raise ValueError("--print-log requires a RUN_ID")
        try:
            out = svc.op_status(
                run_id=args.run_id, host=cli_host, repo=args.repo, project=args.project,
                wait=args.wait, follow=args.follow, offset=args.offset,
                tail_bytes=args.tail_bytes, raw_log=args.raw_log, limit=args.limit,
                config=config_path,
            )
        except KeyboardInterrupt:
            print("\nremote-run: stopped local wait.", file=sys.stderr)
            raise SystemExit(130)
        return out, pretty, bool(args.print_log or args.follow)
    if act == "cancel":
        return svc.op_cancel(args.run_id, host=cli_host, repo=args.repo, config=config_path), pretty, False
    if act == "pull":
        return svc.op_pull(
            args.run_id, args.destination, host=cli_host, repo=args.repo,
            max_file_mib=args.max_file_mib, artifacts=args.artifact,
            paths=args.path, paths_only=args.paths_only, force=args.force, config=config_path,
        ), pretty, False
    if act == "session":
        repo_arg = args.repo if args.repo is not None else (
            None if (args.remote_dir or args.session_action == "list") else Path.cwd()
        )
        return svc.op_session(
            args.session_action, name=getattr(args, "name", ""),
            host=cli_host, repo=repo_arg, workspace=args.workspace,
            remote_dir=args.remote_dir, config=config_path,
        ), pretty, False
    if act == "prune":
        return svc.op_prune(
            host=cli_host, repo=args.repo, days=args.days, apply=args.apply,
            verbose=args.verbose, config=config_path,
        ), pretty, False
    raise ValueError(f"Unsupported action: {act}")


def main() -> None:
    if sys.argv[1:2] == ["_service_rpc"]:
        try:
            raise SystemExit(_run_service_rpc())
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, OverflowError, subprocess.TimeoutExpired, tomllib.TOMLDecodeError) as error:
            print(f"remote-run: {error}", file=sys.stderr)
            raise SystemExit(1)

    try:
        out, pretty, print_log = cli()
        exec_failed = isinstance(out, dict) and bool(out.pop("_exec_failed", False))
        if print_log and isinstance(out, dict) and "log" in out:
            log_obj = out.pop("log")
            if "data" in log_obj:
                svc.output_log_bytes(
                    base64.b64decode(log_obj["data"]),
                    codecs.getincrementaldecoder("utf-8")(errors="replace"),
                    raw_log=True, final=True,
                )
            elif log_obj.get("text"):
                print(log_obj["text"])
            meta = {**out, "log_bytes": log_obj.get("total_bytes"), "truncated": log_obj.get("truncated")}
            print(json.dumps(meta, ensure_ascii=False, indent=2 if pretty else None), file=sys.stderr)
        elif print_log and isinstance(out, dict):
            print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None), file=sys.stderr)
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2 if pretty else None))
        if exec_failed:
            raise SystemExit(2)
    except (ValueError, KeyError, OSError, RuntimeError, OverflowError, subprocess.TimeoutExpired, tomllib.TOMLDecodeError) as error:
        print(f"remote-run: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
