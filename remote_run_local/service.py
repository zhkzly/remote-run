"""Unified typed service layer shared by both CLI (remote_run.py) and MCP (remote_run_mcp.py)."""
from __future__ import annotations

import base64
import codecs
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import shlex
import sys
from uuid import uuid4

from remote_worker import (
    DEFAULT_REMOTE_ROOT,
    TERMINAL,
    check_directory_chain,
    check_env_map,
    identifier,
    join_remote_path,
    open_lock,
    remote_path,
    validate_gpu_indices,
    validate_path_entry,
    write_private_text,
)
from remote_run_local.config import (
    default_hosts_config_path,
    delete_binding,
    host_endpoint_id,
    load_hosts_config,
    redact_config_view,
    render_hosts_toml,
    resolve_effective_config,
    resolve_repo_context,
    sanitize_id,
    select_host_from_catalog,
    ssh_target,
    update_binding_atomic,
)
from remote_run_local.transfer import pull_run_files, sync_workspace
from remote_run_local.transport import ssh_call, workspace_lease


def resolve_config_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None and str(explicit):
        return Path(explicit)
    return default_hosts_config_path()


def continuation(host: dict, run_id: str, config_path: Path | None = None, *, state: dict | None = None) -> str:
    prefix = ["remote-run"]
    if config_path is not None:
        prefix.extend(["--config", str(config_path.absolute())])
    prefix.extend(["--host", host["name"]])
    status_cmd = shlex.join([*prefix, "status", run_id, "--wait", "30"])
    cancel_cmd = shlex.join([*prefix, "cancel", run_id])
    base = f"Check with `{status_cmd}` or cancel with `{cancel_cmd}`"
    if isinstance(state, dict) and state.get("status") == "queued" and state.get("waiting_for_lock"):
        holders = state.get("blocked_by_runs") or []
        if holders:
            return f"Queued waiting for GPU lock (held by {', '.join(holders)}); starts automatically once free. {base}"
        return f"Queued waiting for GPU lock; starts automatically once free. {base}"
    return base


def op_hosts(*, config: str | Path | None = None) -> dict:
    cfg_path = resolve_config_path(config)
    default_host, hosts = load_hosts_config(cfg_path)
    redacted = {}
    for hname, h in hosts.items():
        redacted[hname] = {
            **h,
            "env": {k: "<redacted>" for k in h.get("env", {})},
        }
    return {"default_host": default_host, "config_path": str(cfg_path), "hosts": redacted}


def op_init(
    target: str,
    *,
    name: str = "",
    port: int | None = None,
    root: str = "",
    default: bool = False,
    force: bool = False,
    config: str | Path | None = None,
) -> dict:
    cfg_path = resolve_config_path(config)
    clean_target = ssh_target(target, "target")
    if port is not None and (isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535)):
        raise ValueError("--port must be an integer between 1 and 65535")
    default_base = clean_target.split("@", 1)[1] if "@" in clean_target else clean_target
    host_name = identifier(name, "name") if name else sanitize_id(default_base, "server")
    root_cfg = remote_path(root, "root") if root else DEFAULT_REMOTE_ROOT

    if cfg_path.exists():
        _, existing_hosts = load_hosts_config(cfg_path)
        if host_name in existing_hosts and not force:
            raise ValueError(f"Host {host_name!r} already exists in {cfg_path}; pass --force to update")

    probe_host = {"name": host_name, "ssh": clean_target, "port": port, "root": root_cfg, "path": [], "gpu": None}
    probe = ssh_call(probe_host, "probe_init", {"root": root_cfg}, timeout=30)
    if probe["missing_tools"]:
        raise ValueError(f"Compute host missing required tool(s): {', '.join(probe['missing_tools'])}")

    observed_gpus = probe.get("gpus", [])
    gpu_indices = [g["index"] for g in observed_gpus if isinstance(g.get("index"), int)]
    new_entry = {
        "name": host_name,
        "ssh": clean_target,
        "port": port,
        "root": root_cfg,
        "path": [],
        "env": {},
        "gpu": {"devices": gpu_indices, "locks_dir": join_remote_path(root_cfg, "locks")} if gpu_indices else None,
    }

    cfg_path = cfg_path.parent.resolve() / cfg_path.name
    check_directory_chain(cfg_path.parent, "Host configuration directory", trusted=True)
    cfg_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open_lock(cfg_path.with_name(f".{cfg_path.name}.lock")) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cfg_path.is_symlink():
            raise ValueError(f"Refusing to update symlink host config: {cfg_path}")
        current_default, updated_hosts = load_hosts_config(cfg_path) if cfg_path.exists() else (None, {})
        if host_name in updated_hosts and not force:
            raise ValueError(f"Host {host_name!r} already exists in {cfg_path}; pass --force to update")
        updated_hosts[host_name] = new_entry
        chosen_default = host_name if (default or current_default is None) else current_default
        write_private_text(cfg_path, render_hosts_toml(chosen_default, updated_hosts))
    return {
        "config_path": str(cfg_path),
        "host": host_name,
        "default_host": chosen_default,
        "ssh": clean_target,
        "port": port,
        "root": root_cfg,
        "resolved_root": probe["resolved_root"],
        "gpus": observed_gpus,
        "tools": probe["tools"],
        "python_version": probe.get("python_version", ""),
    }


def op_bind(
    repo: str | Path,
    *,
    scope: str = "checkout",
    host: str = "",
    workspace: str = "",
    gpus: list[int] | None = None,
    clear_gpus: bool = False,
    note: str | None = None,
    paths: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    unset_env: list[str] = (),
    replace: bool = False,
    config: str | Path | None = None,
) -> dict:
    if scope not in ("repo", "checkout"):
        raise ValueError("scope must be 'repo' or 'checkout'")
    if scope == "repo" and workspace:
        raise ValueError("--workspace is only allowed for checkout-scoped bindings (not --scope repo)")
    if clear_gpus and gpus is not None:
        raise ValueError("Cannot combine --gpus and --clear-gpus")

    cfg_path = resolve_config_path(config)
    _, hosts = load_hosts_config(cfg_path)
    repo_ctx = resolve_repo_context(Path(repo))
    key = repo_ctx["repo_binding_key"] if scope == "repo" else repo_ctx["checkout_binding_key"]

    def mutate(loaded: dict) -> dict:
        existing = {
            "host": "", "host_endpoint": "", "workspace": "", "gpus": None, "note": "", "path": [], "env": {},
        } if replace else loaded
        target_host = identifier(host, "host") if host else existing["host"]
        if target_host and target_host not in hosts:
            raise ValueError(f"Unknown host {target_host!r}; configured: {', '.join(hosts)}")
        gpu_managed = bool(target_host and hosts[target_host]["gpu"] is not None)
        target_ep = host_endpoint_id(hosts[target_host]) if target_host else ""

        # If switching host OR host endpoint changed/missing in existing binding, reset old host-bound fields.
        host_changed = bool(
            (host and existing["host"] and target_host != existing["host"])
            or (existing["host"] and target_ep and existing.get("host_endpoint") != target_ep)
        )
        base_gpus = None if host_changed else existing["gpus"]
        base_note = "" if host_changed else existing["note"]
        base_paths = [] if host_changed else list(existing["path"])
        base_env = {} if host_changed else dict(existing["env"])

        new_ws = identifier(workspace, "workspace") if workspace else (existing["workspace"] if scope == "checkout" else "")
        if clear_gpus:
            new_gpus = None
        elif gpus is not None:
            new_gpus = validate_gpu_indices(gpus, "gpus")
        else:
            new_gpus = base_gpus

        new_note = note if note is not None else base_note
        if not isinstance(new_note, str) or any(ord(c) < 32 for c in new_note):
            raise ValueError("note must be a single-line string without control chars")
        new_paths = (
            [validate_path_entry(p, "binding.path", allow_relative=True) for p in paths]
            if paths is not None else base_paths
        )
        new_env = dict(base_env)
        for k in unset_env:
            new_env.pop(k, None)
        if env_vars:
            new_env.update(check_env_map(env_vars, "binding.env", reject_reserved=True, gpu_managed=gpu_managed))
        else:
            check_env_map(new_env, "binding.env", reject_reserved=True, gpu_managed=gpu_managed)

        if (new_gpus is not None or new_paths or new_env or new_note) and not target_host:
            raise ValueError("Setting host-specific fields (gpus/path/env/note) requires --host")
        if new_gpus:
            if not gpu_managed:
                raise ValueError(f"Host {target_host!r} has no [gpu] configured; cannot bind gpus {new_gpus}")
            declared = hosts[target_host]["gpu"]["devices"]
            for idx in new_gpus:
                if idx not in declared:
                    raise ValueError(f"GPU {idx} is not in hosts.{target_host}.gpu.devices {declared}")

        return {
            "host": target_host,
            "host_endpoint": target_ep,
            "workspace": new_ws,
            "gpus": new_gpus,
            "note": new_note,
            "path": new_paths,
            "env": new_env,
        }

    saved, data = update_binding_atomic(scope, key, mutate)
    return {
        "bound": True,
        "scope": scope,
        "file": str(saved),
        "repo_root": str(repo_ctx["repo_root"]),
        "host": data["host"],
        "workspace": data["workspace"],
        "gpus": data["gpus"],
        "note": data["note"],
        "path": data["path"],
        "env_keys": sorted(data["env"]),
    }


def op_unbind(repo: str | Path, *, scope: str = "checkout") -> dict:
    if scope not in ("repo", "checkout"):
        raise ValueError("scope must be 'repo' or 'checkout'")
    repo_ctx = resolve_repo_context(Path(repo))
    key = repo_ctx["repo_binding_key"] if scope == "repo" else repo_ctx["checkout_binding_key"]
    path, existed = delete_binding(scope, key)
    return {"unbound": True, "scope": scope, "file": str(path), "existed": existed}


def op_config(
    *,
    repo: str | Path | None = None,
    host: str | None = None,
    workspace: str = "",
    gpus: list[int] | None = None,
    no_gpu: bool = False,
    paths: list[str] = (),
    env_vars: dict[str, str] | None = None,
    max_untracked_mib: int | None = None,
    config: str | Path | None = None,
) -> dict:
    cfg_path = resolve_config_path(config)
    resolved = resolve_effective_config(
        cfg_path,
        repo_dir=Path(repo) if repo is not None else None,
        cli_host=host,
        workspace_override=workspace,
        cli_gpus=gpus,
        cli_no_gpu=no_gpu,
        cli_paths=list(paths),
        cli_env=env_vars or {},
        max_untracked_mib=max_untracked_mib,
        inspect_dirty=False,
    )
    return redact_config_view(resolved)


def op_doctor(
    *,
    repo: str | Path | None = None,
    host: str | None = None,
    workspace: str = "",
    paths: list[str] = (),
    check_cmds: list[str] = (),
    session: str = "",
    config: str | Path | None = None,
) -> dict:
    cfg_path = resolve_config_path(config)
    resolved = resolve_effective_config(
        cfg_path,
        repo_dir=Path(repo) if repo is not None else None,
        cli_host=host,
        workspace_override=workspace,
        cli_paths=list(paths),
        inspect_dirty=False,
    )
    ws = resolved["workspace"]
    res = ssh_call(resolved["host"], "doctor", {
        "workdir": ws["workdir"] if ws else "",
        "rel_subdir": ws["rel_subdir"] if ws else "",
        "path_layers": resolved["path_layers"],
        "check_cmds": list(check_cmds),
        "session": identifier(session, "session") if session else "",
    }, read_only_transport=True)
    res["host_origin"] = resolved["host_origin"]
    res["note"] = resolved["note"]
    return res


def op_sync(
    repo: str | Path,
    *,
    host: str | None = None,
    workspace: str = "",
    dry_run: bool = False,
    max_untracked_mib: int | None = None,
    config: str | Path | None = None,
) -> dict:
    cfg_path = resolve_config_path(config)
    resolved = resolve_effective_config(
        cfg_path,
        repo_dir=Path(repo),
        cli_host=host,
        workspace_override=workspace,
        max_untracked_mib=max_untracked_mib,
        inspect_dirty=False,
    )
    return sync_workspace(resolved["repo_ctx"]["repo_root"], resolved, dry_run=dry_run)


def wait_for_status(
    host: dict,
    run_id: str,
    wait_sec: float,
    *,
    offset: int | None = None,
    tail_bytes: int = 8192,
    raw_log: bool = False,
    config_path: Path | None = None,
) -> dict:
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
        res["next_action"] = continuation(host, run_id, config_path, state=res)
    return res


def output_log_bytes(data: bytes, decoder, *, raw_log: bool, final: bool) -> None:
    if raw_log and hasattr(sys.stdout, "buffer"):
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    else:
        text = decoder.decode(data, final=final)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()


def follow_run(host: dict, run_id: str, *, raw_log: bool = False) -> dict:
    offset = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        res = ssh_call(
            host,
            "status",
            {
                "run_id": run_id,
                "include_log": True,
                "offset": offset,
                "tail_bytes": 65536,
                "binary_log": True,
                "wait_sec": 20.0,
                "wake_on_output": True,
            },
            timeout=50,
        )
        log_obj = res.get("log", {})
        data = base64.b64decode(log_obj.get("data", ""))
        terminal = res.get("status") in TERMINAL and not log_obj.get("truncated")
        output_log_bytes(data, decoder, raw_log=raw_log, final=terminal)
        offset = log_obj.get("next_offset", offset)
        if terminal:
            res.pop("log", None)
            res["log_bytes"] = log_obj.get("total_bytes")
            res["detached"] = False
            if res.get("status") in ("failed", "interrupted", "cancelled"):
                res["_exec_failed"] = True
            return res


def op_exec(
    *,
    repo: str | Path | None = None,
    host: str | None = None,
    workspace: str = "",
    remote_dir: str = "",
    argv: list[str] | None = None,
    shell: str = "",
    no_sync: bool = False,
    no_project: bool = False,
    session: str = "",
    wait: float = 15.0,
    follow: bool = False,
    gpus: list[int] | None = None,
    no_gpu: bool = False,
    paths: list[str] = (),
    env_vars: dict[str, str] | None = None,
    run_id: str = "",
    tail_bytes: int = 8192,
    raw_log: bool = False,
    max_untracked_mib: int | None = None,
    config: str | Path | None = None,
    emit_banner: bool = False,
) -> dict:
    # Strict mutual exclusion and element validation BEFORE any SSH/sync work.
    if argv is not None and not isinstance(argv, list):
        raise ValueError("argv must be a list of strings")
    if not isinstance(shell, str):
        raise ValueError("shell must be a string")
    has_shell = bool(shell)
    has_argv = bool(argv)
    if has_shell and has_argv:
        raise ValueError("Provide either argv (after '--' / MCP 'argv') OR shell, never both")
    if not has_shell and not has_argv:
        raise ValueError("Provide either argv (after '--' / MCP 'argv') or shell")
    if has_argv:
        if not all(isinstance(a, str) and "\0" not in a for a in argv or ()):
            raise ValueError("argv must be a list of strings without NUL bytes")
        if not argv[0] or any(ord(c) < 32 for c in argv[0]):
            raise ValueError(f"Invalid executable in argv[0]: {argv[0]!r}")
    if has_shell and "\0" in shell:
        raise ValueError("shell command cannot contain NUL bytes")
    if not math.isfinite(wait) or wait < 0:
        raise ValueError("wait must be a finite non-negative number of seconds")
    if tail_bytes < 0:
        raise ValueError("tail_bytes must be non-negative")
    if follow and (wait != 15.0 or tail_bytes != 8192):
        raise ValueError("--follow cannot be combined with custom --wait or --tail-bytes")
    if no_project and not no_sync:
        raise ValueError("--no-project requires --no-sync")
    if no_project and workspace:
        raise ValueError("--workspace cannot be used with --no-project")
    if remote_dir and not no_project:
        raise ValueError("--remote-dir is only for ad-hoc execution with --no-project --no-sync")
    if not no_project and repo is None:
        raise ValueError("Project execution requires repo (or pass --no-project --no-sync for ad-hoc execution)")

    cfg_path = resolve_config_path(config)
    resolved = resolve_effective_config(
        cfg_path,
        repo_dir=None if no_project else Path(repo),  # type: ignore[arg-type]
        cli_host=host,
        workspace_override="" if no_project else workspace,
        cli_gpus=gpus,
        cli_no_gpu=no_gpu,
        cli_paths=list(paths),
        cli_env=env_vars or {},
        max_untracked_mib=max_untracked_mib,
        inspect_dirty=not no_project,
    )
    host_cfg = resolved["host"]

    if no_project:
        adhoc_existing = bool(remote_dir)
        workdir_raw = remote_path(remote_dir, "remote_dir") if remote_dir else join_remote_path(host_cfg["root"], "adhoc")
        ws_info = {
            "project_id": "",
            "workspace_id": "",
            "managed": not adhoc_existing,
            "workdir": workdir_raw,
            "shared_dir": "",
            "rel_subdir": "",
            "in_git": False,
            "root_commit": "",
            "git_commit": "",
            "git_branch": "",
            "git_dirty": False,
        }
    else:
        adhoc_existing = False
        ws_info = resolved["workspace"]

    chosen_run_id = identifier(run_id, "run_id") if run_id else (
        datetime.now(timezone.utc).strftime("rr-%Y%m%dT%H%M%S-") + uuid4().hex[:8]
    )
    sess_name = identifier(session, "session") if session else ""
    resume = continuation(host_cfg, chosen_run_id, cfg_path)
    submit_state = "before_submit"
    try:
        with workspace_lease(host_cfg, ws_info["workdir"]) as lease:
            rpc = {"lease": lease}
            active_warning: list[str] = []
            workdir = ws_info["workdir"]
            shared_dir = ws_info["shared_dir"]
            skipped_for_existing_run = False
            if not no_sync:
                synced = sync_workspace(resolved["repo_ctx"]["repo_root"], resolved, run_id=chosen_run_id, lease=lease)
                workdir = synced.get("workdir", workdir)
                skipped_for_existing_run = bool(synced.get("skipped_for_existing_run"))
            elif not no_project:
                # Validate existing managed workspace ownership WITHOUT creating an unsynced empty workspace.
                prep = ssh_call(host_cfg, "prepare_workspace", {
                    "workspace": ws_info,
                    "shared_dirs": resolved["sync_policy"]["shared"],
                    "run_id": chosen_run_id,
                    "for_sync": False,
                    "create_if_missing": False,
                }, **rpc)
                active_warning = prep.get("active_runs_warning", [])
                workdir = prep["workdir"]
                shared_dir = prep["shared_dir"]

            request = {
                "host": host_cfg["name"],
                "gpus": resolved["gpus"],
                "path_layers": resolved["path_layers"],
                "config_env": resolved["config_env"],
                "invocation_env": resolved["invocation_env"],
                "workdir": workdir,
                "shared_dir": shared_dir,
                "rel_subdir": ws_info["rel_subdir"],
                "require_workdir": bool(no_sync and not no_project),
                "reuse_only": skipped_for_existing_run,
                "adhoc_existing_dir": adhoc_existing,
                "project_id": ws_info["project_id"],
                "workspace_id": ws_info["workspace_id"],
                "root_commit": ws_info.get("root_commit", ""),
                "git_commit": ws_info.get("git_commit", ""),
                "git_branch": ws_info.get("git_branch", ""),
                "git_dirty": bool(ws_info.get("git_dirty", False)),
                "session": sess_name,
                "session_key": "",  # Derived authoritatively on worker after expanding '~/' in workdir
                "argv": [] if has_shell else list(argv or []),
                "shell": shell if has_shell else "",
                "pull_policy": resolved["pull_policy"],
            }
            submit_state = "submitting"
            receipt = ssh_call(host_cfg, "submit", {"run_id": chosen_run_id, "request": request}, **rpc)
            submit_state = "submitted"
            if emit_banner:
                print(f"remote-run: run {chosen_run_id} on {host_cfg['name']}. {resume}", file=sys.stderr, flush=True)
        if follow:
            out = follow_run(host_cfg, chosen_run_id, raw_log=raw_log)
        elif wait > 0:
            out = wait_for_status(
                host_cfg, chosen_run_id, wait, tail_bytes=tail_bytes, raw_log=raw_log, config_path=cfg_path,
            )
        else:
            out = receipt
    except KeyboardInterrupt:
        if emit_banner:
            if submit_state == "submitted":
                print(f"\nremote-run: stopped local wait for {chosen_run_id}. {resume}", file=sys.stderr)
            elif submit_state == "submitting":
                print(
                    f"\nremote-run: interrupted while submitting {chosen_run_id} on {host_cfg['name']}; "
                    f"remote launch status is unconfirmed. {resume}",
                    file=sys.stderr,
                )
            else:
                print("\nremote-run: interrupted before job submission completed.", file=sys.stderr)
        raise SystemExit(130)

    out["reused"] = receipt.get("reused", False)
    out["detached"] = out["status"] not in TERMINAL
    if out["detached"]:
        out["next_action"] = continuation(host_cfg, chosen_run_id, cfg_path, state=out)
    if active_warning:
        out["active_runs_warning"] = active_warning
    if resolved["bindings"]["skipped"]:
        out["skipped_bindings"] = resolved["bindings"]["skipped"]
    if out["status"] in ("failed", "interrupted", "cancelled"):
        out["_exec_failed"] = True
    return out


def _select_host_for_run_op(config_path: Path, host: str | None, repo: str | Path | None = None) -> tuple[dict, dict | None]:
    if repo is not None:
        resolved = resolve_effective_config(config_path, repo_dir=Path(repo), cli_host=host)
        return resolved["host"], resolved
    default_host, hosts = load_hosts_config(config_path)
    selected, _ = select_host_from_catalog(default_host, hosts, cli_host=host)
    return selected, None


def op_status(
    *,
    run_id: str | None = None,
    host: str | None = None,
    repo: str | Path | None = None,
    project: str = "",
    wait: float = 0.0,
    follow: bool = False,
    offset: int | None = None,
    tail_bytes: int = 8192,
    raw_log: bool = False,
    limit: int = 20,
    config: str | Path | None = None,
) -> dict:
    cfg_path = resolve_config_path(config)
    host_cfg, resolved = _select_host_for_run_op(cfg_path, host, repo)
    if not run_id:
        if follow or wait or offset is not None or tail_bytes != 8192 or raw_log:
            raise ValueError("Waiting and log options require a run_id")
        if limit < 1:
            raise ValueError("limit must be positive")
        proj_filter = project
        if not proj_filter and resolved is not None and resolved["workspace"] is not None:
            proj_filter = resolved["workspace"]["project_id"]
        return ssh_call(host_cfg, "list", {"limit": limit, "project_id": proj_filter})

    rid = identifier(run_id, "run_id")
    if project or limit != 20:
        raise ValueError("--project and --limit are for listing runs without a run_id")
    if follow:
        if wait or offset is not None or tail_bytes != 8192:
            raise ValueError("--follow cannot be combined with --wait, --offset or --tail-bytes")
        return follow_run(host_cfg, rid, raw_log=raw_log)
    return wait_for_status(
        host_cfg, rid, wait, offset=offset, tail_bytes=tail_bytes, raw_log=raw_log, config_path=cfg_path,
    )


def op_cancel(run_id: str, *, host: str | None = None, repo: str | Path | None = None, config: str | Path | None = None) -> dict:
    cfg_path = resolve_config_path(config)
    host_cfg, _ = _select_host_for_run_op(cfg_path, host, repo)
    return ssh_call(host_cfg, "cancel", {"run_id": identifier(run_id, "run_id")})


def op_pull(
    run_id: str,
    destination: str | Path,
    *,
    host: str | None = None,
    repo: str | Path | None = None,
    max_file_mib: int | None = None,
    artifacts: list[str] = (),
    paths: list[str] = (),
    paths_only: bool = False,
    force: bool = False,
    config: str | Path | None = None,
) -> dict:
    cfg_path = resolve_config_path(config)
    host_cfg, _ = _select_host_for_run_op(cfg_path, host, repo)
    return pull_run_files(
        host_cfg,
        identifier(run_id, "run_id"),
        Path(destination),
        max_file_mib=max_file_mib,
        artifacts=list(artifacts),
        paths=list(paths),
        paths_only=paths_only,
        force=force,
    )


def op_session(
    action: str,
    *,
    name: str = "",
    host: str | None = None,
    repo: str | Path | None = None,
    workspace: str = "",
    remote_dir: str = "",
    config: str | Path | None = None,
) -> dict:
    if action not in ("list", "reset"):
        raise ValueError(f"Unsupported session action: {action!r}")
    cfg_path = resolve_config_path(config)
    wdir = ""
    if remote_dir:
        wdir = remote_path(remote_dir, "remote_dir")
        host_cfg, _ = _select_host_for_run_op(cfg_path, host, repo)
    elif repo is not None:
        resolved = resolve_effective_config(cfg_path, repo_dir=Path(repo), cli_host=host, workspace_override=workspace)
        host_cfg = resolved["host"]
        wdir = resolved["workspace"]["workdir"]
    else:
        if action == "reset":
            raise ValueError("session reset requires repo or remote_dir")
        host_cfg, _ = _select_host_for_run_op(cfg_path, host, None)
    payload: dict = {"subaction": action, "workdir": wdir}
    if action == "reset":
        payload["name"] = identifier(name, "session")
    return ssh_call(host_cfg, "session", payload)


def op_prune(
    *,
    host: str | None = None,
    repo: str | Path | None = None,
    days: float = 14.0,
    apply: bool = False,
    verbose: bool = False,
    config: str | Path | None = None,
) -> dict:
    if not math.isfinite(days) or days < 0:
        raise ValueError("days must be a non-negative finite number")
    cfg_path = resolve_config_path(config)
    host_cfg, _ = _select_host_for_run_op(cfg_path, host, repo)
    return ssh_call(host_cfg, "prune", {"days": days, "apply": apply, "verbose": verbose})


# === Typed Service Dispatch (for cancellable MCP subprocess invocations) ===

OPERATIONS = {
    "hosts": op_hosts,
    "init": op_init,
    "bind": op_bind,
    "unbind": op_unbind,
    "config": op_config,
    "doctor": op_doctor,
    "sync": op_sync,
    "exec": op_exec,
    "status": op_status,
    "cancel": op_cancel,
    "pull": op_pull,
    "session": op_session,
    "prune": op_prune,
}


def dispatch_operation(op: str, params: dict) -> dict:
    if op not in OPERATIONS:
        raise ValueError(f"Unknown service operation: {op!r}")
    return OPERATIONS[op](**params)
