"""Three-layer configuration, repository identity, bindings, and provenance merging (v2)."""
from __future__ import annotations

import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

from remote_worker import (
    DEFAULT_REMOTE_ROOT,
    SCHEMA_VERSION,
    check_directory_chain,
    check_env_map,
    identifier,
    join_remote_path,
    open_lock,
    rel_path,
    remote_path,
    validate_gpu_indices,
    validate_path_entry,
    validate_pattern,
    write_private_text,
)

HOST_TOP_KEYS = {"schema_version", "default_host", "hosts"}
HOST_ENTRY_KEYS = {"ssh", "port", "root", "path", "env", "gpu"}
HOST_GPU_KEYS = {"devices", "locks_dir"}

PROJECT_TOP_KEYS = {"schema_version", "project", "path", "env", "workspace", "sync", "pull"}
PROJECT_WORKSPACE_KEYS = {"shared"}
PROJECT_SYNC_KEYS = {"exclude", "preserve", "max_untracked_mib"}
PROJECT_PULL_KEYS = {"max_file_mib", "exclude"}

REPO_BINDING_KEYS = {"schema_version", "host", "host_endpoint", "gpus", "note", "path", "env"}
CHECKOUT_BINDING_KEYS = {"schema_version", "host", "host_endpoint", "workspace", "gpus", "note", "path", "env"}


def host_endpoint_id(host: dict) -> str:
    """Stable endpoint identity so bindings never cross different SSH targets/roots sharing an alias."""
    port_s = str(host["port"]) if host.get("port") else ""
    return f"{host['ssh']}|{port_s}|{host['root']}"


def xdg_config_root() -> Path:
    raw = os.environ.get("XDG_CONFIG_HOME", "")
    if raw and os.path.isabs(raw):
        return Path(raw) / "remote-run"
    return Path.home() / ".config" / "remote-run"


def default_hosts_config_path() -> Path:
    env_cfg = os.environ.get("REMOTE_RUN_CONFIG", "")
    if env_cfg:
        return Path(env_cfg)
    return xdg_config_root() / "hosts.toml"


def bindings_dir() -> Path:
    return xdg_config_root() / "bindings"


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else toml_string(key)


def toml_str_list(items: list[str]) -> str:
    return "[" + ", ".join(toml_string(x) for x in items) + "]"


def toml_int_list(items: list[int]) -> str:
    return "[" + ", ".join(str(int(x)) for x in items) + "]"


def check_keys(mapping: object, allowed: set[str], section: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{section} must be a TOML table")
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {section}: {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def check_schema_version(mapping: dict, label: str) -> None:
    if "schema_version" not in mapping:
        raise ValueError(f"{label} requires integer schema_version = {SCHEMA_VERSION}")
    version = mapping["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema_version must be integer {SCHEMA_VERSION} (got {version!r}; "
            f"v1 configs are not supported in remote-run v2)"
        )


def ssh_target(value: object, label: str = "ssh") -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value) or value.startswith("-"):
        raise ValueError(f"Invalid {label} {value!r}; use a ~/.ssh/config alias or 'user@host'")
    if ":" in value:
        raise ValueError(
            f"Invalid {label} {value!r}: ':' is not allowed here; set 'port = <number>' or use ~/.ssh/config"
        )
    if "@" in value:
        user, _, hostname = value.partition("@")
        if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,63}", user) or not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", hostname
        ):
            raise ValueError(f"Invalid {label} {value!r}; expected 'user@host' with safe characters")
        return value
    return identifier(value, label)


def sanitize_id(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip(".-_")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalnum():
        cleaned = "x" + cleaned
    return cleaned[:64]


def _check_no_overlaps(dirs: list[str], label: str) -> list[str]:
    normed: list[str] = []
    seen: set[str] = set()
    for raw in dirs:
        r = rel_path(raw, label)
        if r in seen:
            raise ValueError(f"{label} contains duplicate entry {r!r}")
        for prev in normed:
            if r.startswith(prev + "/") or prev.startswith(r + "/"):
                raise ValueError(f"{label} contains overlapping directories {prev!r} and {r!r}")
        seen.add(r)
        normed.append(r)
    return normed


def git_query(
    argv: list[str],
    *,
    cwd: Path,
    stdin: str | None = None,
    accepted: tuple[int, ...] = (0,),
    strip: bool = True,
) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", *argv],
            cwd=cwd, input=stdin, capture_output=True, text=True, errors="surrogateescape", timeout=15,
        )
    except FileNotFoundError:
        return False, ""
    out = completed.stdout.strip() if strip else completed.stdout
    return completed.returncode in accepted, out


# === Layer 1: Host Catalog (~/.config/remote-run/hosts.toml) ===

def load_hosts_config(path: Path) -> tuple[str | None, dict[str, dict]]:
    if not path.is_file():
        raise ValueError(f"Host config not found: {path}. Run `remote-run init <ssh-target>` or create {path}")
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    check_keys(cfg, HOST_TOP_KEYS, str(path))
    check_schema_version(cfg, str(path))
    hosts = cfg.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError(f"{path} must define at least one [hosts.<name>] entry")

    parsed: dict[str, dict] = {}
    for hname, raw in hosts.items():
        identifier(hname, "host")
        check_keys(raw, HOST_ENTRY_KEYS, f"{path} [hosts.{hname}]")
        if "ssh" not in raw:
            raise ValueError(f"{path} [hosts.{hname}] requires 'ssh'")
        ssh = ssh_target(raw["ssh"], f"hosts.{hname}.ssh")
        port = None
        if "port" in raw:
            port = raw["port"]
            if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
                raise ValueError(f"hosts.{hname}.port must be an integer between 1 and 65535")
        root = remote_path(raw["root"], f"hosts.{hname}.root") if "root" in raw else DEFAULT_REMOTE_ROOT

        raw_path = raw.get("path", [])
        if not isinstance(raw_path, list):
            raise ValueError(f"hosts.{hname}.path must be a list of paths")
        host_path = [validate_path_entry(x, f"hosts.{hname}.path", allow_relative=False) for x in raw_path]

        gpu_cfg = None
        if "gpu" in raw:
            graw = raw["gpu"]
            check_keys(graw, HOST_GPU_KEYS, f"{path} [hosts.{hname}.gpu]")
            if "devices" not in graw:
                raise ValueError(f"hosts.{hname}.gpu requires nonempty 'devices'")
            devices = validate_gpu_indices(graw["devices"], f"hosts.{hname}.gpu.devices")
            if not devices:
                raise ValueError(f"hosts.{hname}.gpu.devices must be nonempty")
            locks_dir = (
                remote_path(graw["locks_dir"], f"hosts.{hname}.gpu.locks_dir")
                if "locks_dir" in graw
                else join_remote_path(root, "locks")
            )
            gpu_cfg = {"devices": devices, "locks_dir": locks_dir}

        host_env = check_env_map(
            raw.get("env", {}), f"hosts.{hname}.env", reject_reserved=True, gpu_managed=gpu_cfg is not None,
        )
        parsed[hname] = {
            "name": hname,
            "ssh": ssh,
            "port": port,
            "root": root,
            "path": host_path,
            "env": host_env,
            "gpu": gpu_cfg,
        }

    default_host = cfg.get("default_host")
    if default_host is not None:
        identifier(default_host, "default_host")
        if default_host not in parsed:
            raise ValueError(f"default_host {default_host!r} not found in hosts")
    return default_host, parsed


def render_hosts_toml(default_host: str | None, hosts: dict[str, dict]) -> str:
    lines = [f"schema_version = {SCHEMA_VERSION}"]
    if default_host:
        lines.append(f"default_host = {toml_string(default_host)}")
    lines.append("")
    for hname, h in hosts.items():
        hk = toml_key(hname)
        lines.append(f"[hosts.{hk}]")
        lines.append(f"ssh = {toml_string(h['ssh'])}")
        if h.get("port"):
            lines.append(f"port = {int(h['port'])}")
        if h.get("root") and h["root"] != DEFAULT_REMOTE_ROOT:
            lines.append(f"root = {toml_string(h['root'])}")
        if h.get("path"):
            lines.append(f"path = {toml_str_list(h['path'])}")
        lines.append("")
        if h.get("env"):
            lines.append(f"[hosts.{hk}.env]")
            for k, v in h["env"].items():
                lines.append(f"{toml_key(k)} = {toml_string(v)}")
            lines.append("")
        if h.get("gpu"):
            g = h["gpu"]
            lines.append(f"[hosts.{hk}.gpu]")
            lines.append(f"devices = {toml_int_list(g['devices'])}")
            default_locks = join_remote_path(h["root"], "locks")
            if g.get("locks_dir") and g["locks_dir"] != default_locks:
                lines.append(f"locks_dir = {toml_string(g['locks_dir'])}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def select_host_from_catalog(
    default_host: str | None,
    hosts: dict[str, dict],
    *,
    cli_host: str | None = None,
    checkout_host: str | None = None,
    repo_host: str | None = None,
) -> tuple[dict, str]:
    """Deterministic host selection: CLI -> checkout binding -> repo binding -> default_host -> sole host."""
    for candidate, origin in (
        (cli_host, "cli"),
        (checkout_host, "checkout_binding"),
        (repo_host, "repo_binding"),
        (default_host, "default_host"),
    ):
        if candidate:
            name = identifier(candidate, "host")
            if name not in hosts:
                raise ValueError(f"Unknown host {name!r} (from {origin}); configured: {', '.join(hosts)}")
            return hosts[name], origin
    if len(hosts) == 1:
        sole = next(iter(hosts.values()))
        return sole, "sole_host"
    raise ValueError(
        f"Multiple hosts configured ({', '.join(hosts)}) and no default_host or binding selected; "
        "pass --host, set default_host in hosts.toml, or run `remote-run bind`"
    )


# === Layer 2: Tracked Project Policy (<repo>/.remote-run.toml) ===

def empty_project_config() -> dict:
    return {
        "present": False,
        "file": None,
        "project": "",
        "path": [],
        "env": {},
        "workspace": {"shared": []},
        "sync": {"exclude": [], "preserve": [], "max_untracked_mib": 256},
        "pull": {"max_file_mib": 10, "exclude": []},
    }


def load_project_config(repo_root: Path) -> dict:
    cfg_path = repo_root / ".remote-run.toml"
    if not cfg_path.is_file():
        return empty_project_config()
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)
    label = str(cfg_path)
    check_keys(raw, PROJECT_TOP_KEYS, label)
    check_schema_version(raw, label)

    raw_path = raw.get("path", [])
    if not isinstance(raw_path, list):
        raise ValueError(f"{label} path must be a list of workspace-relative directories")
    proj_path = [rel_path(x, f"{label} path") for x in raw_path]
    for p in proj_path:
        if ":" in p:
            raise ValueError(f"{label} path entry cannot contain ':': {p!r}")

    proj_env = check_env_map(raw.get("env", {}), f"{label} [env]", reject_reserved=True, gpu_managed=False)

    ws_raw = raw.get("workspace", {})
    check_keys(ws_raw, PROJECT_WORKSPACE_KEYS, f"{label} [workspace]")
    shared_raw = ws_raw.get("shared", [])
    if not isinstance(shared_raw, list):
        raise ValueError(f"{label} [workspace].shared must be a list of relative directory paths")
    shared = _check_no_overlaps(shared_raw, f"{label} [workspace].shared")

    sync_raw = raw.get("sync", {})
    check_keys(sync_raw, PROJECT_SYNC_KEYS, f"{label} [sync]")
    ex_raw = sync_raw.get("exclude", [])
    if not isinstance(ex_raw, list):
        raise ValueError(f"{label} [sync].exclude must be a list of patterns")
    sync_exclude = [validate_pattern(p, f"{label} [sync].exclude") for p in ex_raw]

    pres_raw = sync_raw.get("preserve", [])
    if not isinstance(pres_raw, list):
        raise ValueError(f"{label} [sync].preserve must be a list of relative directories")
    preserve = _check_no_overlaps(pres_raw, f"{label} [sync].preserve")
    _check_no_overlaps(shared + preserve, f"{label} [workspace].shared and [sync].preserve")

    max_untracked = sync_raw.get("max_untracked_mib", 256)
    if isinstance(max_untracked, bool) or not isinstance(max_untracked, int) or max_untracked < 0:
        raise ValueError(f"{label} [sync].max_untracked_mib must be a non-negative integer")

    pull_raw = raw.get("pull", {})
    check_keys(pull_raw, PROJECT_PULL_KEYS, f"{label} [pull]")
    max_file_mib = pull_raw.get("max_file_mib", 10)
    if isinstance(max_file_mib, bool) or not isinstance(max_file_mib, int) or max_file_mib < 0:
        raise ValueError(f"{label} [pull].max_file_mib must be a non-negative integer")
    pull_ex_raw = pull_raw.get("exclude", [])
    if not isinstance(pull_ex_raw, list):
        raise ValueError(f"{label} [pull].exclude must be a list of patterns")
    pull_exclude = [validate_pattern(p, f"{label} [pull].exclude") for p in pull_ex_raw]

    return {
        "present": True,
        "file": str(cfg_path),
        "project": identifier(raw["project"], "project") if "project" in raw else "",
        "path": proj_path,
        "env": proj_env,
        "workspace": {"shared": shared},
        "sync": {"exclude": sync_exclude, "preserve": preserve, "max_untracked_mib": max_untracked},
        "pull": {"max_file_mib": max_file_mib, "exclude": pull_exclude},
    }


# === Repository / Worktree Identity & Layer 3 Private Bindings ===

def resolve_repo_context(repo_dir: Path) -> dict:
    target = repo_dir.resolve()
    if not target.is_dir():
        raise ValueError(f"Directory does not exist: {target}")
    if target == Path("/") or Path.home().resolve().is_relative_to(target):
        raise ValueError(f"Refusing unsafe local project directory: {target}")
    if ".git" in target.parts or (
        (target / "HEAD").is_file() and (target / "objects").is_dir() and (target / "config").is_file()
    ):
        raise ValueError(f"Refusing to use Git metadata directory as a project: {target}")

    ok, git_top = git_query(["rev-parse", "--show-toplevel"], cwd=target)
    common_dir: Path | None = None
    git_dir: Path | None = None
    is_linked_worktree = False
    if ok and git_top:
        root = Path(git_top).resolve()
        try:
            rel = target.relative_to(root).as_posix()
        except ValueError:
            rel = ""
        in_git = True
        ok_gd, gd_out = git_query(["rev-parse", "--git-dir"], cwd=root)
        ok_cd, cd_out = git_query(["rev-parse", "--git-common-dir"], cwd=root)
        if ok_gd and gd_out and ok_cd and cd_out:
            gp, cp = Path(gd_out), Path(cd_out)
            git_dir = (root / gp).resolve() if not gp.is_absolute() else gp.resolve()
            common_dir = (root / cp).resolve() if not cp.is_absolute() else cp.resolve()
            is_linked_worktree = git_dir != common_dir
    else:
        _, bare = git_query(["rev-parse", "--is-bare-repository"], cwd=target)
        if bare == "true":
            raise ValueError(f"Refusing to use a bare Git repository as a project: {target}")
        root, rel, in_git = target, "", False

    if Path.home().resolve().is_relative_to(root):
        raise ValueError(f"Refusing filesystem root or local home directory as project root: {root}")

    repo_key_source = common_dir if (in_git and common_dir is not None) else root
    repo_key = sha256(os.fsencode(repo_key_source)).hexdigest()
    checkout_key = sha256(os.fsencode(root)).hexdigest()
    return {
        "repo_root": root,
        "rel_subdir": "" if rel == "." else rel,
        "in_git": in_git,
        "git_dir": git_dir,
        "common_dir": common_dir,
        "is_linked_worktree": is_linked_worktree,
        "repo_binding_key": repo_key,
        "checkout_binding_key": checkout_key,
    }


def binding_path(scope: str, key: str) -> Path:
    if scope not in ("repo", "checkout"):
        raise ValueError(f"Binding scope must be 'repo' or 'checkout', got {scope!r}")
    return bindings_dir() / f"{scope}-{key}.toml"


def load_binding(scope: str, key: str) -> dict:
    path = binding_path(scope, key)
    if not path.is_file():
        return {
            "present": False, "scope": scope, "file": str(path),
            "host": "", "host_endpoint": "", "workspace": "", "gpus": None, "note": "", "path": [], "env": {},
        }
    with path.open("rb") as f:
        raw = tomllib.load(f)
    label = str(path)
    allowed = REPO_BINDING_KEYS if scope == "repo" else CHECKOUT_BINDING_KEYS
    check_keys(raw, allowed, label)
    check_schema_version(raw, label)

    host = identifier(raw["host"], f"{label} host") if "host" in raw else ""
    host_ep = raw.get("host_endpoint", "")
    if not isinstance(host_ep, str) or any(ord(c) < 32 for c in host_ep):
        raise ValueError(f"{label} host_endpoint must be a clean string")
    workspace = identifier(raw["workspace"], f"{label} workspace") if "workspace" in raw else ""
    gpus = validate_gpu_indices(raw["gpus"], f"{label} gpus") if "gpus" in raw else None
    note = raw.get("note", "")
    if not isinstance(note, str) or any(ord(c) < 32 for c in note):
        raise ValueError(f"{label} note must be a single-line string without control chars")
    raw_path = raw.get("path", [])
    if not isinstance(raw_path, list):
        raise ValueError(f"{label} path must be a list of paths")
    bpath = [validate_path_entry(x, f"{label} path", allow_relative=True) for x in raw_path]
    benv = check_env_map(raw.get("env", {}), f"{label} [env]", reject_reserved=True, gpu_managed=False)

    # Host-bound settings (gpus / path / env / note) require explicit host AND host_endpoint,
    # even when explicitly specified as empty in the TOML file.
    has_host_fields = any(k in raw for k in ("gpus", "path", "env", "note"))
    if has_host_fields and (not host or not host_ep):
        raise ValueError(
            f"{label} sets host-specific fields (gpus/path/env/note) and requires both 'host' and 'host_endpoint'"
        )

    return {
        "present": True,
        "scope": scope,
        "file": str(path),
        "host": host,
        "host_endpoint": host_ep,
        "workspace": workspace,
        "gpus": gpus,
        "note": note,
        "path": bpath,
        "env": benv,
    }


def render_binding_toml(data: dict, scope: str) -> str:
    lines = [f"schema_version = {SCHEMA_VERSION}"]
    if data.get("host"):
        lines.append(f"host = {toml_string(data['host'])}")
    if data.get("host_endpoint"):
        lines.append(f"host_endpoint = {toml_string(data['host_endpoint'])}")
    if scope == "checkout" and data.get("workspace"):
        lines.append(f"workspace = {toml_string(data['workspace'])}")
    if data.get("gpus") is not None:
        lines.append(f"gpus = {toml_int_list(data['gpus'])}")
    if data.get("note"):
        lines.append(f"note = {toml_string(data['note'])}")
    if data.get("path"):
        lines.append(f"path = {toml_str_list(data['path'])}")
    if data.get("env"):
        lines.append("")
        lines.append("[env]")
        for k, v in data["env"].items():
            lines.append(f"{toml_key(k)} = {toml_string(v)}")
    return "\n".join(lines).rstrip() + "\n"


def update_binding_atomic(scope: str, key: str, mutator) -> tuple[Path, dict]:
    """Read, validate, mutate, and write a binding file atomically under an exclusive flock."""
    path = binding_path(scope, key)
    bdir = path.parent
    check_directory_chain(bdir, "Bindings directory", trusted=True)
    bdir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open_lock(bdir / f".{path.name}.lock") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_symlink():
            raise ValueError(f"Refusing to update symlink binding file: {path}")
        existing = load_binding(scope, key)
        updated = mutator(existing)
        write_private_text(path, render_binding_toml(updated, scope))
    return path, updated


def delete_binding(scope: str, key: str) -> tuple[Path, bool]:
    path = binding_path(scope, key)
    if not path.parent.is_dir():
        return path, False
    check_directory_chain(path.parent, "Bindings directory", trusted=True)
    with open_lock(path.parent / f".{path.name}.lock") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_symlink():
            raise ValueError(f"Refusing to unlink symlink binding file: {path}")
        existed = path.is_file()
        path.unlink(missing_ok=True)
    return path, existed


# === Workspace & Complete Layered Resolution ===

def resolve_workspace_info(
    repo_ctx: dict,
    proj_cfg: dict,
    host: dict,
    *,
    workspace_override: str = "",
    checkout_workspace: str = "",
    inspect_dirty: bool = True,
) -> dict:
    repo_root: Path = repo_ctx["repo_root"]
    in_git: bool = repo_ctx["in_git"]
    is_linked: bool = repo_ctx["is_linked_worktree"]
    git_dir: Path | None = repo_ctx["git_dir"]
    common_dir: Path | None = repo_ctx["common_dir"]

    root_commit = ""
    git_commit = ""
    git_branch = ""
    git_dirty = False
    if in_git:
        ok_rc, rc_out = git_query(["rev-list", "--max-parents=0", "HEAD"], cwd=repo_root)
        if ok_rc and rc_out:
            root_commit = sorted(rc_out.splitlines())[0][:8]
        ok_hd, hd_out = git_query(["rev-parse", "HEAD"], cwd=repo_root)
        if ok_hd and hd_out:
            git_commit = hd_out
        if inspect_dirty:
            ok_st, st_out = git_query(["status", "--porcelain"], cwd=repo_root)
            if ok_st:
                git_dirty = bool(st_out.strip())
        ok_br, br_out = git_query(["symbolic-ref", "--short", "HEAD"], cwd=repo_root)
        if ok_br and br_out:
            git_branch = br_out
        elif git_commit:
            git_branch = f"detached-{git_commit[:8]}"
        else:
            git_branch = "unborn"

    if proj_cfg["project"]:
        project_id = proj_cfg["project"]
        project_origin = "project_config"
    else:
        src = common_dir if (in_git and common_dir is not None) else repo_root
        project_id = "p-" + sha256(os.fsencode(src)).hexdigest()[:24]
        project_origin = "derived"

    if workspace_override:
        workspace_id = identifier(workspace_override, "workspace")
        ws_origin = "cli"
    elif checkout_workspace:
        workspace_id = identifier(checkout_workspace, "workspace")
        ws_origin = "checkout_binding"
    elif in_git and is_linked and git_dir is not None and common_dir is not None:
        rel_meta = git_dir.relative_to(common_dir).as_posix()
        workspace_id = "wt-" + sha256(os.fsencode(rel_meta)).hexdigest()[:24]
        ws_origin = "derived_worktree"
    else:
        workspace_id = "default"
        ws_origin = "default"

    workdir = join_remote_path(host["root"], "projects", project_id, "workspaces", workspace_id)
    shared_dir = join_remote_path(host["root"], "projects", project_id, "shared")
    return {
        "project_id": project_id,
        "project_origin": project_origin,
        "workspace_id": workspace_id,
        "workspace_origin": ws_origin,
        "managed": True,
        "workdir": workdir,
        "shared_dir": shared_dir,
        "rel_subdir": repo_ctx["rel_subdir"],
        "in_git": in_git,
        "is_linked_worktree": is_linked,
        "root_commit": root_commit,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
    }


def resolve_effective_config(
    config_path: Path,
    *,
    repo_dir: Path | None = None,
    cli_host: str | None = None,
    workspace_override: str = "",
    cli_gpus: list[int] | None = None,
    cli_no_gpu: bool = False,
    cli_paths: list[str] = (),
    cli_env: dict[str, str] = None,
    max_untracked_mib: int | None = None,
    inspect_dirty: bool = False,
) -> dict:
    """Resolve and merge all 4 hierarchy levels with complete per-field provenance."""
    if cli_no_gpu and cli_gpus is not None:
        raise ValueError("Cannot combine --no-gpu with --gpus")
    default_host, hosts = load_hosts_config(config_path)

    repo_ctx = resolve_repo_context(repo_dir) if repo_dir is not None else None
    proj_cfg = load_project_config(repo_ctx["repo_root"]) if repo_ctx is not None else empty_project_config()
    repo_bind = (
        load_binding("repo", repo_ctx["repo_binding_key"]) if repo_ctx is not None
        else {"present": False, "scope": "repo", "file": None, "host": "", "workspace": "", "gpus": None, "note": "", "path": [], "env": {}}
    )
    checkout_bind = (
        load_binding("checkout", repo_ctx["checkout_binding_key"]) if repo_ctx is not None
        else {"present": False, "scope": "checkout", "file": None, "host": "", "workspace": "", "gpus": None, "note": "", "path": [], "env": {}}
    )

    host, host_origin = select_host_from_catalog(
        default_host, hosts,
        cli_host=cli_host,
        checkout_host=checkout_bind["host"] or None,
        repo_host=repo_bind["host"] or None,
    )
    gpu_managed = host["gpu"] is not None

    # Bindings only contribute host-bound fields when targeting the SAME host alias AND endpoint identity!
    target_ep = host_endpoint_id(host)
    skipped_bindings: list[dict] = []

    def _binding_matches_host(b: dict) -> tuple[bool, str]:
        if not b["present"]:
            return False, ""
        if not b["host"]:
            return True, ""
        if b["host"] != host["name"]:
            return False, f"selected host {host['name']!r} differs from bound host {b['host']!r}"
        # Require matching host_endpoint when host-bound fields exist so endpointless/stale bindings never cross targets.
        has_host_bound = bool(b.get("gpus") is not None or b.get("path") or b.get("env") or b.get("note"))
        if has_host_bound and b.get("host_endpoint") != target_ep:
            return False, f"host {host['name']!r} endpoint does not match binding (re-run `remote-run bind`)"
        if b.get("host_endpoint") and b["host_endpoint"] != target_ep:
            return False, f"host {host['name']!r} endpoint changed since binding was saved"
        return True, ""

    repo_bind_active, repo_skip_reason = _binding_matches_host(repo_bind)
    if repo_bind["present"] and repo_bind["host"] and not repo_bind_active:
        skipped_bindings.append({
            "scope": "repo", "file": repo_bind["file"], "bound_host": repo_bind["host"], "reason": repo_skip_reason,
        })
    checkout_bind_active, co_skip_reason = _binding_matches_host(checkout_bind)
    if checkout_bind["present"] and checkout_bind["host"] and not checkout_bind_active:
        skipped_bindings.append({
            "scope": "checkout", "file": checkout_bind["file"], "bound_host": checkout_bind["host"], "reason": co_skip_reason,
        })

    # Workspace resolution: checkout binding's workspace is checkout-scoped (portable across hosts).
    ws_info = (
        resolve_workspace_info(
            repo_ctx, proj_cfg, host,
            workspace_override=workspace_override,
            checkout_workspace=checkout_bind["workspace"] if checkout_bind["present"] else "",
            inspect_dirty=inspect_dirty,
        )
        if repo_ctx is not None
        else None
    )

    # Validate invocation PATH & env.
    norm_cli_paths = [
        validate_path_entry(p, "invocation.path", allow_relative=repo_ctx is not None)
        for p in cli_paths
    ]
    norm_cli_env = check_env_map(cli_env or {}, "invocation.env", reject_reserved=True, gpu_managed=gpu_managed)

    # Validate project/binding env against GPU-managed host constraint.
    if gpu_managed:
        check_env_map(proj_cfg["env"], ".remote-run.toml [env]", reject_reserved=True, gpu_managed=True)
        if repo_bind_active:
            check_env_map(repo_bind["env"], "repo binding [env]", reject_reserved=True, gpu_managed=True)
        if checkout_bind_active:
            check_env_map(checkout_bind["env"], "checkout binding [env]", reject_reserved=True, gpu_managed=True)

    # PATH layers ordered highest -> lowest priority:
    # invocation -> checkout_binding -> repo_binding -> project -> host -> (ssh_base on worker)
    path_layers: list[dict] = []
    for entry in norm_cli_paths:
        path_layers.append({"source": "invocation", "entry": entry, "allow_pending": False})
    if checkout_bind_active:
        for entry in checkout_bind["path"]:
            path_layers.append({"source": "checkout_binding", "entry": entry, "allow_pending": not (entry.startswith("/") or entry.startswith("~"))})
    if repo_bind_active:
        for entry in repo_bind["path"]:
            path_layers.append({"source": "repo_binding", "entry": entry, "allow_pending": not (entry.startswith("/") or entry.startswith("~"))})
    for entry in proj_cfg["path"]:
        path_layers.append({"source": "project", "entry": entry, "allow_pending": True})
    for entry in host["path"]:
        path_layers.append({"source": "host", "entry": entry, "allow_pending": False})

    # Config env merge (lowest -> highest): host -> project -> repo_binding -> checkout_binding
    config_env: dict[str, str] = {}
    env_origins: dict[str, str] = {}
    for layer_name, mapping in (
        ("host", host["env"]),
        ("project", proj_cfg["env"]),
        ("repo_binding", repo_bind["env"] if repo_bind_active else {}),
        ("checkout_binding", checkout_bind["env"] if checkout_bind_active else {}),
    ):
        for k, v in mapping.items():
            config_env[k] = v
            env_origins[k] = layer_name
    for k in norm_cli_env:
        env_origins[k] = "invocation"

    # Note (checkout_binding > repo_binding)
    note = ""
    note_origin = ""
    if checkout_bind_active and checkout_bind["note"]:
        note, note_origin = checkout_bind["note"], "checkout_binding"
    elif repo_bind_active and repo_bind["note"]:
        note, note_origin = repo_bind["note"], "repo_binding"

    # GPU selection: --no-gpu / --gpus -> checkout_binding -> repo_binding -> [] (CPU default)
    if cli_no_gpu:
        gpus, gpus_origin = [], "cli_no_gpu"
    elif cli_gpus is not None:
        gpus, gpus_origin = validate_gpu_indices(cli_gpus, "gpus"), "cli"
    elif checkout_bind_active and checkout_bind["gpus"] is not None:
        gpus, gpus_origin = list(checkout_bind["gpus"]), "checkout_binding"
    elif repo_bind_active and repo_bind["gpus"] is not None:
        gpus, gpus_origin = list(repo_bind["gpus"]), "repo_binding"
    else:
        gpus, gpus_origin = [], "default_cpu"

    if gpus and not gpu_managed:
        raise ValueError(f"Host {host['name']!r} has no [gpu] configured; cannot request GPUs {gpus}")
    if gpus and gpu_managed:
        declared = host["gpu"]["devices"]
        for idx in gpus:
            if idx not in declared:
                raise ValueError(
                    f"Requested GPU {idx} (from {gpus_origin}) is not in hosts.{host['name']}.gpu.devices {declared}"
                )

    eff_max_untracked = (
        max_untracked_mib if max_untracked_mib is not None else proj_cfg["sync"]["max_untracked_mib"]
    )
    if isinstance(eff_max_untracked, bool) or not isinstance(eff_max_untracked, int) or eff_max_untracked < 0:
        raise ValueError("max_untracked_mib must be a non-negative integer")

    sync_policy = {
        "exclude": list(proj_cfg["sync"]["exclude"]),
        "preserve": list(proj_cfg["sync"]["preserve"]),
        "shared": list(proj_cfg["workspace"]["shared"]),
        "max_untracked_mib": eff_max_untracked,
    }
    pull_policy = {
        "max_file_mib": proj_cfg["pull"]["max_file_mib"],
        "exclude": list(proj_cfg["pull"]["exclude"]),
    }

    return {
        "config_path": str(config_path),
        "host": host,
        "host_origin": host_origin,
        "repo_ctx": repo_ctx,
        "project_config": proj_cfg,
        "bindings": {
            "repo": {**repo_bind, "active": repo_bind_active},
            "checkout": {**checkout_bind, "active": checkout_bind_active},
            "skipped": skipped_bindings,
        },
        "workspace": ws_info,
        "path_layers": path_layers,
        "config_env": config_env,
        "invocation_env": norm_cli_env,
        "env_origins": env_origins,
        "note": note,
        "note_origin": note_origin,
        "gpus": gpus,
        "gpus_origin": gpus_origin,
        "sync_policy": sync_policy,
        "pull_policy": pull_policy,
    }


def redact_config_view(resolved: dict) -> dict:
    """Public redacted representation returned by `remote-run config` and `remote_config`."""
    host = resolved["host"]
    repo_bind = resolved["bindings"]["repo"]
    co_bind = resolved["bindings"]["checkout"]
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": resolved["config_path"],
        "repo_root": str(resolved["repo_ctx"]["repo_root"]) if resolved["repo_ctx"] else None,
        "host": {
            "name": host["name"],
            "origin": resolved["host_origin"],
            "ssh": host["ssh"],
            "port": host["port"],
            "root": host["root"],
            "gpu_devices": host["gpu"]["devices"] if host["gpu"] else [],
        },
        "workspace": resolved["workspace"],
        "files_loaded": {
            "hosts": resolved["config_path"],
            "project": resolved["project_config"]["file"],
            "repo_binding": {"file": repo_bind["file"], "present": repo_bind["present"], "active": repo_bind["active"]},
            "checkout_binding": {"file": co_bind["file"], "present": co_bind["present"], "active": co_bind["active"]},
            "skipped_bindings": resolved["bindings"]["skipped"],
        },
        "environment": {
            "note": resolved["note"],
            "note_origin": resolved["note_origin"],
            "path_layers": [
                *resolved["path_layers"],
                {"source": "ssh_base", "entry": "<remote noninteractive SSH PATH>", "allow_pending": False},
            ],
            "env_keys": {k: {"origin": origin, "value": "<redacted>"} for k, origin in sorted(resolved["env_origins"].items())},
        },
        "resources": {
            "gpu_managed": host["gpu"] is not None,
            "gpus": resolved["gpus"],
            "gpus_origin": resolved["gpus_origin"],
        },
        "sync": resolved["sync_policy"],
        "pull": resolved["pull_policy"],
    }
