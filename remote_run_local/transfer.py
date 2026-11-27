"""Component 2: Single-manifest Git/plain sync, untracked budget preflight, rsync driver, staged pull."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, suppress
import ctypes
import functools
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tempfile

from remote_worker import (
    CONTROL_ROOT_FILES,
    file_snapshot,
    match_pattern,
    rel_path,
    workspace_file,
)
from remote_run_local.config import bindings_dir, git_query, xdg_config_root
from remote_run_local.transport import build_ssh_argv, run_local, ssh_call, workspace_lease

HARD_EXCLUDED_COMPONENTS = {".git", ".ssh", ".aws", ".env", ".env.local"}
ENV_LOCAL_RE = re.compile(r"^\.env\.[^/]+\.local$")


def is_hard_excluded(
    rel: str,
    *,
    extra_blocked: set[str] = frozenset(),
    extra_blocked_prefixes: tuple[str, ...] = (),
) -> bool:
    norm = rel.strip("/")
    if not norm:
        return True
    if norm in extra_blocked or norm in CONTROL_ROOT_FILES:
        return True
    if any(norm == p or norm.startswith(p + "/") for p in extra_blocked_prefixes):
        return True
    parts = norm.split("/")
    if parts[0].startswith(".remote-run-gc-"):
        return True
    if norm == ".remote-run.toml":
        return True
    for part in parts:
        if part in HARD_EXCLUDED_COMPONENTS or ENV_LOCAL_RE.fullmatch(part):
            return True
    return False


def _matches_prefix_tree(rel: str, prefixes: list[str]) -> bool:
    return any(rel == p or rel.startswith(p + "/") for p in prefixes)


@functools.lru_cache(maxsize=1)
def rsync_supports_protect_args() -> bool:
    completed = subprocess.run(["rsync", "--help"], capture_output=True, text=True)
    help_text = completed.stdout + completed.stderr
    return "--protect-args" in help_text or "--secluded-args" in help_text


def rsync_protect_args(peer: dict | None = None) -> list[str]:
    return ["-s"] if rsync_supports_protect_args() and (peer is None or peer.get("secluded")) else []


def rsync_remote_spec(ssh_host: str, remote_dir: str, peer: dict | None = None) -> str:
    path_str = f"{remote_dir.rstrip('/')}/"
    if rsync_protect_args(peer):
        return f"{ssh_host}:{path_str}"
    help_text = run_local(["rsync", "--help"])
    modern = "--old-args" in help_text
    return f"{ssh_host}:{path_str if modern else shlex.quote(path_str)}"


def rsync_path_flag(peer: dict | None) -> list[str]:
    return ["--rsync-path", shlex.quote(peer["path"])] if (peer and peer.get("path")) else []


def iter_rsync_itemize(stdout: str):
    for line in stdout.split("\n"):
        if line[11:12] == " " and len(line) > 12:
            raw = os.fsencode(line[12:])
            path = re.sub(rb"\\#([0-7]{3})", lambda match: bytes([int(match[1], 8)]), raw)
            yield line[:11], os.fsdecode(path)


def parse_rsync_itemize(stdout: str) -> tuple[list[str], list[str]]:
    updated, deleted = [], []
    for code, path in iter_rsync_itemize(stdout):
        if code.startswith("*deleting"):
            deleted.append(path)
        elif code[0] in "<>ch":
            updated.append(path)
    return updated, deleted


def _git_collect_tree(repo_root: Path, checkout: Path) -> tuple[list[str], list[str], list[str]]:
    """Collect (tracked, untracked, uninitialized_submodules) relative to repo_root."""
    base = checkout.relative_to(repo_root).as_posix()
    prefix = "" if base == "." else base + "/"

    ok_t, out_t = git_query(["ls-files", "-z", "--stage"], cwd=checkout, strip=False)
    if not ok_t:
        raise ValueError(f"Could not read Git index for {checkout}")

    tracked: list[str] = []
    submodules: list[Path] = []
    uninit_submodules: list[str] = []
    seen_t: set[str] = set()
    for rec in out_t.split("\0"):
        if not rec or "\t" not in rec:
            continue
        meta, raw_rel = rec.split("\t", 1)
        rel_in_co = rel_path(raw_rel, "Git tracked path")
        full_rel = prefix + rel_in_co
        if meta.startswith("160000 "):
            sub_dir = checkout / rel_in_co
            if sub_dir.is_dir() and (sub_dir / ".git").exists():
                ok_top, top = git_query(["rev-parse", "--show-toplevel"], cwd=sub_dir)
                if not ok_top or Path(top).resolve() != sub_dir.resolve():
                    raise ValueError(f"Invalid Git submodule checkout: {sub_dir}")
                submodules.append(sub_dir)
            else:
                uninit_submodules.append(full_rel)
            continue
        if full_rel not in seen_t:
            seen_t.add(full_rel)
            tracked.append(full_rel)

    ok_u, out_u = git_query(["ls-files", "-z", "--others", "--exclude-standard"], cwd=checkout, strip=False)
    if not ok_u:
        raise ValueError(f"Could not enumerate untracked files for {checkout}")
    untracked: list[str] = []
    for raw_rel in filter(None, out_u.split("\0")):
        rel_in_co = rel_path(raw_rel, "Git untracked path")
        untracked.append(prefix + rel_in_co)

    for sub in submodules:
        st, su, sunit = _git_collect_tree(repo_root, sub)
        tracked.extend(st)
        untracked.extend(su)
        uninit_submodules.extend(sunit)

    return tracked, untracked, uninit_submodules


def _plain_collect_tree(repo_root: Path) -> list[str]:
    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root, followlinks=False):
        dp = Path(dirpath)
        rel_dir = "" if dp == repo_root else dp.relative_to(repo_root).as_posix()
        # Prune hard-excluded directories before descending.
        kept_dirs: list[str] = []
        for d in sorted(dirnames):
            cand_rel = f"{rel_dir}/{d}" if rel_dir else d
            full = dp / d
            if full.is_symlink():
                candidates.append(cand_rel)
                continue
            if not is_hard_excluded(cand_rel):
                kept_dirs.append(d)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            cand_rel = f"{rel_dir}/{name}" if rel_dir else name
            candidates.append(cand_rel)
    return candidates


def _validate_safe_relative_symlink(repo_root: Path, rel: str, target_raw: str) -> None:
    if os.path.isabs(target_raw):
        raise ValueError(f"Refusing absolute symlink in sync manifest: {rel!r} -> {target_raw!r}")
    parent = PurePosixPath(rel).parent
    combined = PurePosixPath(os.path.normpath((parent / target_raw).as_posix()))
    if combined.is_absolute() or ".." in combined.parts:
        raise ValueError(f"Refusing escaping symlink in sync manifest: {rel!r} -> {target_raw!r}")


def _in_repo_private_prefixes(repo_root: Path) -> tuple[str, ...]:
    """Return relative prefixes for any XDG/bindings state directories or remote-run private state inside repo_root."""
    prefixes: list[str] = []
    candidates = [xdg_config_root(), bindings_dir(), Path.home() / ".config" / "remote-run"]
    for candidate in candidates:
        try:
            res = candidate.resolve()
        except OSError:
            continue
        if res.is_relative_to(repo_root) and res != repo_root:
            rel = res.relative_to(repo_root).as_posix().strip("/")
            if rel and rel not in prefixes:
                prefixes.append(rel)
    return tuple(prefixes)


_PRIVATE_BINDING_FILE_RE = re.compile(r"^(?:repo|checkout)-[0-9a-f]{64}\.toml(?:\.lock)?$")


def _is_private_remote_run_state(rel: str) -> bool:
    """Catch private remote-run XDG binding/catalog files anywhere in a repo across XDG contexts."""
    parts = rel.strip("/").split("/")
    for i, part in enumerate(parts):
        if _PRIVATE_BINDING_FILE_RE.fullmatch(part) or (part.startswith(".") and _PRIVATE_BINDING_FILE_RE.fullmatch(part[1:])):
            return True
        if part == "remote-run" and i + 1 < len(parts) and parts[i + 1] in ("hosts.toml", "bindings"):
            return True
    return False


def build_sync_manifest(
    repo_root: Path,
    *,
    in_git: bool,
    sync_policy: dict,
    config_files_in_repo: set[str] = frozenset(),
) -> dict:
    """Build single explicit sender manifest + preflight untracked byte budget.

    Rules:
    - ZERO language-specific BUILTIN_IGNORE or BUILTIN_PRESERVE.
    - Hard-excluded security paths (including in-repo XDG config/bindings) are always omitted.
    - Tracked files colliding with preserve/shared raise immediately (protects remote-owned trees).
    - Sync exclude patterns apply uniformly on the sender (and report omitted tracked files).
    - Untracked regular files are summed against sync.max_untracked_mib before any transfer.
    """
    repo_root = repo_root.resolve()
    private_prefixes = _in_repo_private_prefixes(repo_root)
    excludes: list[str] = list(sync_policy.get("exclude", []))
    preserves: list[str] = list(sync_policy.get("preserve", []))
    shared_dirs: list[str] = list(sync_policy.get("shared", []))
    reserved_dirs = list(dict.fromkeys([*preserves, *shared_dirs]))
    max_untracked_mib = int(sync_policy.get("max_untracked_mib", 256))

    uninit_submodules: list[str] = []
    if in_git:
        tracked_raw, untracked_raw, uninit_submodules = _git_collect_tree(repo_root, repo_root)
    else:
        tracked_raw = []
        untracked_raw = _plain_collect_tree(repo_root)

    # Reject tracked paths at, below, or obstructing any reserved (preserve/shared) directory.
    for rel in tracked_raw:
        for rdir in reserved_dirs:
            if rel == rdir or rel.startswith(rdir + "/") or rdir.startswith(rel + "/"):
                raise ValueError(
                    f"Tracked path {rel!r} conflicts with reserved remote directory {rdir!r} "
                    "(remove it from Git or from [sync].preserve / [workspace].shared)"
                )

    files_to_send: list[str] = []
    blocked_tracked: list[str] = []
    excluded_tracked: list[str] = []
    untracked_bytes = 0
    untracked_dir_bytes: dict[str, int] = defaultdict(int)
    seen_send: set[str] = set()

    def process_candidate(rel: str, *, is_tracked: bool) -> None:
        nonlocal untracked_bytes
        if rel in seen_send:
            return
        if is_hard_excluded(rel, extra_blocked=config_files_in_repo, extra_blocked_prefixes=private_prefixes) or _is_private_remote_run_state(rel):
            if is_tracked:
                blocked_tracked.append(rel)
            return
        if _matches_prefix_tree(rel, reserved_dirs):
            return
        if any(match_pattern(rel, pat) for pat in excludes):
            if is_tracked:
                excluded_tracked.append(rel)
            return

        full = repo_root / rel
        try:
            st = full.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISDIR(st.st_mode):
            return
        if stat.S_ISLNK(st.st_mode):
            _validate_safe_relative_symlink(repo_root, rel, os.readlink(full))
            seen_send.add(rel)
            files_to_send.append(rel)
            return
        if not stat.S_ISREG(st.st_mode):
            return

        # Reject any symlink parent directory inside the repo tree.
        cur = repo_root
        for part in PurePosixPath(rel).parts[:-1]:
            cur = cur / part
            if cur.is_symlink():
                raise ValueError(f"Refusing path under symlink directory in sync manifest: {rel!r}")

        seen_send.add(rel)
        files_to_send.append(rel)
        if not is_tracked:
            untracked_bytes += st.st_size
            top = rel.split("/", 1)[0] if "/" in rel else "."
            untracked_dir_bytes[top] += st.st_size

    for rel in sorted(tracked_raw):
        process_candidate(rel, is_tracked=True)
    for rel in sorted(untracked_raw):
        process_candidate(rel, is_tracked=False)

    budget_bytes = max_untracked_mib * 1024 * 1024
    budget_exceeded = budget_bytes > 0 and untracked_bytes > budget_bytes
    top_dirs = [
        {"dir": d, "bytes": b}
        for d, b in sorted(untracked_dir_bytes.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    ]
    if budget_exceeded:
        summary = ", ".join(f"{x['dir']} ({x['bytes'] / (1024 * 1024):.1f} MiB)" for x in top_dirs[:5])
        raise ValueError(
            f"Untracked upload size {untracked_bytes / (1024 * 1024):.1f} MiB exceeds "
            f"max_untracked_mib={max_untracked_mib} MiB (largest dirs: {summary}). "
            "Add build/cache dirs to [sync].exclude or [sync].preserve, or pass --max-untracked-mib."
        )

    return {
        "files": files_to_send,
        "file_count": len(files_to_send),
        "untracked_bytes": untracked_bytes,
        "max_untracked_mib": max_untracked_mib,
        "top_untracked_dirs": top_dirs,
        "blocked_tracked": blocked_tracked,
        "excluded_tracked": excluded_tracked,
        "uninitialized_submodules": uninit_submodules,
        "protected_prefixes": reserved_dirs,
    }


def prune_ghost_candidates(
    repo_root: Path,
    manifest_files: set[str],
    previous_owned: list[str],
    sync_policy: dict,
    *,
    config_files_in_repo: set[str] = frozenset(),
) -> list[str]:
    """Only schedule remote deletion for previously uploaded files that no longer exist locally.
    If a file still exists locally (e.g. newly excluded or newly gitignored), NEVER delete its remote copy.
    """
    reserved_dirs = list(dict.fromkeys([*sync_policy.get("preserve", []), *sync_policy.get("shared", [])]))
    excludes = list(sync_policy.get("exclude", []))
    candidates: list[str] = []
    for raw in previous_owned:
        try:
            rel = rel_path(raw, "synced file")
        except ValueError:
            continue
        if rel in manifest_files:
            continue
        if is_hard_excluded(rel, extra_blocked=config_files_in_repo):
            continue
        if _matches_prefix_tree(rel, reserved_dirs):
            continue
        if any(match_pattern(rel, pat) for pat in excludes):
            continue
        try:
            (repo_root / rel).lstat()
            # File still exists locally -> do NOT delete remotely.
            continue
        except FileNotFoundError:
            candidates.append(rel)
        except NotADirectoryError:
            candidates.append(rel)
    return sorted(candidates)


def sync_workspace(
    repo_root: Path,
    resolved: dict,
    *,
    run_id: str = "",
    dry_run: bool = False,
    lease=None,
) -> dict:
    host = resolved["host"]
    ws_info = resolved["workspace"]
    sync_policy = resolved["sync_policy"]
    repo_root = repo_root.resolve()

    cfg_files: set[str] = set()
    cfg_p = Path(resolved["config_path"]).resolve()
    if cfg_p.is_relative_to(repo_root):
        cfg_files.add(cfg_p.relative_to(repo_root).as_posix())

    manifest = build_sync_manifest(
        repo_root,
        in_git=bool(ws_info.get("in_git", False)),
        sync_policy=sync_policy,
        config_files_in_repo=cfg_files,
    )

    manifest_parents = sorted({
        PurePosixPath(p).parent.as_posix()
        for p in manifest["files"]
        if "/" in p
    })

    with ExitStack() as stack:
        if not dry_run and lease is None:
            lease = stack.enter_context(workspace_lease(host, ws_info["workdir"]))
        rpc = {"lease": lease} if lease is not None else {}
        prep = ssh_call(host, "prepare_workspace", {
            "workspace": ws_info,
            "shared_dirs": sync_policy["shared"],
            "run_id": run_id,
            "dry_run": dry_run,
            "for_sync": True,
            "manifest_parents": manifest_parents,
        }, **rpc)
        if prep.get("run_exists"):
            return {
                "host": host["name"],
                "workdir": prep["workdir"],
                "project_id": ws_info["project_id"],
                "workspace_id": ws_info["workspace_id"],
                "skipped_for_existing_run": True,
            }
        if prep.get("active_runs_warning") and not dry_run:
            runs = ", ".join(prep["active_runs_warning"])
            raise ValueError(f"Cannot sync a workspace used by active runs: {runs}")

        remote_workdir = prep["workdir"]
        peer = prep.get("rsync")
        pargs = rsync_protect_args(peer)
        file_list_bytes = b"\0".join(os.fsencode(p) for p in manifest["files"])
        if manifest["files"]:
            file_list_bytes += b"\0"

        # Exact NUL-delimited manifest transfer: NO implicit recursion (-r omitted), NO --delete, NO dereference (-l).
        # Use standard size+mtime (-ltz) so large repos sync fast without hashing every unchanged file on both hosts.
        # Parent symlink containment on receiver was verified by prepare_workspace under the workspace lease.
        rsync_opts = [
            "-ltz", "--executability", "--no-implied-dirs", "--relative",
            "--files-from=-", "--from0", *pargs,
            "--itemize-changes", "--out-format=%i %n",
        ]
        if dry_run and not prep.get("workspace_exists", True):
            with tempfile.TemporaryDirectory() as empty_dst:
                dry_cmd = ["rsync", *rsync_opts, "-n", f"{repo_root}/", f"{empty_dst}/"]
                out = run_local(dry_cmd, stdin=file_list_bytes, timeout=120, trim=False)
        else:
            dest = rsync_remote_spec(host["ssh"], remote_workdir, peer)
            rsync_cmd = ["rsync", *rsync_opts]
            if dry_run:
                rsync_cmd.append("-n")
            rsync_cmd.extend([
                "-e", shlex.join(build_ssh_argv(host)), *rsync_path_flag(peer),
                f"{repo_root}/", dest,
            ])
            out = run_local(rsync_cmd, stdin=file_list_bytes, timeout=120, trim=False, lease=lease)

        prune_candidates = prune_ghost_candidates(
            repo_root,
            set(manifest["files"]),
            prep.get("prev_synced_files", []),
            sync_policy,
            config_files_in_repo=cfg_files,
        )
        protected_prefixes = manifest["protected_prefixes"]

        res = {
            "host": host["name"],
            "workdir": remote_workdir,
            "project_id": ws_info["project_id"],
            "workspace_id": ws_info["workspace_id"],
            "dry_run": dry_run,
            "manifest_files": manifest["file_count"],
            "untracked_bytes": manifest["untracked_bytes"],
            "max_untracked_mib": manifest["max_untracked_mib"],
            "blocked_tracked": manifest["blocked_tracked"],
            "excluded_tracked": manifest["excluded_tracked"],
            "uninitialized_submodules": manifest["uninitialized_submodules"],
            "active_runs_warning": prep.get("active_runs_warning", []),
        }

        if dry_run:
            upd, _ = parse_rsync_itemize(out)
            res["would_update"] = upd
            if prune_candidates:
                would_prune = ssh_call(host, "reconcile_synced", {
                    "workdir": remote_workdir,
                    "prune_candidates": prune_candidates,
                    "protected_prefixes": protected_prefixes,
                    "dry_run": True,
                }).get("pruned_ghost_files", [])
                if would_prune:
                    res["would_prune_ghost"] = sorted(would_prune)
            return res

        uploaded_files: dict[str, str] = {}
        uploaded_symlinks: dict[str, str] = {}
        for code, path in iter_rsync_itemize(out):
            if code[:2] in (">f", "<f"):
                try:
                    rel = rel_path(path, "transferred file")
                    with workspace_file(repo_root, rel) as (directory, name):
                        uploaded_files[rel] = file_snapshot(directory, name)["sha256"]
                except (ValueError, FileNotFoundError, NotADirectoryError):
                    continue
            elif code[:2] in (">L", "<L", "cL"):
                try:
                    rel = rel_path(path, "transferred symlink")
                    with workspace_file(repo_root, rel) as (directory, name):
                        st_l = os.stat(name, dir_fd=directory, follow_symlinks=False)
                        if stat.S_ISLNK(st_l.st_mode):
                            uploaded_symlinks[rel] = os.readlink(name, dir_fd=directory)
                except (ValueError, FileNotFoundError, NotADirectoryError):
                    continue
        rec = ssh_call(host, "reconcile_synced", {
            "workdir": remote_workdir,
            "uploaded_files": uploaded_files,
            "uploaded_symlinks": uploaded_symlinks,
            "prune_candidates": prune_candidates,
            "protected_prefixes": protected_prefixes,
        }, **rpc)
        if rec.get("pruned_ghost_files"):
            res["pruned_ghost_files"] = rec["pruned_ghost_files"]
        return res


_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_DARWIN_RENAME_SWAP = 0x00000002
_DARWIN_RENAME_EXCL = 0x00000004


def _atomic_rename(src: Path, dst: Path, *, exchange: bool = False) -> bool:
    """Atomic non-replacing rename or in-place directory swap on Linux (renameat2) and macOS (renamex_np)."""
    s_b, d_b = os.fsencode(src), os.fsencode(dst)
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            fn = libc.renameat2
        except (OSError, AttributeError):
            return False
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        rc = fn(-100, s_b, -100, d_b, _RENAME_EXCHANGE if exchange else _RENAME_NOREPLACE)
        if rc == 0:
            return True
        err = ctypes.get_errno()
        if err in (38, 95, 22):  # ENOSYS, EOPNOTSUPP, EINVAL
            return False
        raise OSError(err, os.strerror(err), str(src), None, str(dst))
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
            fn = libc.renamex_np
        except (OSError, AttributeError):
            return False
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        rc = fn(s_b, d_b, _DARWIN_RENAME_SWAP if exchange else _DARWIN_RENAME_EXCL)
        if rc == 0:
            return True
        err = ctypes.get_errno()
        if err in (45, 78, 22):  # ENOTSUP, ENOSYS, EINVAL
            return False
        raise OSError(err, os.strerror(err), str(src), None, str(dst))
    return False


def pull_run_files(
    host: dict,
    run_id: str,
    destination: Path,
    *,
    max_file_mib: int | None = None,
    artifacts: list[str] = (),
    paths: list[str] = (),
    paths_only: bool = False,
    force: bool = False,
) -> dict:
    dest = Path(os.path.abspath(destination))
    if dest.is_symlink():
        raise ValueError(f"Refusing a symlink pull destination: {dest}")
    dest = dest.parent.resolve() / dest.name
    if Path.home().resolve().is_relative_to(dest):
        raise ValueError(f"Refusing dangerous pull destination: {dest}")
    cwd_res = Path.cwd().resolve()
    dest_check = dest.resolve() if dest.exists() else dest
    if cwd_res == dest_check or cwd_res.is_relative_to(dest_check):
        raise ValueError(f"Refusing to pull into or above current working directory: {dest}")
    if (dest.exists() or dest.is_symlink()) and not force:
        raise ValueError(f"Destination already exists (use a new directory or --force): {dest}")
    if max_file_mib is not None and (isinstance(max_file_mib, bool) or not isinstance(max_file_mib, int) or max_file_mib < 0):
        raise ValueError("max_file_mib must be a non-negative integer")
    if paths_only and not paths:
        raise ValueError("--paths-only requires at least one --path")
    if paths_only and artifacts:
        raise ValueError("--paths-only cannot be combined with --artifact")

    meta = ssh_call(host, "pull", {
        "run_id": run_id,
        "max_file_mib": max_file_mib,
        "artifacts": list(artifacts),
        "paths": list(paths),
        "paths_only": paths_only,
    })
    stage_id = meta.get("stage_id", "")
    stage_dir = meta.get("stage_dir", "")
    staged_files = list(meta.get("staged_files") or ())
    peer = meta.get("rsync")
    try:
        pargs = rsync_protect_args(peer)
        ssh_cmd = shlex.join(build_ssh_argv(host))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".remote-run-pull-", dir=dest.parent) as stage:
            contents = Path(stage) / "contents"
            contents.mkdir(mode=0o755)
            # Pull ONLY from the worker's pinned 0700 stage_dir (rsync never reopens live workspace/run paths).
            if staged_files and stage_dir:
                run_local([
                    "rsync", "-aR", "--no-links", *pargs, "--files-from=-", "--from0", "-e", ssh_cmd,
                    *rsync_path_flag(peer),
                    rsync_remote_spec(host["ssh"], stage_dir, peer), str(contents),
                ], stdin="\0".join(staged_files), timeout=300)

            if dest.exists() or dest.is_symlink():
                if not force:
                    raise ValueError(f"Destination already exists (use a new directory or --force): {dest}")
                if dest.is_symlink() or not dest.is_dir():
                    raise ValueError(f"Refusing --force into non-directory destination: {dest}")
                merged = Path(stage) / "merged"
                merged.mkdir(mode=0o700)
                run_local(["rsync", "-a", f"{dest}/", f"{merged}/"], timeout=120)
                if staged_files:
                    # Refuse writing through any existing symlink along a pulled path inside merged.
                    for rel in staged_files:
                        cur_p = merged
                        for part in rel.split("/"):
                            cur_p = cur_p / part
                            if cur_p.is_symlink():
                                raise ValueError(
                                    f"Refusing --force pull overwrite through existing symlink: {dest / rel}"
                                )
                    run_local([
                        "rsync", "-aIR", "--no-implied-dirs", "--files-from=-", "--from0",
                        f"{contents}/", f"{merged}/",
                    ], stdin="\0".join(staged_files), timeout=120)
                if not _atomic_rename(merged, dest, exchange=True):
                    raise RuntimeError(
                        f"Filesystem does not support atomic directory exchange for --force pull into {dest}; "
                        "pull into a new destination directory instead"
                    )
            else:
                try:
                    if not _atomic_rename(contents, dest, exchange=False):
                        raise RuntimeError(
                            f"Filesystem does not support atomic non-replacing rename for pull publication at {dest}"
                        )
                except FileExistsError as exc:
                    raise ValueError(f"Destination already exists (use a new directory or --force): {dest}") from exc
                except OSError as exc:
                    raise ValueError(f"Destination already exists or could not be published atomically: {dest}") from exc
    finally:
        if stage_id:
            with suppress(Exception):
                ssh_call(host, "pull_cleanup", {"stage_id": stage_id}, timeout=30)

    clean_meta = {k: v for k, v in meta.items() if k not in ("stage_id", "stage_dir", "staged_files")}
    result = {"run_id": run_id, "destination": str(dest), **clean_meta}
    if meta.get("workdir_files"):
        result["workdir_destination"] = str(dest / "workspace")
    return result
