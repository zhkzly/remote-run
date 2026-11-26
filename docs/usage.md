# remote-run Usage Guide

`remote-run` lets you edit code and run AI CLI agents (`agy`, `Codex`, `Claude Code`) on a lightweight laptop while offloading compute (tests, builds, data scripts, GPU training) to a remote Linux host.

## Core Design

1. **Stateless default `exec` + opt-in `--session`**: Every `exec` runs inside an isolated remote `tmux` session and process group (`PGID`), auto-detaching into the background if it exceeds `--wait` (default `15`s). Pass `--session <name>` when you want `cwd` and environment diffs preserved across commands.
2. **Additive-by-default one-way sync + ownership markers**: Code stays locally on your laptop and syncs additively from the repository root (`rsync -rlptz`, never deleting remote files unless you explicitly run `remote-run sync --delete`). Remote roots (`.remote-run-root.json`) and workspaces (`.remote-run-workspace.json`) verify ownership dynamically across any machine without hardcoded path denylists. Remote `.venv/`, `outputs/`, `checkpoints/`, `.cache/` (and `[sync].preserve` / `[share].project`) are neither uploaded from local nor deleted by `--delete`.
3. **Strict 2-file config model**: Only `~/.config/remote-run/hosts.toml` and optional `<repo_root>/.remote-run.toml` exist. Unknown keys are rejected immediately.

## 1. Host Setup (`~/.config/remote-run/hosts.toml`)

Remote Linux host requirements:
- Batch-mode SSH alias (e.g., `linux_desktop`) with known host key (tip: enable SSH `ControlMaster auto` / `ControlPersist 60` in `~/.ssh/config` for faster edit-test loops)
- `python3` (3.11+), `git`, `tmux`, `rsync`
- For CUDA hosts: NVIDIA driver + `nvidia-smi`

Copy [examples/hosts.example.toml](../examples/hosts.example.toml) to `~/.config/remote-run/hosts.toml` and verify:

```bash
remote-run hosts
remote-run doctor
remote-run doctor --check-cuda --check-import torch
```

*(If pointing `hosts.toml` `root` at a pre-existing non-empty directory that already belongs to `remote-run`, run `remote-run doctor --init-root` once after verifying its path.)*

### Switching Machines, GPUs, and Environments
- Pass `--host <name>` (before or after any subcommand), or set `host = "<name>"` in `.remote-run.toml`.
- Define a CPU profile (`[hosts.desktop_cpu]` with `device = "cpu"`) alongside your GPU profile so tests/builds never block behind training runs.
- Built-in `--env venv` automatically prepends `<remote_workspace>/.venv/bin` if present (ideal for `uv sync` / `python3 -m venv .venv`); built-in `--env none` runs without injecting any host Python.
- For `device = "cuda:N"`, `CUDA_VISIBLE_DEVICES=N` is set automatically unless overridden.

### Moving Local Folders & Multi-Worktree Development
- Moving `/a/my-proj` to `/b/my-proj` keeps the same remote workspace (keyed by common repo name + root commit + branch, or explicit `project` in `.remote-run.toml`), never local absolute paths.
- Opening multiple Git worktrees on different branches gives each branch an isolated remote code directory (`.../workspaces/<branch>`) while sharing `[share].project` directories (`.../shared/<name>`).
- Single-directory developers who switch branches in-place (`git checkout feat`) can set `workspace = "default"` in `.remote-run.toml` to reuse `.venv/` and build caches across branches.
- Pass `--remote-dir /abs/path` (or set `remote_dir` in `.remote-run.toml`) to target an exact remote directory directly (cannot be combined with `[share].project`; pre-existing non-empty folders require explicit `remote-run sync --adopt`).

## 2. Daily Commands

```bash
# Quick check without syncing or waiting behind a GPU job
remote-run exec --no-sync --no-gpu -- nvidia-smi

# Project-local uv/.venv workflow
remote-run exec --env venv -- uv run pytest

# Preview and clean stale remote files if you deleted/renamed local files
remote-run sync --dry-run --delete
remote-run sync --delete

# Opt-in persistent session across calls
remote-run exec --session dev --shell "cd sub && export TAG=v1"
remote-run exec --session dev --no-sync --shell 'pwd && echo $TAG'
remote-run session reset dev

# Readable terminal logs on test failures (--print-log) or indented JSON (--pretty)
remote-run exec --print-log -- pytest -v

# Reconnect after disconnect / list project runs / inspect raw logs
remote-run status RUN_ID --wait 30
remote-run status --repo .
remote-run status RUN_ID --offset 8192 --raw-log

# Cancel a queued or running job cleanly (kills PGID + frees VRAM, keeps logs)
remote-run cancel RUN_ID

# Pull logs, metadata, and eligible artifacts (or merge into existing dir with --force)
remote-run pull RUN_ID ./results/RUN_ID --path outputs/notes.txt

# Inspect or clean old terminal runs (>14 days) on the host (skips unmanaged dirs)
remote-run prune --days 14
remote-run prune --days 14 --apply
```

## 3. Global Cross-Harness Installation (`agy`, `Codex`, `Claude Code`)

See [skills/remote-run/references/schema-and-setup.md](../skills/remote-run/references/schema-and-setup.md) for the exact multi-harness setup commands.
