# remote-run Usage Guide

`remote-run` keeps editing and AI agents (`agy`, Codex, Claude Code) on your local machine while transparently executing commands, builds, tests, and GPU workloads on any remote Linux host over SSH.

## Architecture Overview

- **Connectivity, Config & Environments**: Built-in `0700` SSH `ControlMaster` multiplexing, optional `port`, `user@host`, `~/` remote paths, named environments (`PATH` + env vars + optional Python), and automatic discovery of existing standard remote tool dirs (`~/.local/bin`, `~/.cargo/bin`, `~/go/bin`, `/usr/local/go/bin`).
- **Workspace, Sync & Data Flow**: 4-category sync model (see below), automatic single-checkout cache preservation vs. linked Git worktree isolation, ownership markers, and safe artifact retrieval.
- **Execution, Sessions & Lifecycle**: Stateless isolated `tmux` + PGID execution per run, opt-in `--session <name>` persistence, real-time `--follow` streaming, `tmux_session` reporting, and whitelist-only `prune`.
- **Optional Resource & Concurrency Aspect**: Lock-free concurrency by default (`device = "cpu"`). When a host declares `cuda`, sorted multi-slot `flock` locks (`--device cuda:0,1` locks `gpu-0` and `gpu-1` in order) and CUDA visibility isolation apply cleanly; `--no-gpu` skips locks and hides CUDA.

---

## 1. Host Setup (`~/.config/remote-run/hosts.toml`)

Remote Linux host requirements:
- Batch-mode SSH access (`ssh -o BatchMode=yes <alias-or-user@host> true`)
- `python3` (3.11+), `bash`, `tmux`, `rsync` — **no remote `git` or `sudo` required**
- For CUDA hosts: NVIDIA driver + `nvidia-smi`

Quickest setup:

```bash
remote-run init my-server
# Or with explicit user, port, and custom name:
remote-run init ubuntu@192.168.1.50 --port 22345 --name lab --default
```

Or copy [examples/hosts.example.toml](../examples/hosts.example.toml) to `~/.config/remote-run/hosts.toml`, then verify:

```bash
remote-run hosts
remote-run doctor --check-cmd cargo --pretty
remote-run doctor --repo . --check-cuda --check-import torch
```

*(If `root` points to an existing non-empty directory that belongs to `remote-run`, run `remote-run doctor --init-root` once after verifying its path.)*

### Multi-GPU, CPU-Only Runs, and Named Environments
- Use `--device <dev>` on `exec` to override the host default for one run (`--device cuda:1`, `--device cuda:0,1`, or `--device cpu`). Multi-GPU specs acquire per-card locks (`gpu-0`, `gpu-1`) in sorted order so `cuda:0,1` never collides with `cuda:0` and cannot deadlock.
- Pass `--no-gpu` on `exec` to skip GPU locks and set `CUDA_VISIBLE_DEVICES=""` for quick checks or CPU tests on a GPU host.
- Pass `--env <name>`, built-in `--env venv` (auto-activates `<remote_workspace>/.venv/bin`), or built-in `--env none`.
- On CPU hosts (`device = "cpu"`), no `CUDA_VISIBLE_DEVICES` or `CUBLAS_WORKSPACE_CONFIG` env vars are injected.

---

## 2. Four-Category Sync & Data Model

| Content Type | Controlled By | Uploaded? | Remote Deletion / Protection |
|---|---|---|---|
| **1. Source & tracked files** | Default (non-ignored files) | Yes (incremental `rsync`) | Tracked in `.remote-run-synced.json`; automatically pruned ONLY if previously uploaded AND truly deleted/renamed locally |
| **2. Ignored files** | `.gitignore` + `BUILTIN_IGNORE` + `[sync].ignore` | No | Skipped during upload; newly ignored files that still exist locally are **never** deleted remotely |
| **3. Remote-owned outputs & caches** | `BUILTIN_PRESERVE` + `[sync].preserve` | No | Bidirectional protection: never uploaded/overwritten and never deleted by `--delete` or ghost cleanup |
| **4. Shared assets** | `[share].project` (per-project symlinks)<br>`[share].host` (read-only host assets) | No | `project` mounts `projects/<id>/shared/<dir>` via symlinks + auto-preserve; `host` checks path/SHA-256 and sets `REMOTE_RUN_SHARED_*` env vars (no symlinks) |

Built-in ignores: `.git/`, `.venv/`, `venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `node_modules/`, `.DS_Store`, `/results/` (anchored at repo root so `src/results/` is still synced).
Built-in preserves: `.venv`, `outputs`, `checkpoints`, `.cache`.

### Moving Local Folders & Multi-Worktree Development
- Project identity is derived from repo name + root commit (never local absolute path), so moving a local folder preserves the remote workspace.
- **Single-checkout repos** default to `workspace_id = "default"`, so switching branches with `git checkout -b ...` preserves remote `target/`, `build/`, `.venv/`, and `node_modules/`.
- **Linked Git worktrees** (`git worktree add`) automatically isolate per branch under `projects/<id>/workspaces/<branch>` while sharing `[share].project` directories.
- Pass `--remote-dir <path>` (or set `remote_dir` in `.remote-run.toml`) to target a specific remote directory (cannot combine with `[share].project`; claiming a non-empty unmanaged dir for sync requires `remote-run sync --adopt` once).

---

## 3. Daily Commands

```bash
# Live streaming execution (prints metadata JSON to stderr, exits 0/2; Ctrl-C detaches locally)
remote-run exec --follow -- cargo test
remote-run exec --env venv -- uv run pytest -q

# Override GPU or skip GPU lock for one command
remote-run exec --no-gpu --no-sync -- nvidia-smi
remote-run exec --device cuda:0,1 -- torchrun --nproc_per_node=2 train.py

# Run directly in an arbitrary remote dir without syncing laptop files
remote-run exec --no-gpu --no-sync --remote-dir ~/datasets/v1 --shell "hf download ... --local-dir ."

# Opt-in persistent shell across commands (saves cwd & env diffs when exit_code == 0)
remote-run exec --session dev --shell "cd sub && export TAG=v1"
remote-run exec --session dev --no-sync --shell 'pwd && echo $TAG'
remote-run session list --repo .
remote-run session reset dev

# Check status, follow an existing detached run, list runs, or attach manually over SSH
remote-run status RUN_ID --wait 30 --print-log
remote-run status RUN_ID --follow
remote-run status --repo .
# ssh -t <ssh_target> tmux attach -t <tmux_session>

# Cancel cleanly (kills PGID + tmux session) & pull results / specific workspace paths
remote-run cancel RUN_ID
remote-run pull RUN_ID ./results/RUN_ID
remote-run pull RUN_ID ./results/RUN_ID --path reports/bench.md --paths-only --force

# Preview stale files or prune old terminal run directories (>14 days)
remote-run sync --dry-run --delete
remote-run prune --days 14
remote-run prune --days 14 --apply
```

## 4. Multi-Harness Agent Setup (`agy`, Codex, Claude Code)

See [skills/remote-run/references/schema-and-setup.md](../skills/remote-run/references/schema-and-setup.md) for the leveled TOML schema and exact installation commands.
