# remote-run

**English** | [简体中文](README.zh-CN.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib)-brightgreen.svg)](remote_run.py)
[![MCP Server](https://img.shields.io/badge/MCP-stdio%20server-purple.svg)](remote_run_mcp.py)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-orange.svg)](skills/remote-run/SKILL.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **Keep Claude Code, Codex, Antigravity, your API keys, MCP servers, Skills, and editor on your laptop. Turn any Linux GPU server into a zero-setup compute backend.**

No installing AI CLI agents on rented cloud GPUs. No exposing API keys or proxies on shared lab machines. No noisy `wip` Git commits just to run tests or training on another box.

---

## Why `remote-run`?

If you code on a thin laptop using **Claude Code**, **Codex**, **Antigravity (`agy`)**, or your local IDE—while your compute lives on a desktop GPU, lab workstation, university cluster, or rented cloud server—existing workflows get painful fast:

- **VS Code Remote-SSH**: Treats the server as your development machine. Every time you switch or rent a new GPU host, you have to reinstall AI agents, configure proxies/keys, sync extensions/memories, and clean up afterwards—and many HPC/lab servers cannot reach external LLM APIs at all.
- **Git `push` / `pull` ping-pong**: Editing two lines on your laptop forces you to commit, push, SSH over, pull, run, hit a bug, and switch back—leaving polluted commit history and constant context switching.
- **Ad-hoc `ssh` + `rsync` scripts**: Easy to lose jobs on Wi-Fi drops, collide on GPU memory, or accidentally wipe remote `.venv/` and checkpoints with an unguarded `rsync --delete`.

**`remote-run` flips the model:**

```text
Your Laptop (Local Source of Truth)
  Editor + Claude Code / Codex / agy + API Keys + MCPs + Memory
        │
        │  SSH + incremental additive rsync (transparent to you)
        ▼
Any Linux Server / Lab Desktop / Rented Cloud GPU (Zero AI setup!)
  ├─ Remote workspace auto-isolated per project & branch (move-safe, worktree-safe)
  ├─ Detached tmux execution + physical GPU flock queuing + clean PGID cancel
  ├─ Protects remote .venv/, outputs/, checkpoints/, .cache/ from overwrite/delete
  └─ Pulls back logs, metrics & small artifacts automatically (skips huge weights by default)
```

## Comparison

| Workflow | Needs AI tools / keys / proxy on server? | Works on rented or borrowed servers? | Needs Git commits to test changes? | Survives laptop sleep / Wi-Fi drop? | Built-in safety & GPU queue | AI Agent native (MCP / Skill) |
|---|---|---|---|---|---|---|
| **`remote-run`** | **No (zero remote AI setup)** | **Yes (plug-and-play)** | **No (live additive sync)** | **Yes (`tmux` + `--wait`)** | **Yes (ownership markers + `flock`)** | **Yes (CLI + MCP + `SKILL.md`)** |
| **VS Code Remote-SSH** | Yes (reconfigure per server) | Hard / risky for keys | No | Partial | No | Runs agent remotely |
| **Git push/pull loop** | Yes (if using remote AI) | Yes | Yes (noisy WIP commits) | No (needs manual tmux) | No | Manual |
| **Raw `ssh` + `rsync`** | No | Yes | No | No (needs manual tmux) | Risk of accidental `rsync --delete` | Unstructured text |

---

## Common Use Cases

1. **Zero-setup rented cloud GPUs & shared lab desktops**: Keep your entire AI workflow, prompt history, and credentials on your own laptop; only `python3`, `git`, `tmux`, and `rsync` (no `sudo`) are needed on the server.
2. **Fast edit–test loops without Git churn**: Edit Python/Rust/Go/C++ locally and immediately run `pytest`, `cargo test`, or builds remotely (`remote-run exec -- pytest -q`) without pushing commits.
3. **Multi-hour PyTorch / CUDA training**: Jobs auto-detach into `tmux` after `--wait` (default 15s), serialize on physical GPU locks (`gpu_slot`), isolate `CUDA_VISIBLE_DEVICES`, fold tqdm progress bars in logs, and clean up VRAM cleanly on `cancel`.
4. **Download huge datasets directly on the remote host**: Use `--no-sync` with a CPU profile (`--host cpu_box` or `--no-gpu`) to pull HuggingFace models, DVC buckets, or archives straight onto the server without touching laptop bandwidth or blocking GPU jobs.
5. **Dual-speed local/remote AI workflow**: Let Claude Code / Codex / `agy` edit files locally and invoke `remote-run` via **MCP (`remote_*`)** or **Agent Skill** whenever heavy execution is needed.

---

## 2-Minute Quickstart

### Prerequisites

- **Laptop**: Linux/macOS with Python 3.11+, `ssh`, and `rsync`
- **Remote Linux host**: `python3` (3.11+), `git`, `tmux`, `rsync` on SSH `PATH`, and batch-mode key auth (`ssh -o BatchMode=yes <alias> true`) — **no `sudo` required**

### 1. Clone and install CLI & MCP binaries

```bash
git clone https://github.com/zhkzly/remote-run.git ~/remote-run
mkdir -p ~/.local/bin
ln -sf ~/remote-run/bin/remote-run ~/.local/bin/remote-run
ln -sf ~/remote-run/bin/remote-run-mcp ~/.local/bin/remote-run-mcp
```

### 2. Configure your host (`~/.config/remote-run/hosts.toml`)

```toml
schema_version = 1
default_host = "desktop"

[hosts.desktop]
ssh = "my-gpu-box"             # Batch-mode SSH alias from ~/.ssh/config
root = "/data/user/remote-jobs"
locks_dir = "/data/user/remote-locks"
device = "cuda:0"
gpu_slot = "gpu-0"
default_env = "venv"           # Uses <remote_workspace>/.venv if present
path_prepend = ["/home/user/.local/bin"]

# Optional lock-free CPU alias sharing the exact same root & workspaces
[hosts.cpu_box]
ssh = "my-gpu-box"
root = "/data/user/remote-jobs"
locks_dir = "/data/user/remote-locks"
device = "cpu"
default_env = "venv"
```

Verify SSH, `tmux`, `git`, `rsync`, and GPU detection:

```bash
remote-run doctor --pretty
# If `root` is a pre-existing non-empty directory you own:
remote-run doctor --init-root
```

### 3. Connect to Claude Code, Codex, or Antigravity (`agy`)

```bash
# Install Agent Skill (optional, teaches agents best practices & safety rules)
mkdir -p ~/.agents/skills ~/.claude/skills
ln -sfn ~/remote-run/skills/remote-run ~/.agents/skills/remote-run   # Codex & agy
ln -sfn ~/remote-run/skills/remote-run ~/.claude/skills/remote-run   # Claude Code

# Register zero-dependency stdio MCP Server (native remote_* tools)
claude mcp add -s user remote-run -- ~/.local/bin/remote-run-mcp
codex mcp add remote-run -- ~/.local/bin/remote-run-mcp
agy mcp add remote-run ~/.local/bin/remote-run-mcp
```

### 4. Run anything from your local repo

```bash
# Sync incremental code changes + run tests on remote (auto-uses project .venv if present)
remote-run exec --host cpu_box -- pytest -q

# Run GPU training (auto-detaches after 15s, queues on GPU lock, prints run_id)
remote-run exec -- python train.py --epochs 20

# Direct remote action without syncing from laptop & without GPU lock (e.g., HF/DVC download)
remote-run exec --host cpu_box --no-sync --remote-dir /data/user/datasets/ds --shell "hf download ... --local-dir ."

# Opt-in persistent shell session (keeps cd & exports across calls when exit_code == 0)
remote-run exec --session dev --shell "cd sub && export TAG=v1"
remote-run exec --session dev --no-sync -- python eval.py

# Check incremental logs, cancel safely (frees GPU VRAM), and pull small artifacts
remote-run status RUN_ID --wait 30 --print-log
remote-run cancel RUN_ID
remote-run pull RUN_ID ./results/RUN_ID
```

---

## Safety Designed for Autonomous AI Agents

Giving an AI agent remote shell + sync tools requires hard guardrails in **code**, not just prompt instructions:

1. **Additive-only sync by default**: `sync` and `exec` use incremental `rsync -rlptz` **without `--delete`**. Files generated remotely (logs, datasets, checkpoints) are never deleted by accident. Explicit deletion requires `remote-run sync --dry-run --delete` followed by `remote-run sync --delete`.
2. **Machine-agnostic dual ownership markers**: Pre-existing non-empty directories are never overwritten silently. Roots require `.remote-run-root.json` (`doctor --init-root`), workspaces track `.remote-run-workspace.json` (`project_id`, `root_commit`, `workspace_id`), and `sync --adopt` is required once to claim an existing non-empty folder.
3. **Dynamic topology guards**: Refuses `/`, local/remote `$HOME` and `$HOME` ancestors, control directories (`runs/`, `locks/`, `sessions/`, `projects/*/shared/`), and symlink escapes during `pull`.
4. **Remote-owned preservation**: `.venv/`, `outputs/`, `checkpoints/`, `.cache/`, `[sync].preserve`, and `[share].project` are excluded from local upload and protected from remote deletion.
5. **Safe cleanup & retrieval**: `prune` is dry-run by default and `--apply` only deletes verified `remote-run` run directories whose `tmux`/worker/`PGID` processes are dead; `pull` skips checkpoint suffixes (`.pt`, `.safetensors`, etc.) and files over `--max-file-mb` (10 MB) unless `--all-artifacts` is set.

---

## MCP Server Tools (`remote-run-mcp`)

Zero-dependency JSON-RPC 2.0 stdio server (`2025-06-18` / `2025-03-26` / `2024-11-05`) with MCP tool annotations:

| Tool | Purpose | Safety boundary |
|---|---|---|
| `remote_hosts` | List configured hosts, devices, Python envs, and shared assets | Read-only |
| `remote_config` | Inspect merged `hosts.toml` + `.remote-run.toml` workspace & ignore rules | Read-only |
| `remote_doctor` | Probe SSH, tools, GPUs, Python imports, CUDA, and shared asset SHA-256 | Read-only check |
| `remote_sync` | One-way additive `rsync` (supports `dry_run: true`) | Never exposes `delete` or `adopt` |
| `remote_exec` | Sync (unless `no_sync: true`) and execute command/shell on remote host | Interactive/background with PGID isolation |
| `remote_status` | Poll job state / byte-offset logs or list recent runs | Read-only |
| `remote_cancel` | Stop queued/running job by PGID + `tmux` while preserving logs | DestructiveHint |
| `remote_pull` | Pull logs, metadata, small artifacts, or `--path` files locally | Skips checkpoints >10 MB by default |
| `remote_session` | List or reset opt-in persistent shell sessions (`cwd` + `env`) | Explicit reset only |
| `remote_prune` | Preview or clean expired terminal run dirs (`dry-run` unless `apply: true`) | Whitelist-only, preserves live/unmanaged dirs |

---

## Documentation & Configuration

- **[docs/usage.md](docs/usage.md)** — Full CLI guide, multi-GPU isolation, `[share].project` cross-worktree sharing, `[share].host` SHA-256 pinning, and placeholders (`{artifacts}`, `{device}`, `{shared.<alias>.path}`)
- **[skills/remote-run/SKILL.md](skills/remote-run/SKILL.md)** & **[references/schema-and-setup.md](skills/remote-run/references/schema-and-setup.md)** — Agent Skill specification & strict 2-file TOML schema
- **[examples/hosts.example.toml](examples/hosts.example.toml)** & **[examples/project.example.toml](examples/project.example.toml)** — Ready-to-copy templates
- **Tests**: Run `python3 -B -m unittest discover -s tests -v`

## License

[MIT](LICENSE)
