# remote-run

**English** | [简体中文](README.zh-CN.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib)-brightgreen.svg)](remote_run.py)
[![MCP Server](https://img.shields.io/badge/MCP-stdio%20server-purple.svg)](remote_run_mcp.py)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-orange.svg)](skills/remote-run/SKILL.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **Edit locally with Claude Code, Codex, or Antigravity (`agy`). Execute transparently on any remote Linux host—CPU or GPU, any language, zero remote AI setup.**

Keep your editor, AI agents, API keys, MCP servers, and local source tree on your laptop. Turn any Linux server, lab desktop, or cloud VM into an invisible remote compute backend with almost zero configuration.

---

## Why `remote-run`?

Traditional remote workflows get painful fast when you code locally on a laptop with **Claude Code**, **Codex**, or **Antigravity (`agy`)** while compiling, testing, or training on a remote Linux machine:

- **VS Code Remote-SSH**: Forces you to install AI tools, proxies, API keys, and extensions on every server you touch—impossible or risky on borrowed workstations, clusters, and ephemeral cloud boxes.
- **Git `push` / `pull` ping-pong**: Pollutes commit history with `wip` commits just to run tests or builds on another box.
- **Ad-hoc `ssh` + `rsync` scripts**: Hang on SSH handshakes, leave deleted/renamed ghost files that break Go/Rust/C++ builds, risk wiping remote caches/outputs with raw `rsync --delete`, and collide on GPUs.

**`remote-run` separates local intelligence from remote compute across 4 orthogonal layers:**

```text
Laptop (Source of Truth: Editor + Claude Code / Codex / agy + API Keys + MCP)
        │
        │  Multiplexed SSH (0700 ControlMaster) + smart manifest-tracked sync
        ▼
Any Remote Linux Host (Zero AI tools, zero sudo, no daemon to install)
  ├─ Connectivity & Env : user@host / port / ~/paths, auto-detected ~/.cargo/bin & ~/go/bin, venv/named envs
  ├─ Workspace & Flow   : branch-safe single checkouts, auto-isolated Git worktrees, 4-class sync safety
  ├─ Execution          : stateless tmux or opt-in --session, live --follow streaming, clean PGID cancel
  └─ Optional Resources : lock-free on CPU; automatic sorted flock + CUDA isolation on GPU (--device / --no-gpu)
```

---

## Common Use Cases

1. **Any-language edit–test–build loop**: Write **Rust, Go, C++, Node/TS, or Python** locally and run `remote-run exec --follow -- cargo test` or `pytest -q`. Owned-file manifest tracking (`.remote-run-synced.json`) safely removes ghost files when you delete or rename local source files—without ever deleting remote outputs, build caches, or newly ignored files.
2. **Normal checkouts keep caches across branches**: Switching branches in a single local checkout keeps `default` on the remote host (preserving `target/`, `build/`, `.venv/`, `node_modules/`). Actual linked Git worktrees automatically isolate per worktree while sharing `[share].project`.
3. **Zero-setup rented cloud VMs & lab GPUs**: Keep AI keys and prompts on your laptop; onboard any box in seconds with `remote-run init user@host --port 22345` (zero `sudo`, no remote daemon).
4. **Multi-hour GPU runs without VRAM collisions**: Jobs exceeding `--wait` (default `15s`) detach into remote `tmux`; CUDA hosts serialize cleanly on sorted GPU locks (including multi-GPU `--device cuda:0,1` acquiring both `gpu-0` and `gpu-1` in deadlock-free order) and release VRAM via PGID kill on `cancel`. Light tasks use `--no-gpu` to run lock-free with `CUDA_VISIBLE_DEVICES=""`.
5. **AI Agent native (CLI + MCP + Skill)**: Exposes equivalent workflows via terminal CLI, zero-dependency stdio MCP (`remote_*`), and a progressive-disclosure `SKILL.md`.

---

## Quickstart

### Prerequisites

- **Laptop**: Linux or macOS (`rsync` auto-adapts for macOS), Python 3.11+, `ssh`
- **Remote Linux host**: Python 3.11+, `bash`, `tmux`, `rsync` (no `git` or `sudo` needed on the remote host), batch-mode SSH auth (`ssh -o BatchMode=yes <target> true`)

### 1. Clone and link CLI & MCP binaries

```bash
git clone https://github.com/zhkzly/remote-run.git ~/remote-run
mkdir -p ~/.local/bin
ln -sf ~/remote-run/bin/remote-run ~/.local/bin/remote-run
ln -sf ~/remote-run/bin/remote-run-mcp ~/.local/bin/remote-run-mcp
```

### 2. Initialize or write `~/.config/remote-run/hosts.toml`

Run interactive-free auto-probe initialization:

```bash
remote-run init my-server           # or: remote-run init user@1.2.3.4 --port 22345 --name box
```

Or write the minimal 3-line config directly (`root` defaults to `~/.local/share/remote-run`, `device` defaults to `cpu`):

```toml
schema_version = 1

[hosts.server]
ssh = "my-server"                   # ~/.ssh/config alias or "user@host"
# port = 22345                      # optional custom SSH port
# device = "cuda:0"                 # uncomment on GPU hosts to enable GPU locks + CUDA isolation
```

Verify SSH, tools (`bash`/`tmux`/`rsync`), optional commands (`--check-cmd`), and GPUs:

```bash
remote-run doctor --check-cmd cargo --pretty
# If `root` points to a pre-existing non-empty directory you own:
remote-run doctor --init-root
```

### 3. Connect Claude Code, Codex, or Antigravity (`agy`)

```bash
# Install Agent Skill
mkdir -p ~/.agents/skills ~/.claude/skills
ln -sfn ~/remote-run/skills/remote-run ~/.agents/skills/remote-run   # Codex & agy
ln -sfn ~/remote-run/skills/remote-run ~/.claude/skills/remote-run   # Claude Code

# Register zero-dependency stdio MCP Server
claude mcp add -s user remote-run -- ~/.local/bin/remote-run-mcp
codex mcp add remote-run -- ~/.local/bin/remote-run-mcp
agy mcp add remote-run ~/.local/bin/remote-run-mcp
```

### 4. Run anything from your local repo

```bash
# Sync changes and stream remote output live (works for Rust, Go, C++, Node, Python...)
remote-run exec --follow -- cargo test
remote-run exec --env venv --follow -- uv run pytest

# Override GPU selection per run on a CUDA host (lock-free CPU or multi-GPU sorted locking)
remote-run exec --no-gpu -- pytest -q
remote-run exec --device cuda:0,1 -- torchrun --nproc_per_node=2 train.py

# Direct remote action without uploading laptop files (e.g., download dataset directly on host)
remote-run exec --no-gpu --no-sync --remote-dir ~/datasets/v1 --shell "hf download ... --local-dir ."

# Opt-in persistent shell session (keeps cd & env diffs across calls when exit_code == 0)
remote-run exec --session dev --shell "cd sub && export TAG=v1"
remote-run exec --session dev --no-sync -- make test

# Reconnect, attach tmux_session, cancel cleanly, or pull results/workspace files
remote-run status RUN_ID --follow
remote-run cancel RUN_ID
remote-run pull RUN_ID ./results/RUN_ID
```

---

## Safety Designed for Shared Servers & AI Agents

1. **Zero silent data loss**: Sync never deletes remote-owned files or newly ignored files. Only regular files previously uploaded by `remote-run` (tracked in `.remote-run-synced.json`) that no longer exist locally are automatically pruned on rename/delete.
2. **Ownership & topology guards**: Roots require `.remote-run-root.json`, managed workspaces require `.remote-run-workspace.json`, pre-existing non-empty folders require explicit `--init-root` / `--adopt`, and `/`, `$HOME`, ancestors, control dirs, and pull symlink escapes are rejected in code.
3. **Multi-user privacy**: SSH `ControlMaster` sockets, `locks/`, `sessions/`, and `runs/` (`request.json` env vars, logs, artifacts) are locked down to mode `0700` while preserving your workspace's natural `umask`.
4. **Safe cleanup & retrieval**: `prune` is dry-run by default and never touches live or unmanaged directories; `pull` skips ML checkpoint suffixes (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, `.h5`) and files >10 MB unless `--all-artifacts` is passed.

---

## MCP Server Tools (`remote-run-mcp`)

Zero-dependency JSON-RPC 2.0 stdio server (`2025-06-18` / `2025-03-26` / `2024-11-05`) with concurrent request handling:

| Tool | Purpose | Safety boundary |
|---|---|---|
| `remote_hosts` | List hosts, devices, named environments, and shared assets | Read-only |
| `remote_config` | Inspect merged `hosts.toml` + `.remote-run.toml` config & rules | Read-only |
| `remote_doctor` | Probe SSH, `bash`/`tmux`/`rsync`, `--check-cmd`, GPUs, imports, shared SHA-256 | Read-only check |
| `remote_sync` | Incremental sync (+ safe ghost cleanup for deleted owned files) | Never exposes `delete` or `adopt` |
| `remote_exec` | Sync (unless `no_sync=true`) and run command/shell (`device`, `no_gpu`, `wait`) | PGID + `tmux` isolation |
| `remote_status` | Poll job state / byte-offset logs (`tmux_session` included) or list runs | Read-only |
| `remote_cancel` | Cancel queued/running job by PGID + `tmux` while preserving logs | `destructiveHint` |
| `remote_pull` | Pull logs, metadata, artifacts, or `--path` workspace files | Skips large/checkpoint artifacts by default |
| `remote_session` | List or reset opt-in persistent shell sessions (`cwd` + `env`) | Explicit reset only |
| `remote_prune` | Preview or clean expired terminal run dirs (`dry-run` unless `apply: true`) | Whitelist-only |

---

## Documentation & Tests

- **[docs/usage.md](docs/usage.md)** — Complete workflow guide, 4-class sync table, multi-GPU sorted locking, `[share]`, and placeholders
- **[skills/remote-run/SKILL.md](skills/remote-run/SKILL.md)** & **[references/schema-and-setup.md](skills/remote-run/references/schema-and-setup.md)** — Agent Skill specification & leveled Level 0/1/2 TOML schema
- **[examples/hosts.example.toml](examples/hosts.example.toml)** & **[examples/project.example.toml](examples/project.example.toml)** — Progressive-disclosure templates
- **Tests**: Run `python3 -B -m unittest discover -s tests -v`

## License

[MIT](LICENSE)
