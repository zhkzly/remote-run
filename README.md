# remote-run

**English** | [简体中文](README.zh-CN.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib)-brightgreen.svg)](remote_worker.py)
[![MCP Server](https://img.shields.io/badge/MCP-stdio%20server-purple.svg)](remote_run_mcp.py)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-orange.svg)](skills/remote-run/SKILL.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **Edit locally with Claude Code, Codex, or Antigravity (`agy`). Execute transparently on any remote Linux host — language-universal, GPU-optional, zero remote installation.**

Keep your editor, AI agents, and credentials on your laptop. Ship a single stdlib-only worker on demand over SSH, transfer an explicit verified manifest, and run any toolchain (Rust, Go, C++, Node/TS, Python, ...) cleanly on a remote Linux host.

---

## Architecture (4 Orthogonal Components + Unified Interaction Layer)

```text
                ┌──────────────────────────────────────────────────────┐
                │   Interaction Layer: CLI / MCP / Agent Skill         │
                └──────────────────────────┬───────────────────────────┘
                                           │ Typed Service (remote_run_local/service.py)
                  3-Layer Config: hosts.toml + .remote-run.toml + XDG bindings/
        ┌────────────────┬─────────────────┼─────────────────┬────────────────┐
        ▼                ▼                 ▼                 ▼                │
  1. Transport     2. Space & Transfer  3. Execution & Env  4. Resources      │ Security Invariants
  - OpenSSH mux    - 4-level hierarchy  - Pure PATH + env   - CPU by default  │ - 0700 state dirs
  - Standalone     - Single manifest    - argv xor shell    - Optional [gpu]  │ - Ownership markers
    remote_worker  - Untracked budget   - tmux / --session  - UUID-sorted     │ - Pinned pull/sync
  - Auth hints     - Atomic pull        - Read-only doctor    flock + mask    │ - Strict schema v2
```

### Remote 4-Level Hierarchy

```text
<host.root>/                                    # Level 1: host root (default ~/.local/share/remote-run-v2)
├── projects/<project_id>/                      # Level 2: multi-project isolation
│   ├── shared/<rel_dir>/                       #   cross-worktree shared dirs ([workspace].shared)
│   └── workspaces/                             # Level 3: worktree isolation
│       ├── default/                            #   single checkout (reused across branch switches)
│       └── wt-<hash>/ (or bound workspace ID)  #   each linked Git worktree isolated automatically
├── runs/<run_id>/                              # Level 4: 0700 run receipt, logs, artifacts
├── locks/                                      # Optional 0700 GPU lock directory
└── sessions/                                   # Optional 0700 persistent shell sessions
```

---

## Quickstart

### Prerequisites
- **Laptop**: Linux or macOS, Python 3.11+, `ssh`, `rsync` (Git for repo identity and ignore rules).
- **Remote Linux host**: Python 3.11+ (worker infrastructure), `bash`, `tmux`, `rsync`, working batch-mode SSH auth (`ssh -o BatchMode=yes <target> true`). No `sudo`, no remote package install, no remote AI setup.

### 1. Clone and link CLI & MCP launchers

```bash
git clone <repository-url> ~/remote-run
mkdir -p ~/.local/bin
ln -sf ~/remote-run/bin/remote-run ~/.local/bin/remote-run
ln -sf ~/remote-run/bin/remote-run-mcp ~/.local/bin/remote-run-mcp
```

### 2. Configure `~/.config/remote-run/hosts.toml` (Machines Only)

Run `remote-run init <ssh-target>` or write a minimal 3-line catalog:

```toml
schema_version = 2
default_host = "builder"

[hosts.builder]
ssh = "linux-builder"                   # ~/.ssh/config alias or "user@host"
# port = 22
# root = "~/.local/share/remote-run-v2"
# path = ["~/.local/bin", "~/.cargo/bin"]

# Optional: declare GPUs only on CUDA hosts; omit on pure CPU machines
# [hosts.gpu-workstation]
# ssh = "workstation"
# [hosts.gpu-workstation.gpu]
# devices = [0, 1]
```

### 3. Bind a Project to a Host (Optional, Zero Repo Pollution)

Keep personal host names, remote prebuilt paths, dataset paths, and optional GPU picks out of Git using XDG bindings (`~/.config/remote-run/bindings/`):

```bash
# Repo-scoped binding on a CPU host: automatically inherited by ALL Git worktrees
remote-run bind --scope repo --host builder \
  --note "Prebuilt toolchain ready; invoke cargo/pytest directly" \
  --path /opt/envs/proj/bin \
  --env DATASET_DIR=/srv/datasets/v2

# Checkout-scoped binding: pin a readable workspace name (add --gpus 0 on a GPU-configured host)
remote-run bind --scope checkout --host builder --workspace exp-a
```

Inspect the merged result anytime (redacts env values while showing origins and PATH order):

```bash
remote-run config --pretty
remote-run doctor --check-cmd cargo --check-cmd /opt/envs/proj/bin/python --pretty
```

### 4. Connect Claude Code, Codex, or Antigravity (`agy`)

```bash
mkdir -p ~/.agents/skills ~/.claude/skills
ln -sfn ~/remote-run/skills/remote-run ~/.agents/skills/remote-run
ln -sfn ~/remote-run/skills/remote-run ~/.claude/skills/remote-run

claude mcp add -s user remote-run -- ~/.local/bin/remote-run-mcp
codex mcp add remote-run -- ~/.local/bin/remote-run-mcp
agy mcp add remote-run ~/.local/bin/remote-run-mcp
```

> **Codex SSH Agent Note**: Codex strips environment variables for stdio MCP servers by default. If your SSH login uses `ssh-agent`, add `env_vars = ["SSH_AUTH_SOCK"]` under `[mcp_servers.remote-run]` in `~/.codex/config.toml` (or configure `IdentityFile` in `~/.ssh/config`).

```toml
[mcp_servers.remote-run]
command = "/path/to/remote-run-mcp"
env_vars = ["SSH_AUTH_SOCK"]
```

### 5. Execute Any Language Remotely

```bash
remote-run exec --follow -- cargo test
remote-run exec --follow -- go test ./...
remote-run exec --follow -- node --test
remote-run exec --follow -- cmake --build build
remote-run exec --gpus 0,1 --wait 15 -- python -u train.py
remote-run exec --no-gpu -- pytest -q
remote-run pull <run_id> ./pulled-run --path build/summary.json
```

See [docs/usage.md](docs/usage.md) for the complete v2 CLI, MCP, and schema reference.
