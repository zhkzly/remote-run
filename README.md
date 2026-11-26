# remote-run

**English** | [简体中文](README.zh-CN.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib)-brightgreen.svg)](remote_run.py)
[![MCP Server](https://img.shields.io/badge/MCP-stdio%20server-purple.svg)](remote_run_mcp.py)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-orange.svg)](skills/remote-run/SKILL.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **Edit locally with Claude Code, Codex, or Antigravity (`agy`). Execute transparently on any remote Linux host—CPU or GPU, any language, zero remote AI setup.**

Keep your editor, AI agents and MCP services on your laptop. Upload the project files needed for the run. Turn any Linux server, lab desktop, or cloud VM into an invisible remote compute backend with almost zero configuration.

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
  ├─ Connectivity & Env : user@host / port / ~/paths, auto login-shell PATH inheritance, venv/named envs
  ├─ Workspace & Flow   : branch-safe single checkouts, auto-isolated Git worktrees, 4-class sync safety
  ├─ Execution          : stateless tmux or opt-in --session, live --follow streaming, clean PGID cancel
  └─ Optional Resources : lock-free on CPU; automatic sorted flock + CUDA isolation on GPU (--device / --no-gpu)
```

---

## Common Use Cases

1. **Any-language edit–test–build loop**: Write **Rust, Go, C++, Node/TS, or Python** locally and run `remote-run exec --follow -- cargo test` or `pytest -q`. Owned-file manifest tracking (`.remote-run-synced.json`) safely removes ghost files when you delete or rename local source files—when the remote copy still matches its recorded version. Configure `[sync].preserve` for remote outputs and caches.
2. **Normal checkouts keep caches across branches**: Switching branches in a single local checkout keeps `default` on the remote host (keeping excluded and preserved caches in place). Linked Git worktrees automatically isolate by checkout identity while sharing `[share].project`.
3. **Zero-setup rented cloud VMs & lab GPUs**: Keep AI keys and prompts on your laptop; onboard any box in seconds with `remote-run init user@host --port 22345` (zero `sudo`, no remote daemon).
4. **Long-running GPU jobs.** Jobs start in remote `tmux` and continue after the local `--wait` window (default `15s`). CUDA jobs take sorted per-card locks in the configured lock directory. `cancel` signals the verified process group. Use `--no-gpu` to skip card locks and clear `CUDA_VISIBLE_DEVICES` for programs that honor it.
5. **AI Agent native (CLI + MCP + Skill)**: Exposes equivalent workflows via terminal CLI, standard-library stdio MCP (`remote_*`), and an Agent Skill. See the [CLI and MCP reference](docs/usage.md#cli-and-mcp-reference) for supported fields.

---

## Quickstart

### Prerequisites

- **Laptop**: Linux or macOS, Bash, Python 3.11+ on `PATH`, `ssh` and `rsync`. Git supplies project identity and Git ignore rules
- **Remote Linux host**: Python 3.11+, `bash`, `tmux`, `rsync` (no `git` or `sudo` needed on the remote host), batch-mode SSH auth and an already verified host key in `known_hosts` (`ssh -o BatchMode=yes <target> true`)

### 1. Clone and link CLI & MCP binaries

```bash
git clone https://github.com/zhkzly/remote-run.git ~/remote-run
mkdir -p ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
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
# Add ~/.local/bin to your shell startup PATH for future shells.
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

### 4. Run from your local repo

```bash
# Sync changes and stream remote output live (works for Rust, Go, C++, Node, Python...)
remote-run exec --follow -- cargo test
remote-run exec --env venv --follow -- uv run pytest

# Override GPU selection per run on a CUDA host (lock-free CPU or multi-GPU sorted locking)
remote-run exec --no-gpu -- pytest -q
remote-run exec --device cuda:0,1 -- torchrun --nproc_per_node=2 train.py

# Direct remote action without uploading laptop files (e.g., download dataset directly on host)
remote-run exec --no-gpu --no-sync --remote-dir '~/datasets/v1' --shell "hf download ... --local-dir ."

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

1. **Owned source cleanup.** Default sync removes an absent source only when the remote regular file matches its registered version. Keep remote-generated files in preserved paths and preview explicit mirroring with `sync --dry-run --delete`.
2. **Managed directories.** Root, workspace and run ownership markers identify where the tool may write or prune. Control directories require trusted paths. An existing non-empty root or workspace requires explicit initialization or adoption.
3. **Private state.** SSH sockets, runs, sessions and lock directories are private. Common local credential files are excluded. Inspection outputs redact configured environment values. Job output can still include whatever the command prints.
4. **Cleanup and retrieval.** `prune` previews by default and skips unmarked jobs. `pull` copies log, state and eligible artifacts, with a default 10 MiB cap per automatically selected artifact. Explicit workspace paths follow separate rules in [the usage guide](docs/usage.md).

---

## MCP Server Tools (`remote-run-mcp`)

Zero-dependency JSON-RPC 2.0 stdio server (`2025-06-18` / `2025-03-26` / `2024-11-05`) with concurrent request handling:

| Tool | Purpose | Safety boundary |
|---|---|---|
| `remote_hosts` | List hosts, devices, named environments, and shared assets | Read-only |
| `remote_config` | Inspect merged `hosts.toml` + `.remote-run.toml` config & rules | Read-only |
| `remote_doctor` | Probe SSH, `bash`/`tmux`/`rsync`, `--check-cmd`, GPUs, imports, shared SHA-256 | Optional executable probes |
| `remote_sync` | Incremental sync (+ safe ghost cleanup for deleted owned files) | May replace source and prune owned files. Preview with `dry_run` |
| `remote_exec` | Sync (unless `no_sync=true`) and run command/shell (`device`, `no_gpu`, `wait`) | PGID + `tmux` isolation |
| `remote_status` | Poll job state / byte-offset logs (`tmux_session` included) or list runs | Read-only |
| `remote_cancel` | Request cancellation and report observed status, keeping logs | `destructiveHint` |
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
