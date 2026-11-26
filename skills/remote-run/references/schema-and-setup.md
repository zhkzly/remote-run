# 2-File Schema, Validation Rules & Multi-Harness Setup

## 1. Global Host Config (`~/.config/remote-run/hosts.toml`)

Override path with `$REMOTE_RUN_CONFIG` or `--config`. Strict validation rejects unknown keys immediately.
You can bootstrap this file automatically with `remote-run init <ssh-alias-or-user@host>`.

### Level 0 — Minimal Required Fields (CPU, any language, zero Python assumption)
```toml
schema_version = 1

[hosts.server]
ssh = "my-server"    # Required: ~/.ssh/config alias or "user@host" (NO ":port", NO leading "-", NO spaces)
```
- `default_host`: optional at top level (defaults to the first `[hosts.<name>]` table).
- `root`: optional under `[hosts.<name>]` (defaults to `"~/.local/share/remote-run"`).

### Level 1 — Common Optional Fields (`[hosts.<name>]`)
| Key | Allowed Values / Rules | Default |
|---|---|---|
| `port` | Integer `1..65535` (optional custom SSH port) | SSH default / `~/.ssh/config` |
| `root` | Remote path starting with `/...` or `~/...` (forbids `/`, bare `~`/`~/`, `..`, remote `$HOME`, `$HOME` ancestors) | `"~/.local/share/remote-run"` |
| `device` | `"cpu"`, `"cuda"`, `"cuda:N"`, or `"cuda:N,M"` (unique GPU indices) | `"cpu"` |
| `default_env` | Host env name from `.envs.*`, or built-in `"venv"` / `"none"` | `""` (no host Python injected) |
| `path_prepend` | List of remote `/...` or `~/...` dirs added to `PATH` across all envs | `[]` (+ auto-discovered existing `~/.local/bin`, `~/.cargo/bin`, `~/go/bin`, `/usr/local/go/bin`) |

### Level 2 — Advanced Sections
- **Resource lock overrides (`[hosts.<name>]`)**:
  - `locks_dir`: remote `/...` or `~/...` dir (defaults to `<root>/locks`).
  - `gpu_slot`: identifier override for GPU lock filename (when omitted, CUDA derives sorted `gpu-N` lock(s), e.g., `["gpu-0", "gpu-1"]` for `cuda:0,1`; CPU never acquires GPU locks).
  - Tip: Prefer `--no-gpu` or `--device` over duplicating host blocks; only add a `[hosts.desktop_cpu]` alias if you want to pin `host = "desktop_cpu"` inside a project's `.remote-run.toml`.
- **Named environments (`[hosts.<name>.envs.<env>]`)**:
  - `python`: optional remote path (`/...` or `~/...`) — omit for non-Python toolchains (Rust/Go/Node/C++).
  - `path_prepend`: list of remote `/...` or `~/...` dirs prepended to `PATH`.
  - `[hosts.<name>.envs.<env>.env]`: string map of static env vars (note: `.env` table here vs `[env_vars]` in `.remote-run.toml`).
- **Host shared read-only assets (`[hosts.<name>.shared.<asset>]`)**:
  - `path` (required, `/...` or `~/...`), `manifest` (optional relative path), `sha256` (optional 64 hex chars), `version` (optional string).
  - Does **NOT** create symlinks; injects `REMOTE_RUN_SHARED_<ALIAS>_PATH`, `_NAME`, `_VERSION`, `_SHA256` and `{shared.<alias>.*}`.

*(Note: `remote-run` automatically uses a private `0700` SSH `ControlMaster` socket under `~/.cache/remote-run/ssh/` so no manual `ControlMaster` edit in `~/.ssh/config` is needed unless disabled via `REMOTE_RUN_SSH_CONTROL=0`.)*

---

## 2. Optional Project Config (`<repo_root>/.remote-run.toml`)

Only create when needed. Allowed keys ONLY:
- **Top-level**: `schema_version = 1`, `project` (fixed folder ID), `host`, `env` (`"venv"`, `"none"`, or host env), `workspace` (overrides auto `default` / worktree-branch naming), `remote_dir` (`/...` or `~/...`; cannot combine with `[share].project`).
- **`[sync]`**: `use_gitignore` (bool, default `true`), `ignore` (list), `preserve` (list of relative dirs).
- **`[share]`**: `project` (relative dirs symlinked across linked worktrees via `projects/<id>/shared/<dir>`), `host` (list of host asset names or `"alias=name"`).
- **`[env_vars]`**: string map supporting `{run_id}`, `{host}`, `{device}`, `{artifacts}`, `{code}`, `{python}`, `{git_commit}`, `{git_branch}`, `{shared.<alias>.path|version|sha256}` (placeholders expand ONLY here and in `--env-var`).

### Sync Content Categories
| Category | Uploaded? | Remote Ghost Cleanup / Delete Protection |
|---|---|---|
| Regular source files | Yes | Pruned on rename/delete only if tracked in `.remote-run-synced.json` & absent locally |
| `.gitignore` + `BUILTIN_IGNORE` + `[sync].ignore` | No | Never deleted if local file still exists |
| `BUILTIN_PRESERVE` (`.venv`, `outputs`, `checkpoints`, `.cache`) + `[sync].preserve` | No | Never uploaded and never deleted (`sync --delete` or ghost cleanup) |
| `[share].project` / `[share].host` | No | `project` symlinked + auto-preserved; `host` env-var/SHA-256 verified only (no symlinks) |

Validate anytime with `remote-run config`.

---

## 3. Installing for Codex, Claude Code, and Antigravity (`agy`)

1. **Expose CLI & MCP binaries** on `PATH` (replace `<repo>` with your clone path):
   ```bash
   ln -sf <repo>/bin/remote-run ~/.local/bin/remote-run
   ln -sf <repo>/bin/remote-run-mcp ~/.local/bin/remote-run-mcp
   ```
2. **Install Skill**:
   ```bash
   # Codex & Antigravity (agy)
   ln -sfn <repo>/skills/remote-run ~/.agents/skills/remote-run
   # Claude Code
   ln -sfn <repo>/skills/remote-run ~/.claude/skills/remote-run
   ```
3. **Register Stdio MCP Server**:
   ```bash
   codex mcp add remote-run -- remote-run-mcp
   claude mcp add -s user remote-run -- remote-run-mcp
   agy mcp add remote-run ~/.local/bin/remote-run-mcp
   ```
   - **Claude Code note**: Start a **new** Claude Code session after registering the MCP server so `mcp__remote-run__*` tools attach.
   - **Claude Code read-only allowlist (optional)**: Add `"mcp__remote-run__remote_hosts"`, `"mcp__remote-run__remote_config"`, `"mcp__remote-run__remote_doctor"`, `"mcp__remote-run__remote_status"` to `.claude/settings.json` `permissions.allow`.
