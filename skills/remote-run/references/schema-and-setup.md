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
| `path_prepend` | List of remote `/...` or `~/...` dirs prepended to `PATH` (for `~/.bashrc`-guarded tools like `nvm`/`conda` or toolchain overrides) | `[]` (login shell `PATH` inherited automatically unless `REMOTE_RUN_INHERIT_LOGIN_PATH=0`) |

### Level 2 — Advanced Sections
- **Resource locks (`[hosts.<name>]`)**:
  - `locks_dir`: owned private remote directory outside workspaces and other control paths. Defaults to `<root>/locks`.
  - `gpu_slot`: optional legacy lock identifier added to the sorted per-card `gpu-N` locks. CPU jobs skip card locks. Let active jobs using the previous lock rules finish before switching.
  - Tip: Prefer `--no-gpu` or `--device` over duplicating host blocks; only add a `[hosts.desktop_cpu]` alias if you want to pin `host = "desktop_cpu"` inside a project's `.remote-run.toml`.
- **Named environments (`[hosts.<name>.envs.<env>]`)**:
  - `python`: optional remote path (`/...` or `~/...`) — omit for non-Python toolchains (Rust/Go/Node/C++).
  - `path_prepend`: list of remote `/...` or `~/...` dirs prepended to `PATH`.
  - `[hosts.<name>.envs.<env>.env]`: string map of static env vars (note: `.env` table here vs `[env_vars]` in `.remote-run.toml`).
- **Host shared assets (`[hosts.<name>.shared.<asset>]`)**:
  - `path` (required, `/...` or `~/...`), `manifest` (optional relative path), `sha256` (optional 64 hex chars), `version` (optional string).
  - Injects `REMOTE_RUN_SHARED_<ALIAS>_PATH`, `_NAME`, `_VERSION`, `_SHA256` and `{shared.<alias>.*}`. Linux file permissions govern writes. Aliases must remain unique when converted to uppercase and underscores.

SSH multiplexing uses the actual endpoint and a private directory under `~/.cache/remote-run/ssh/`. Set `REMOTE_RUN_SSH_CONTROL=0` to disable it. Long socket paths also disable multiplexing. Set `REMOTE_RUN_INHERIT_LOGIN_PATH=0` on the client to disable optional login PATH discovery.

---

## 2. Optional Project Config (`<repo_root>/.remote-run.toml`)

Only create when needed. Allowed keys ONLY:
- **Top-level**: `schema_version = 1`, `project` (fixed folder ID), `host`, `env` (`"venv"`, `"none"`, or host env), `workspace` (overrides `default` or linked checkout naming), `remote_dir` (`/...` or `~/...`). A syncing workspace cannot combine `remote_dir` with `[share].project`.
- **`[sync]`**: `use_gitignore` (bool, default `true`), `ignore` (list), `preserve` (list of relative dirs).
- **`[share]`**: `project` (relative dirs symlinked across linked worktrees via `projects/<id>/shared/<dir>`), `host` (list of host asset names or `"alias=name"`).
- **`[env_vars]`**: string map supporting `{run_id}`, `{host}`, `{device}`, `{artifacts}`, `{code}`, `{python}`, `{git_commit}`, `{git_branch}`, `{shared.<alias>.path|version|sha256}` (placeholders expand ONLY here and in `--env-var`).

### Sync Content Categories
| Category | Uploaded? | Remote Ghost Cleanup / Delete Protection |
|---|---|---|
| Regular source files | Yes | Default cleanup requires a matching registered remote version and an absent, unignored local path |
| Git ignores, built-in exclusions and `[sync].ignore` | No | Protected from default cleanup when the current rules match |
| `BUILTIN_PRESERVE` (`.venv`, `outputs`, `checkpoints`, `.cache`) + `[sync].preserve` | No | Never uploaded and never deleted (`sync --delete` or ghost cleanup) |
| `[share].project` / `[share].host` | No | `project` symlinked + auto-preserved; `host` env-var/SHA-256 verified only (no symlinks) |

Validate anytime with `remote-run config`.

---

## 3. Installing for Codex, Claude Code, and Antigravity (`agy`)

1. **Expose CLI & MCP binaries** on `PATH` (replace `<repo>` with your clone path):
   ```bash
   mkdir -p ~/.local/bin
   export PATH="$HOME/.local/bin:$PATH"
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
   - **Client restart & upgrade note**: Long-lived `remote-run-mcp` stdio processes do not hot-reload. Start a **new** session or restart the MCP client (Codex, Claude Code, `agy`) after registering the server and after upgrading `remote-run` so updated `remote_*` schemas and options attach.
   - **Claude Code allowlist (recommended)**: Add `permissions.allow` in **user-global `~/.claude/settings.json`** (so linked Git / Orca worktrees inherit it automatically) rather than only per-repo `.claude/settings.json`:
     - Safe read-only inspection tools: `"mcp__remote-run__remote_hosts"`, `"mcp__remote-run__remote_config"`, `"mcp__remote-run__remote_status"`.
     - Diagnostic probes: `"mcp__remote-run__remote_doctor"` is non-destructive (`destructiveHint: false`) but can execute configured Python/CUDA probes (`readOnlyHint: false`); add it to `permissions.allow` too if you trust your configured hosts/envs.
     - Routine remote execution (optional, for trusted personal hosts): add `"mcp__remote-run__remote_sync"`, `"mcp__remote-run__remote_exec"`, `"mcp__remote-run__remote_pull"` if you want `auto` mode to run remote builds/training without hitting the safety classifier. Keep `remote_cancel` / `remote_prune` / `remote_session` gated unless you explicitly want them auto-approved.
