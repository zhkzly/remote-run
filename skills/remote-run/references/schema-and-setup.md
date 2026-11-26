# 2-File Schema & Multi-Harness Installation

## 1. Global Host Config (`~/.config/remote-run/hosts.toml`)

Override path with `$REMOTE_RUN_CONFIG` or `--config`. Unknown keys are rejected.

- **Top-level**: `schema_version = 1`, `default_host`, `[hosts.<name>]`
- **`[hosts.<name>]`**: `ssh` (required), `root` (required, abs path), `device` (`cpu` | `cuda[:N]`), `gpu_slot`, `locks_dir`, `default_env`, `path_prepend` (list of remote abs dirs added to `PATH` across all envs, e.g., `/home/<user>/.local/bin` for `uv`, `/home/<user>/.cargo/bin`)
- **`[hosts.<name>.envs.<env>]`**: `python` (abs path), `path_prepend` (list), `env` (string map)
- **`[hosts.<name>.shared.<asset>]`**: `path` (required, abs path), `manifest` (rel path), `sha256` (64-hex), `version`

**Tip**: For a GPU desktop, define both `[hosts.desktop]` (`device = "cuda:0"`) and `[hosts.desktop_cpu]` (`device = "cpu"`, shared `root`) so tests/builds never wait behind GPU training locks.

**Tip**: Enable SSH `ControlMaster auto` / `ControlPersist 60` in `~/.ssh/config` to eliminate multi-second SSH handshake overhead during rapid edit-test loops.

## 2. Optional Project Config (`<repo_root>/.remote-run.toml`)

Only create when needed. Allowed top-level keys ONLY:
- `schema_version = 1`
- `project` (identifier; fixes remote project folder across local renames)
- `host` (default host for this repo)
- `env` (default env for this repo, e.g. `"venv"`, `"none"`, or host env name)
- `workspace` (identifier; e.g., `"default"` if you switch branches in-place and want to reuse `.venv` / build caches)
- `remote_dir` (explicit remote folder override; cannot be combined with `[share].project`)
- `[sync]`: `use_gitignore` (bool, default `true`), `ignore` (list), `preserve` (list)
- `[share]`: `project` (rel dirs shared across all worktrees of this project via symlinks into `.../shared/<name>`), `host` (list of host asset names or `"alias=name"`)
- `[env_vars]`: string map supporting `{run_id}`, `{host}`, `{device}`, `{artifacts}`, `{code}`, `{python}`, `{git_commit}`, `{git_branch}`, `{shared.<alias>.path|version|sha256}` (placeholders expand ONLY here and in `--env-var`, never in `--shell` or command args). On remote execution, `REMOTE_RUN_GIT_COMMIT`, `REMOTE_RUN_GIT_BRANCH`, and `REMOTE_RUN_GIT_DIRTY` (`1`/`0`) are also injected automatically.

**Tip (Multi-GB Shared Datasets)**: Copy large read-only datasets directly to the remote host once with `rsync -avh --progress ./local_data/ <ssh_host>:/path/to/shared_data/`, compute a `manifest.json` SHA-256 on the host, and register `[hosts.<name>.shared.<asset>]` in `hosts.toml` (define under each host alias sharing that machine).

Validate anytime with `remote-run config`.

## 3. Installing for Codex, Claude Code, and Antigravity (`agy`)

1. **Expose CLI & MCP binaries** on `PATH` (replace `<repo>` with your clone path, e.g., `~/pycodes/remote-run`):
   ```bash
   ln -sf <repo>/bin/remote-run ~/.local/bin/remote-run
   ln -sf <repo>/bin/remote-run-mcp ~/.local/bin/remote-run-mcp
   ```
2. **Install Skill** (follows Agent Skills spec; shared across agents):
   ```bash
   # Codex & Antigravity (agy)
   ln -sfn <repo>/skills/remote-run ~/.agents/skills/remote-run
   # Claude Code
   ln -sfn <repo>/skills/remote-run ~/.claude/skills/remote-run
   ```
3. **Register Stdio MCP Server** (optional; gives native `remote_*` structured tools):
   ```bash
   codex mcp add remote-run -- remote-run-mcp
   claude mcp add -s user remote-run -- remote-run-mcp
   agy mcp add remote-run ~/.local/bin/remote-run-mcp
   ```
   - **Claude Code note**: Start a **new** Claude Code session after registering the MCP server so `mcp__remote-run__*` tools attach (resumed sessions hot-reload `SKILL.md` but not newly added MCP servers).
   - **Claude Code read-only allowlist (optional)**: Add `"mcp__remote-run__remote_hosts"`, `"mcp__remote-run__remote_config"`, `"mcp__remote-run__remote_doctor"`, `"mcp__remote-run__remote_status"` to `.claude/settings.json` `permissions.allow` to skip prompts and avoid `auto`-mode classifier stalls on read-only checks.
