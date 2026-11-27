# `remote-run` v2 Schema, Multi-Harness Setup & v1 Migration

## 1. Three-Layer Schema (`schema_version = 2`)

### A. `~/.config/remote-run/hosts.toml` (Machines Only)
```toml
schema_version = 2
default_host = "builder"

[hosts.builder]
ssh = "linux-builder"                   # ~/.ssh/config alias or "user@host"
# port = 22
# root = "~/.local/share/remote-run-v2"
# path = ["~/.local/bin", "~/.cargo/bin"]

# [hosts.builder.env]
# CARGO_HOME = "/srv/cache/cargo"

# Optional: omit on pure CPU machines
[hosts.builder.gpu]
devices = [0, 1]
# locks_dir = "~/.local/share/remote-run-v2/locks"
```

### B. `<repo>/.remote-run.toml` (Committable Project Policy, Optional)
```toml
schema_version = 2
project = "my-project"
path = ["build/bin", "node_modules/.bin"]

[env]
RUST_BACKTRACE = "1"

[workspace]
shared = ["tool-cache"]

[sync]
exclude = ["docs/generated/"]
preserve = ["build", "target"]
max_untracked_mib = 256

[pull]
max_file_mib = 10
exclude = ["**/*.trace"]
```

### C. Private XDG Bindings (`~/.config/remote-run/bindings/`)
Managed via `remote-run bind` / `remote_bind` (never dirties the Git repo). Every host-bound binding automatically records `host_endpoint` (`ssh|port|root`) so host-specific fields (`gpus`, `note`, `path`, `env`) never leak if the host alias is retargeted:
- `--scope repo`: keyed by `git-common-dir`, inherited by **all linked Git worktrees**.
  ```bash
  remote-run bind --scope repo --host builder --gpus 0 \
    --note "Prebuilt Python env ready; invoke python directly" \
    --path /opt/envs/proj/bin \
    --env VIRTUAL_ENV=/opt/envs/proj \
    --env PYTHONUNBUFFERED=1 \
    --env DATA_ROOT=/srv/data/v2
  ```
- `--scope checkout`: keyed by worktree root, overrides `workspace` and/or host-bound settings for one worktree.
  ```bash
  remote-run bind --scope checkout --host builder --workspace exp-a --gpus 1
  ```

---

## 2. Multi-Harness MCP Setup (Claude Code / Codex / Antigravity)

```bash
claude mcp add -s user remote-run -- ~/.local/bin/remote-run-mcp
codex mcp add remote-run -- ~/.local/bin/remote-run-mcp
agy mcp add remote-run ~/.local/bin/remote-run-mcp
```

### Codex `SSH_AUTH_SOCK` Requirement
Codex strips parent environment variables for stdio MCP servers by default. If your SSH login relies on `ssh-agent`, add `env_vars = ["SSH_AUTH_SOCK"]` in `~/.codex/config.toml`:

```toml
[mcp_servers.remote-run]
command = "/path/to/remote-run-mcp"
env_vars = ["SSH_AUTH_SOCK"]
```

---

## 3. Manual Migration Guide (v1 → v2)

v2 uses a separate default remote root (`~/.local/share/remote-run-v2`) so in-flight v1 runs are never disturbed.

| v1 Setting | v2 Replacement |
| --- | --- |
| `schema_version = 1` | `schema_version = 2` |
| `hosts.<h>.device = "cuda:0"` + `gpu_slot` + `locks_dir` | `[hosts.<h>.gpu]` with `devices = [0]` and optional `locks_dir` |
| `hosts.<h>.path_prepend` | `hosts.<h>.path` |
| `hosts.<h>.envs.<name>` (`python`, `path_prepend`, `env`) | `remote-run bind --scope repo --host <h> --path ... --env ... --note ...` |
| `hosts.<h>.shared.<name>` + `.remote-run.toml [share].host` | Ordinary `--env KEY=/remote/path` in repo binding (or `hosts.<h>.env` if truly machine-wide) |
| `.remote-run.toml` `host` / `env` / `workspace` | `remote-run bind --scope repo` (host/env/path/gpus) and `--scope checkout --workspace ...` |
| `.remote-run.toml [share].project` | `.remote-run.toml [workspace].shared` |
| `.remote-run.toml [sync].ignore` | `.remote-run.toml [sync].exclude` (plus `max_untracked_mib`) |
| `exec --device cuda:0,1` / `--env ...` | `exec --gpus 0,1` / `--path` / `--env KEY=VAL` |
| `remote_exec` `command` | `remote_exec` `argv` (strictly mutually exclusive with `shell`) |
| `pull --all-artifacts` / `--max-file-mb` | `pull --max-file-mib 0` or `--artifact REL_PATH` |
| `doctor --check-import` / `--check-cuda` | `doctor --check-cmd ...` (read-only) or explicit `exec --no-sync -- ...` |
