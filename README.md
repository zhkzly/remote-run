# remote-run

Lightweight, zero-dependency remote compute offloader for coding locally on a thin laptop (with `agy`, `Codex`, or `Claude Code`) and executing tests, builds, scripts, and GPU training on a remote Linux machine.

## Highlights

- **Isolated execution + opt-in persistent shell**: `tmux` + independent process groups (`PGID`), automatic background detach after `--wait` (default 15s), `--print-log` / `--pretty` debugging flags, and opt-in `--session <name>` persistence for `cwd` and environment diffs.
- **Safe, additive-by-default sync**: incremental `rsync -rlptz` from repository root **without `--delete` by default** (explicit `sync --dry-run --delete` / `sync --delete` when needed), dual ownership markers (`.remote-run-root.json` & `.remote-run-workspace.json`) with dynamic topology isolation across any machine, remote-owned `preserve` protection (`.venv/`, `outputs/`, `checkpoints/`, `.cache/`, `[share].project`), and automatic upload `ignore` (`.git/`, local `.venv/`, `__pycache__/`, `/results/`, `.gitignore`, project `ignore`).
- **Works for any language/workflow**: built-in `--env venv` (project `.venv`), `--env none`, host-level `path_prepend` (`uv`, `cargo`, etc.), and whitelist-safe `prune` for remote disk hygiene.
- **Move-safe & worktree-safe**: remote workspace identity is derived from common repo + branch (or `workspace = "default"` / `--remote-dir`), never local laptop absolute paths.
- **GPU serialization & clean cancel**: physical GPU `flock` queuing, automatic `CUDA_VISIBLE_DEVICES` isolation, `--no-gpu` bypass for quick inspection, and process-group `SIGTERM`/`SIGKILL` cancellation (including queued jobs) that frees VRAM without touching other jobs.
- **Strict 2-file config**: `~/.config/remote-run/hosts.toml` + optional `<repo_root>/.remote-run.toml`, strictly schema-validated.
- **Standardized across harnesses**: `bin/remote-run` CLI, `skills/remote-run/` (`agy` / `Codex` / `Claude Code`), and `bin/remote-run-mcp` stdio MCP server.

See [docs/usage.md](docs/usage.md) and [skills/remote-run/references/schema-and-setup.md](skills/remote-run/references/schema-and-setup.md) for setup and examples.
