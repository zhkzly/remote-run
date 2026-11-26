---
name: remote-run
description: >-
  Offload arbitrary commands, builds, tests, scripts, and GPU training from a local
  laptop workspace to any configured remote Linux host, sync code safely without ghost
  files, manage opt-in persistent sessions, and pull results back. Use when the user asks
  to run, test, build, debug, monitor, or train on a remote Linux host or GPU desktop.
---

# remote-run

Local editing stays on the laptop. Use the `remote-run` CLI (or `remote_*` MCP tools when already loaded in your toolset) whenever work should execute on a configured remote Linux host.

**CLI vs. MCP differences**: For safety, MCP intentionally omits `sync --delete`, `sync --adopt`, `doctor --init-root`, and blocking `--follow`. Use `remote-run init`, `sync --delete`, `sync --adopt`, `doctor --init-root`, and `--follow` from the CLI when needed. Both `remote_exec` and `remote_sync` automatically clean up previously uploaded ghost files after local deletes/renames.

## Safety Rules (Read First)

1. **4-category sync protects remote data**:
   - Regular synced files deleted/renamed locally are safely removed on remote via `.remote-run-synced.json` — newly ignored files that still exist locally are **never** deleted.
   - Remote-owned directories (`.venv/`, `outputs/`, `checkpoints/`, `.cache/`, plus `.remote-run.toml` `[sync].preserve` and `[share].project`) are excluded from upload and never deleted.
2. **Ownership markers protect existing data**: Managed roots (`.remote-run-root.json`) and workspaces (`.remote-run-workspace.json`) verify identity before syncing. Syncing into a pre-existing non-empty unmanaged folder or another project's workspace is rejected in code.
3. **Never run destructive actions without explicit user confirmation**:
   - Prefer managed workspaces; do **not** pass `--remote-dir` unless the user gives an explicit remote path.
   - **Never** run `remote-run sync --delete`, `remote-run sync --adopt`, `remote-run doctor --init-root`, `remote-run prune --apply`, or destructive `--shell` commands (`rm -rf`) without first running `--dry-run` (for `sync`/`prune`), showing the user what will be affected, and getting explicit approval.

## 1. Everyday Zero-Config Usage

1. **Onboard a new host, discover hosts, inspect config, & check readiness**:
   ```bash
   remote-run init my-server                                # probe host & add to hosts.toml
   remote-run hosts                                         # compact JSON summary
   remote-run config                                        # inspect workspace_id, ignore, preserve
   remote-run doctor --repo . --check-cmd bash --check-cmd rsync
   remote-run doctor --repo . --check-cmd cargo
   remote-run doctor --repo . --check-cuda --check-import torch
   ```
   - `PATH` on the remote host automatically includes `host.path_prepend`, env `path_prepend`, and any existing `~/.local/bin`, `~/.cargo/bin`, `~/go/bin`, `/usr/local/go/bin`.

2. **Run commands on the remote host**:
   - `exec` is **stateless by default** (isolated `tmux` + PGID, auto-detaches if longer than `--wait`, default `15`s).
   - **Workspace isolation**: Single-checkout repos default to `workspace_id = "default"` so `git checkout -b` keeps build caches (`target/`, `build/`, `.venv/`). Linked Git worktrees automatically isolate per branch and share `[share].project`.
   - **Device & GPU concurrency**: CPU hosts run lock-free with no CUDA env vars injected. CUDA hosts default to host `device` slot lock(s); override per run with `--no-gpu` (lock-free, `CUDA_VISIBLE_DEVICES=""`) or `--device cuda:N[,M...]` (acquires sorted slot locks, e.g., `gpu-0` + `gpu-1` for `cuda:0,1`).
   - **Environments**: uses project `env` or host `default_env`, or pass `--env <name>` / `--env venv` (`<workdir>/.venv/bin`) / `--env none`.
   - **Exit codes**: remote failure (`failed`/`cancelled`/`interrupted`) prints JSON receipt and exits `2`; `exec` exceeding `--wait` detaches cleanly and exits `0` (`detached: true`); local config/SSH errors exit `1`. Pass `--follow` for live terminal streaming (prints metadata JSON to stderr) or `--print-log` / `--pretty`.
   ```bash
   remote-run exec --follow -- cargo test
   remote-run exec --env venv -- uv run pytest -q
   remote-run exec -- python train.py --epochs 10
   remote-run exec --device cuda:0,1 -- torchrun --nproc_per_node=2 train.py
   remote-run exec --no-gpu -- bash run.sh "arg with space"
   ```
   - **Direct remote actions without laptop upload** (pass `--no-sync`, plus `--no-gpu` for light/download tasks, and `--remote-dir` only when user specifies a path):
     ```bash
     remote-run exec --no-sync --no-gpu -- nvidia-smi
     remote-run exec --no-sync --no-gpu --remote-dir /path/on/remote --shell "hf download ..."
     ```
   - **Opt-in persistent shell** (saves `cwd` and env diffs across calls when `exit_code == 0`):
     ```bash
     remote-run exec --session dev --shell "cd sub && export TAG=v1"
     remote-run exec --session dev --no-sync -- make test
     remote-run session list --repo .
     remote-run session reset dev
     ```

3. **Environment variables injected on the remote host**:
   - `REMOTE_RUN_ID`, `REMOTE_RUN_HOST`, `REMOTE_RUN_DEVICE` (`cpu`, `cuda:0`, or `cuda:0,1,...`), `REMOTE_RUN_ARTIFACTS` (per-run artifact dir), `REMOTE_RUN_CODE` (remote workspace), `REMOTE_RUN_PYTHON`
   - `REMOTE_RUN_GIT_COMMIT` (40-hex `HEAD`), `REMOTE_RUN_GIT_BRANCH`, `REMOTE_RUN_GIT_DIRTY` (`1`/`0`)
   - Bound host shared assets (`--shared [ALIAS=]NAME` or `[share].host`): `REMOTE_RUN_SHARED_<ALIAS>_PATH`, `_NAME`, `_VERSION`, `_SHA256`

4. **Monitor, attach, cancel, pull, and prune**:
   - Run statuses: `queued`, `running`, and terminal states `completed`, `failed`, `cancelled`, `interrupted`.
   - When `detached: true`, immediately report `run_id`, `host`, `device`, `workdir`, `tmux_session`, and `next_action`.
   - For long jobs in background-capable harnesses, run `remote-run status RUN_ID --wait 3600` in the background (or `remote-run status RUN_ID --follow` in interactive CLI). When polling manually, pass `--offset <next_offset>`.
   ```bash
   remote-run status RUN_ID --wait 30
   remote-run status RUN_ID --follow
   remote-run status --repo .
   remote-run cancel RUN_ID                       # kills process group & tmux session cleanly
   remote-run pull RUN_ID ./results/RUN_ID
   remote-run pull RUN_ID ./results/RUN_ID --path reports/bench.md --paths-only --force
   remote-run sync --dry-run --delete
   remote-run prune --days 14                     # dry-run; ask user before --apply
   ```
   - `pull` skips ML checkpoints (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`, `.h5`) and files > `--max-file-mb` (default `10` MB) unless `--all-artifacts` is passed. `.bin` artifacts within size limit are pulled normally.

## 2. Optional 2-File Config & Multi-Harness Setup

There are **only two** config files (`~/.config/remote-run/hosts.toml` and optional `<repo_root>/.remote-run.toml`). Never create ad-hoc dotfiles. Always run `remote-run config` after writing `.remote-run.toml`.

For exact Level 0 / Level 1 / Level 2 schemas, validation rules, and Codex / Claude Code / Antigravity (`agy`) setup, read [references/schema-and-setup.md](./references/schema-and-setup.md).
