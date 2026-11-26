---
name: remote-run
description: >-
  Offload arbitrary commands, tests, builds, scripts, and GPU training from a local
  laptop workspace to a configured remote compute host, sync local code safely, manage
  opt-in persistent remote sessions, and pull results back. Use when the user asks to
  run, test, build, debug, monitor, or train on their desktop or remote compute machine.
---

# remote-run

Local editing and quick checks stay on the laptop. Use the `remote-run` CLI (or `remote_*` MCP tools only if already present in your active toolset) whenever the user wants work executed on a remote compute host (e.g., `desktop` or `cpu_box`).

## Safety Rules (Read First)

1. **Code lives locally; sync is one-way and additive by default**: `remote-run exec` and `remote-run sync` **never delete** files on the remote host automatically. Remote-owned directories (`.venv/`, `outputs/`, `checkpoints/`, `.cache/`, plus `.remote-run.toml` `[sync].preserve` and `[share].project`) are both excluded from upload and protected from deletion.
2. **Ownership markers protect existing data**: Managed roots (`.remote-run-root.json`) and workspaces (`.remote-run-workspace.json`) verify project/workspace identity before syncing. Syncing into a pre-existing non-empty unmanaged folder or another project's workspace is rejected in code.
3. **Never run destructive actions without explicit user confirmation**:
   - Prefer managed workspaces; do **not** pass `--remote-dir` unless the user explicitly provides a remote path.
   - **Never** run `remote-run sync --delete`, `remote-run sync --adopt`, `remote-run doctor --init-root`, `remote-run prune --apply`, or destructive `--shell` commands (`rm -rf`) without first running `--dry-run` (for `sync`/`prune`), showing the user what will be affected, and getting explicit approval.
   - If local `.py` or test files were deleted/renamed and you suspect stale files on the remote host, run `remote-run sync --dry-run --delete` first and ask before deleting.

## 1. Everyday Zero-Config Usage

1. **Discover hosts, inspect effective config, & check readiness**:
   ```bash
   remote-run hosts --pretty
   remote-run config                              # inspect effective_ignore, effective_preserve, remote workdir
   remote-run doctor                              # returns host, env, GPU details, python
   remote-run doctor --env venv                   # check host tools without requiring host python
   remote-run doctor --check-import torch --check-cuda
   ```
2. **Run commands on the remote host**:
   - `exec` is **stateless by default** (runs cleanly in the remote workspace root or matching subdirectory) and auto-detaches if a job takes longer than `--wait` (default `15`s).
   - **Environments**: uses host `default_env` unless `--env` is set. Built-in `--env venv` automatically activates `<remote_workspace>/.venv/bin` if present (ideal for `uv sync` / `python -m venv .venv`); built-in `--env none` runs without injecting a host Python.
   - **Exit codes**: remote failure (`failed`/`cancelled`/`interrupted`) prints the full JSON receipt to stdout and exits `2`; local config/SSH errors exit `1`. Pass `--print-log` for human-readable terminal stack traces (writes metadata JSON to stderr), or `--pretty` for indented JSON.
   ```bash
   remote-run exec -- pytest -q
   remote-run exec --env venv -- uv run pytest
   remote-run exec -- python train.py --epochs 10
   remote-run exec --host cpu_box -- bash run.sh "arg with space"
   ```
   - **Quick monitoring** (skips upload and GPU lock):
     ```bash
     remote-run exec --no-sync --no-gpu -- nvidia-smi
     ```
   - **Opt-in persistent shell** (saves `cwd` and env diffs across calls when `exit_code == 0`):
     ```bash
     remote-run exec --session dev --shell "cd sub && export TAG=v1"
     remote-run exec --session dev --no-sync -- python train.py
     remote-run session reset dev
     ```
3. **Environment variables injected on the remote host**:
   - `REMOTE_RUN_ID`, `REMOTE_RUN_HOST`, `REMOTE_RUN_DEVICE` (`cuda:0` or `cpu`; multi-GPU `cuda:N` sets `CUDA_VISIBLE_DEVICES=N` unless overridden), `REMOTE_RUN_ARTIFACTS` (job-isolated artifact dir), `REMOTE_RUN_CODE` (remote workspace dir), `REMOTE_RUN_PYTHON`
   - For bound host shared assets (`--shared [ALIAS=]NAME` or `[share].host`): `REMOTE_RUN_SHARED_<ALIAS>_PATH`, `_NAME`, `_VERSION`, `_SHA256`
4. **Long-running jobs, status, cancel, pull, and prune**:
   - **Run statuses**: `queued`, `running`, and terminal states `completed`, `failed`, `cancelled`, `interrupted`.
   - When `detached: true`, immediately report `run_id`, `host`, `device`, `workdir`, and `next_action`.
   ```bash
   remote-run status RUN_ID --wait 30
   remote-run status --repo .                     # list recent runs for this project
   remote-run status RUN_ID --offset 8192 --raw-log
   remote-run cancel RUN_ID                       # kills process group (even while queued), frees VRAM
   remote-run pull RUN_ID ./results/RUN_ID
   remote-run pull RUN_ID ./results/RUN_ID --path reports/bench.md --force
   remote-run sync --dry-run --delete             # preview stale remote files before explicit --delete
   remote-run prune --days 14                     # dry-run; ask user before running --apply
   ```
   - **Pull directory layout**: `DEST/` contains `request.json`, `state.json`, `run.log`, `artifacts/` (from `$REMOTE_RUN_ARTIFACTS`), and `workspace/` (when `--path` is passed for files/dirs). `--path` reads from the live shared workspace rather than a frozen run snapshot, whereas `artifacts/` is isolated per `run_id`. Adding `/results/` to local `.gitignore` keeps `git status` clean when pulling to `./results/<run_id>`.
   - Without `--all-artifacts`, files in `artifacts/` with a checkpoint suffix (`.pt`, `.pth`, `.ckpt`, `.bin`, `.safetensors`, `.onnx`, `.h5`) **OR** exceeding `--max-file-mb` (default `10` MB) are skipped and listed in `skipped_artifacts`.
   - `--force` safely merges pulled files into an existing `DEST` without deleting unrelated local files.

## 2. Optional 2-File Config & Multi-Harness Setup

There are **only two** config files (`~/.config/remote-run/hosts.toml` and optional `<repo_root>/.remote-run.toml`). Never create ad-hoc dotfiles. Always run `remote-run config` after writing `.remote-run.toml` to validate keys.

For exact TOML schemas, multi-host CPU/GPU setup, `ControlMaster` tips, and installation on **Codex**, **Claude Code**, and **Antigravity (`agy`)**, read [references/schema-and-setup.md](./references/schema-and-setup.md).
