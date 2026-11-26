---
name: remote-run
description: >-
  Offload arbitrary commands, builds, tests, scripts, and GPU training from a local
  laptop workspace to any configured remote Linux host, sync code safely without ghost
  files, manage opt-in persistent sessions, and pull results back. Use when the user asks
  to run, test, build, debug, monitor, or train on a remote Linux host or GPU desktop.
---

# remote-run

Local editing stays on the laptop. When `remote_*` MCP tools are already loaded in your toolset, prefer them over shelling out via `Bash` (they support per-tool allowlisting in auto/permission-guarded modes and return `tail_bytes` logs directly without shell pipelines). Use the `remote-run` CLI for CLI-only actions (`init`, explicit sync `--delete`/`--adopt`, `doctor --init-root`, `--follow`, `--print-log`). Review the [CLI and MCP reference](../../docs/usage.md#cli-and-mcp-reference).

## Safety Rules (Read First)

1. **4-category sync protects remote data**:
   - Default cleanup requires an absent and unignored local path and a matching registered remote file version. Unverified copies are kept.
   - Workspace-root `.venv/`, `outputs/`, `checkpoints/`, `.cache/` and paths in `[sync].preserve` or `[share].project` are excluded from upload and managed cleanup. Common local credentials and project config are excluded from upload.
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
   remote-run doctor --repo . --env pytorch --check-cuda --check-import torch  # with the named env configured
   ```
   - Use `doctor` / `remote_doctor` (`--check-cmd`, `--check-import`, `--check-cuda` / `check_cmds`, `check_imports`, `check_cuda`) instead of ad-hoc `exec --no-sync` commands to probe remote binaries, Python packages, shared assets, or GPU readiness.
   - Set `REMOTE_RUN_INHERIT_LOGIN_PATH=0` on the client to disable optional login PATH discovery. Noninteractive SSH must already find Python 3.11+. Doctor can execute configured Python and GPU probes.

2. **Run commands on the remote host**:
   - `exec` is **stateless by default** (isolated `tmux` + PGID, auto-detaches if longer than `--wait`, default `15`s).
   - Single checkouts use `default`. Linked Git worktrees isolate per checkout and share `[share].project`. Sync rejects a workspace with active jobs. Add build caches to `[sync].preserve` explicitly.
   - CPU jobs skip card locks. `--no-gpu` clears `CUDA_VISIBLE_DEVICES`. A CUDA job takes sorted per-card locks and any configured legacy lock. Coordination is scoped to the lock directory.
   - **Environments**: uses project `env` or host `default_env`, or pass `--env <name>` / `--env venv` (`<workdir>/.venv/bin`) / `--env none`.
   - `exec` and `status --follow` exit 2 for terminal task failures. Plain status exits 0 on successful inspection. Detach exits 0, runtime and SSH errors exit 1, and Ctrl-C stops the local wait with 130. Follow streams stdout and places metadata on stderr.
   ```bash
   remote-run exec --follow -- cargo test
   remote-run exec --env venv -- uv run pytest -q
   remote-run exec -- python train.py --epochs 10
   remote-run exec --device cuda:0,1 -- torchrun --nproc_per_node=2 train.py
   remote-run exec --no-gpu -- bash run.sh "arg with space"
   ```
   - **Direct remote actions without laptop upload**:
     - Use `--no-project --no-sync` (`no_project=true, no_sync=true`) for an ad-hoc host check before the project has been synced (ignores `.remote-run.toml`).
     - Use `--no-sync --remote-dir <path>` **without** `--no-project` to run on an explicit directory (e.g., a host shared asset path from `hosts`/`doctor`) while **keeping `.remote-run.toml`'s `host`, `env`, `[share].host`, and `[env_vars]`** before the first sync.
     - For MCP calls (`remote_doctor`, `remote_exec` even with `no_sync=true`, `remote_status`), always pass `repo` explicitly whenever `.remote-run.toml` settings should apply.
     ```bash
     remote-run exec --no-project --no-sync --no-gpu -- nvidia-smi
     remote-run exec --no-sync --no-gpu --remote-dir /path/on/remote --shell "hf download ..."
     ```
   - **Opt-in persistent shell** (serializes calls per session, saves cwd and exported environment changes on success, and resets only while idle):
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
   remote-run cancel RUN_ID                       # signal the verified process group and report observed state
   remote-run pull RUN_ID ./results/RUN_ID
   remote-run pull RUN_ID ./results/RUN_ID --path reports/bench.md --paths-only --force
   remote-run sync --dry-run --delete
   remote-run prune --days 14                     # dry-run; ask user before --apply
   ```
   - `pull` applies checkpoint suffix and 10 MiB per-file limits to automatically selected artifacts. Explicit workspace paths are independent. Private request metadata is excluded from default pull. Prune skips jobs without a run ownership marker.

## 2. Optional 2-File Config & Multi-Harness Setup

There are **only two** config files (`~/.config/remote-run/hosts.toml` and optional `<repo_root>/.remote-run.toml`). Never create ad-hoc dotfiles. Always run `remote-run config` after writing `.remote-run.toml`.

For exact Level 0 / Level 1 / Level 2 schemas, validation rules, and Codex / Claude Code / Antigravity (`agy`) setup, read [references/schema-and-setup.md](./references/schema-and-setup.md).
