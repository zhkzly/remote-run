# remote-run Usage Guide

Use the CLI or stdio MCP to run commands on an SSH-configured Linux host. The project uses Python 3.11+ standard library. The remote host needs Bash, tmux and rsync already installed. The local host needs Bash, ssh and rsync on PATH. Git is optional for plain directories and supplies identity and ignore rules for Git checkouts. Remote Python 3.11+ must be available as `python3` to a plain noninteractive SSH command. Keep noninteractive shell stdout clear for rsync.

## Configure a host

Confirm its SSH host key and noninteractive authentication before running `init`. For an IPv6 host use an SSH config alias. Add `~/.local/bin` to your shell PATH when installing the two launchers.

```bash
remote-run init my-server
remote-run hosts
remote-run doctor --check-cmd cargo --pretty
```

`init` probes tools before adding a host to `hosts.toml`. A CPU host works without a Python environment configuration or CUDA tools. For Python import and CUDA probes, configure or select a Python environment. A project venv probe needs `doctor --repo . --env venv` and an existing remote `.venv`.

```bash
remote-run doctor --repo . --env pytorch --check-cuda --check-import torch
```

An existing non-empty remote root requires `doctor --init-root` after you verify it belongs to this tool. By default doctor inspects the root without creating it. Executable probes can run the configured interpreter, import modules or run NVIDIA diagnostics. Use trusted host and project configurations. Set `REMOTE_RUN_INHERIT_LOGIN_PATH=0` on the client to disable optional login PATH discovery. Remote Python bootstrap still uses the SSH command environment.

## Sync and workspace identity

| Content | Default behavior |
|---|---|
| Source | rsync sends non-excluded files, including eligible untracked files. Same-path remote source can be replaced. |
| Ignored | Git checkouts and initialized submodules use their Git ignore rules for untracked files. Plain directories use rsync exclude syntax in `.gitignore`. Built-in rsync excludes and `[sync].ignore` also apply. |
| Local credentials | Common `.env` paths, `.ssh`, `.aws`, and the project `.remote-run.toml` are excluded. Put other local-only files in an ignore rule. |
| Preserved | Workspace-root `.venv`, `outputs`, `checkpoints`, `.cache`, relative `[sync].preserve` paths and project shares are excluded and protected from managed cleanup. Add `target` or `build` to protect those caches independently of ignore patterns. |
| Shared | `[share].project` mounts project directories into workspaces. `[share].host` supplies paths and optional checksum verification. Linux file permissions determine whether a program can write to the asset. |

Default ghost cleanup requires a previously uploaded regular file, a matching recorded remote version, and an absent local path that is outside the ignore and preserve rules. Old path-list manifests contain insufficient information for removal. An interrupted transfer can leave already copied files. A failed transfer does not advance the ownership manifest. Explicit `sync --delete` can delete unregistered files. Add remote-only paths to `[sync].preserve` to protect them. Git global excludes and `.git/info/exclude` are local to the sender. Use `sync --dry-run --delete` before applying it.

A single checkout uses `default`. A linked Git worktree gets a separate workspace based on its checkout identity, including while detached. Existing workspaces from earlier branch-based naming stay where they are. Choose the old ID with `--workspace` if you intend to resume it. A submodule is discovered as its own Git checkout when invoked from inside it. The parent copy omits nested Git control files. Bare repositories are rejected. Set `project` for an empty Git history or to preserve identity across checkout renames.

Sync refuses active workspaces and overlapping active paths. Writers using the same configured remote root serialize workspace preparation, upload and submission for overlapping paths. Use a linked worktree or a separate `--workspace` to build concurrently with a long run. Existing non-empty unmanaged directories require `sync --adopt` to accept uploads. Other processes with the same Linux account can change files outside this lock. Direct commands run with that account's filesystem permissions.

## Run, follow and recover

```bash
remote-run exec --follow -- cargo test
remote-run exec --env venv -- uv run pytest -q
remote-run exec --no-gpu --no-sync --remote-dir '~/datasets/v1' --shell 'hf download ...'
remote-run exec --device cuda:0,1 -- torchrun --nproc_per_node=2 train.py
```

`--no-sync` skips upload and normally retains project settings. An implicit project workspace must already exist, except when a session creates its workspace. Passing `--no-sync --remote-dir <path>` without `--no-project` runs on an explicit directory (such as a host shared asset path) before the first sync while still keeping project `host`, `env`, and shared settings. From the local home directory or with explicit `--no-project --no-sync`, execution uses an ad hoc workspace under the remote root. Use a quoted `~/…` for remote home expansion. When calling MCP, always pass `repo` (even with `no_sync=true`) whenever project settings are needed.

CPU is the default. A CUDA run takes one lock per physical index in sorted order and remaps the indices inside the child. A configured legacy `gpu_slot` adds another lock. Run receipts expose the requested `physical_device` and report a CPU job as `device: cpu`. `--no-gpu` sets an empty `CUDA_VISIBLE_DEVICES` and skips card locks. The same applies to `--device cpu` on a CUDA-profile host. Locks coordinate jobs in the same lock directory. CUDA environment variables affect programs that honor them. Separate accounts and roots require a scheduling policy outside this tool.

Jobs start in tmux. `--wait` is the local terminal wait window, default 15 seconds. `--wait 0` returns a submission receipt without logs. `--follow` streams from the beginning with original line breaks and terminal CR progress. `--raw-log` retains exact bytes in a terminal stream or in the snapshot field `log.data` (Base64). Log offsets count bytes. Commands receive EOF on stdin. Output buffering depends on the program.

```bash
remote-run --host server status RUN_ID --wait 30 --print-log
remote-run --host server status RUN_ID --follow
remote-run status --repo . --limit 20
remote-run --host server cancel RUN_ID
```

The CLI reports the submission identity to stderr before waiting. Ctrl-C stops the local wait. A job already submitted continues until completion or explicit cancellation. `cancel` signals the verified process group and reports whether cancellation was requested. A legacy job without process identity can be cancelled through its tmux session, with confirmation based on observed liveness. Shell commands should wait for work they spawn. Processes that leave the group require their own lifecycle management.

A persistent session serializes jobs with the same name and workspace. Successful Bash exits save cwd and exported environment changes, including unsets. Replacing the shell or its exit trap can skip session capture. Reset returns an error while the session is active.

```bash
remote-run exec --session dev --shell 'cd sub && export TAG=v1'
remote-run exec --session dev --no-sync --shell 'pwd && echo "$TAG"'
remote-run session list --repo .
remote-run session reset dev
```

## Pull and prune

```bash
remote-run pull RUN_ID ./results/RUN_ID
remote-run pull RUN_ID ./results/RUN_ID --path reports/bench.md --paths-only --force
remote-run prune --days 14
remote-run prune --days 14 --apply
```

`pull` defaults to a new destination and copies state, log and eligible artifacts. A 10 MiB per-file limit and checkpoint suffix exclusions apply to automatic artifact selection. `--max-file-mb 0` disables the size check. `--all-artifacts` removes both artifact filters. Explicit `--path` uses workspace and shared-root validation, excludes internal control paths and is independent of the artifact filters. Use `--force` to merge into an existing destination. Configured environment values are redacted in host, config and session inspection. The private worker request is outside the default pull set. Logs and user-created artifacts can contain values printed or written by the job.

`prune` previews by default and requires `--apply` to delete. It verifies directory and run ownership, skips busy entries and operates on positively identified terminal runs. Legacy jobs without a run ownership marker remain accessible and are skipped by prune.

## CLI and MCP reference

CLI common flags are `--config PATH`, `--host NAME`, and `--pretty`. Host selection starts with the explicit host, then the applicable project host, then the global default. `hosts` lists all profiles and `init` selects its target positionally. MCP uses `config` and, where applicable, `host`. MCP tools return one text content item containing the CLI JSON on success, and include `isError` when the CLI fails.

| CLI command and options | MCP tool and differences |
|---|---|
| `hosts` | `remote_hosts` with optional `config`. |
| `init TARGET` with `--name`, `--port`, `--root`, `--default`, `--force` | CLI only. |
| `config` with `--repo`, `--remote-dir`, `--workspace` | `remote_config` requires `repo`. CLI defaults to cwd. Inspection leaves Git dirty state uncomputed. |
| `doctor` with `--repo`, `--env`, repeatable `--shared`, `--check-cmd`, `--check-import`, `--check-cuda`, `--init-root` | `remote_doctor` exposes `check_cmds` and `check_imports` as arrays and supports `check_cuda`. Root claiming is CLI only. |
| `sync` with `--repo`, `--remote-dir`, `--workspace`, `--dry-run`, `--delete`, `--adopt` | `remote_sync` requires `repo` and omits explicit deletion and adoption. Default sync can still change or remove source files. |
| `exec` with `--repo`, `--remote-dir`, `--workspace`, `--no-sync`, `--no-project`, `--session`, `--device`, `--no-gpu`, `--env`, repeatable `--shared`, `--env-var`, `--run-id`, `--wait`, `--tail-bytes`, `--raw-log`, `--follow`, `--print-log`, and either `--shell` or `-- COMMAND…` | `remote_exec` requires exactly one of `command` (argv array) and `shell`. `shared` is an array, `env_vars` a string map. `repo` is required for upload and still recommended with `no_sync=true` when applying `.remote-run.toml`. `no_project` requires `no_sync`. `--follow` and `--print-log` are CLI only. |
| `status [RUN_ID]` with `--repo`, `--project`, `--limit`, `--wait`, `--offset`, `--tail-bytes`, `--raw-log`, `--follow`, `--print-log` | `remote_status` exposes `run_id` and all snapshot and listing fields. `--follow` and `--print-log` are CLI only. Listing defaults to 20. Byte offset and wait apply to a specific run. |
| `cancel RUN_ID` | `remote_cancel` requires `run_id`. |
| `pull RUN_ID DEST` with `--all-artifacts`, `--max-file-mb`, repeatable `--path`, `--paths-only`, `--force` | `remote_pull` requires `run_id` and `destination`, with an array `paths`. `paths_only` requires at least one path. |
| `session list`, `session reset NAME` with `--repo`, `--remote-dir`, `--workspace` | `remote_session` requires `action` and requires `name` for reset. A reset without path options targets the current project workspace. |
| `prune` with `--days`, `--apply` | `remote_prune` defaults to 14 days and preview. Applying requires a JSON boolean `true`. |

Run log snapshots default to 8192 bytes. `--tail-bytes 0` reads to the end. `--follow` streams the full log instead of applying snapshot wait and offset options. CLI `--print-log` and `--follow` send metadata to stderr. Otherwise the receipt is JSON on stdout. `exec` and `status --follow` exit 2 for failed, cancelled or interrupted runs. Ordinary `status` exits 0 when the inspection succeeds. Argument syntax errors and doctor missing-command/import checks may also exit 2. Local and SSH runtime errors exit 1. Interrupting a CLI run or follow exits 130. Cancelling an MCP request stops its local CLI process. Use `remote_cancel` to cancel an already submitted job.

See [the schema and Agent setup guide](../skills/remote-run/references/schema-and-setup.md) for both TOML files.
