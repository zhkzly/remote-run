---
name: remote-run
description: Offload arbitrary commands, builds, tests, scripts, and GPU training from a local laptop workspace to any configured remote Linux host, sync code safely without ghost files, manage opt-in persistent sessions, and pull results back. Use when the user asks to run, test, build, debug, monitor, or train on a remote Linux host or GPU desktop.
---

# remote-run (v2)

Use `remote-run` to keep editing and AI collaboration on the laptop while executing commands cleanly on a remote Linux host (CPU or GPU, any language, zero remote installation).

## Core Workflow

1. **Inspect merged config first** (`remote_config` / `remote-run config --repo <abs_path>`):
   - Always pass explicit absolute `repo` when calling MCP tools.
   - Check `environment.note`, `environment.path_layers`, `environment.env_keys`, `workspace`, and `resources.gpus`.
   - **Important**: If `environment.note` or `path_layers` shows a prebuilt toolchain/environment on the remote host, invoke its executables directly (e.g. `["python", "-u", ...]`, `["cargo", "test"]`, `["pytest", "-q"]`) instead of running package-manager wrappers that download duplicate multi-GB environments into the remote workspace.
2. **Verify readiness if needed** (`remote_doctor` / `remote-run doctor --repo <abs_path> --check-cmd ...`):
   - Strictly read-only. Supports both bare executable names (`cargo`, `node`, `python`) and literal paths (`/opt/envs/proj/bin/python`, `scripts/run.sh`).
   - For framework-specific probes (e.g. `torch.cuda.is_available()` or `nvcc --version`), run an explicit `remote_exec` with `no_sync: true` (pass `repo` only if already synced, or omit `repo` for host-only probes).
3. **Execute remotely** (`remote_exec` / `remote-run exec`):
   - Provide **exactly one** of `argv` (array of strings, e.g. `["cargo", "test"]`) OR `shell` (explicit bash string, e.g. `"cmake --build build && ctest"`). Never pass both, and never pass a boolean for `shell`.
   - For ad-hoc host commands without a local project: in CLI pass `--no-project --no-sync`; in MCP `remote_exec` simply **omit `repo`** and pass `no_sync: true` (MCP has no `no_project` parameter). Omit `remote_dir` to run in `<root>/adhoc` and pass absolute paths to your command; to inspect a synced workspace subpath, pass `repo` + `no_sync: true`.
   - Prefer `argv` for simple commands; use `shell` only for pipes, redirections, `&&`, or `cd` in a persistent `session`.
   - On GPU hosts, CPU execution is the default unless `gpus` is bound or passed explicitly (`gpus: [0]` / `--gpus 0`). Use `no_gpu: true` (`--no-gpu`) to override a GPU binding for lightweight CPU tests.
   - Use `wait` (default `15`s). Long jobs detach cleanly in remote `tmux` and return `detached: true` with `run_id`, `host`, and `next_action`.
   - If `status == "queued"` and `waiting_for_lock == true`, check `blocked_by_runs` and wait rather than re-submitting duplicate jobs.
4. **Monitor & pull** (`remote_status`, `remote_cancel`, `remote_pull`):
   - Always pass the `host` from the submission receipt to `remote_status`, `remote_cancel`, and `remote_pull`.
   - For MCP `remote_status`, use backoff between polls and pass `offset` (from previous `log.next_offset`) rather than repeatedly polling `wait=0` with no offset; keep `wait` short (e.g. `30–90`s) over MCP to stay within client call timeouts. For CLI, avoid piping `remote-run exec ... | tail` (it truncates JSON/`run_id`) — use `--tail-bytes`.
   - Pull writes atomically into a new destination directory (`--artifact REL` bypasses size/exclude filters for a specific artifact; `--path REL` lands under `<destination>/workspace/<REL>`, reported in `workdir_destination`). Write small summaries/index files to `$REMOTE_RUN_ARTIFACTS` so one `remote_pull` fetches them cleanly; for large multi-hundred-MiB directories, prefer CLI `remote-run pull` (not MCP `remote_pull`) to avoid MCP client 180s tool timeouts.
   - Note: after upgrading `remote-run`, restart long-lived MCP client sessions so the updated `remote-run-mcp` schema is loaded.

## Configuration & Setup

See [references/schema-and-setup.md](references/schema-and-setup.md) for the exact v2 3-layer schema (`hosts.toml`, `.remote-run.toml`, XDG `bindings/`), Codex `SSH_AUTH_SOCK` MCP setup, and v1→v2 migration map.
