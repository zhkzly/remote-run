# `remote-run` v2 Usage & Reference

All v2 configuration files require integer `schema_version = 2` and reject unknown keys immediately.

## 1. Three-Layer Configuration & 4-Level Hierarchy

### Layer 1: Host Catalog (`$XDG_CONFIG_HOME/remote-run/hosts.toml`)
Describes **machines only** (no project bindings or language-specific keys).

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema_version` | int (`2`) | required | Strict v2 schema marker |
| `default_host` | identifier | none | Optional default host alias (omittable when only 1 host exists) |
| `[hosts.<name>].ssh` | string | required | OpenSSH alias from `~/.ssh/config` or `user@host` |
| `[hosts.<name>].port` | int `1..65535` | OpenSSH default | Optional SSH port |
| `[hosts.<name>].root` | remote path | `~/.local/share/remote-run-v2` | Level 1 remote state root |
| `[hosts.<name>].path` | list[abs/`~/`] | `[]` | Machine-wide shared `PATH` prepends |
| `[hosts.<name>.env]` | map[str, str] | `{}` | Machine-wide shared literal env vars |
| `[hosts.<name>.gpu].devices` | list[int] | omitted (CPU-only) | Declared NVIDIA GPU indices |
| `[hosts.<name>.gpu].locks_dir` | remote path | `<root>/locks` | Trusted `0700` lock directory |

### Layer 2: Project Policy (`<repo>/.remote-run.toml`, Optional, Committable)
Machine-agnostic policy shared across all developers and worktrees.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema_version` | int (`2`) | required | Strict v2 schema marker |
| `project` | identifier | derived `p-<hash>` | Level 2 remote project directory (`projects/<project>`) |
| `path` | list[rel dir] | `[]` | Workspace-relative executable dirs (e.g. `build/bin`, `node_modules/.bin`, `.venv/bin`) |
| `[env]` | map[str, str] | `{}` | Machine-agnostic literal env vars |
| `[workspace].shared` | list[rel dir] | `[]` | Cross-worktree symlinked dirs under `projects/<project>/shared/<rel>` |
| `[sync].exclude` | list[glob] | `[]` | Root-relative POSIX sender exclusions (`*`, `?`, `**`, trailing `/` for dirs) |
| `[sync].preserve` | list[rel dir] | `[]` | Remote-owned trees: never uploaded, never ghost-cleaned |
| `[sync].max_untracked_mib` | int >= 0 | `256` | Language-neutral untracked upload guard (`0` disables) |
| `[pull].max_file_mib` | int >= 0 | `10` | Automatic artifact size cap in MiB (`0` disables) |
| `[pull].exclude` | list[glob] | `[]` | Artifact-relative globs skipped unless selected via `--artifact` |

### Layer 3: Private XDG Bindings (`$XDG_CONFIG_HOME/remote-run/bindings/<scope>-<sha256>.toml`)
Managed via `remote-run bind` / `remote-run unbind` (zero modifications to the target repo):
- **`--scope repo`** (keyed by canonical `git-common-dir`): inherited automatically by **every linked Git worktree**. Allowed keys: `schema_version`, `host`, `host_endpoint` (automatically recorded endpoint fingerprint `ssh|port|root`), `gpus`, `note`, `path`, `env`.
- **`--scope checkout`** (keyed by canonical checkout root): applies to **this checkout/worktree only**, and also allows `workspace` (Level 3 directory `workspaces/<workspace>`).
- **Cross-host safety**: Host-bound fields (`gpus`, `note`, `path`, `env`) require both `host` and `host_endpoint`. If you pass `--host other-box` or point the alias to a different endpoint, host-bound fields from the previous target are NOT applied (and reported under `skipped_bindings`), while checkout `workspace` remains portable.

### Unified Precedence
- **Host**: `--host` > checkout binding > repo binding > `default_host` > sole configured host.
- **PATH** (leftmost = highest priority): `invocation --path` > `checkout binding` > `repo binding` > `project path` > `host path` > `remote noninteractive SSH PATH`.
- **Env** (later wins, literal strings only): `SSH baseline` < `host env` < `project env` < `repo binding env` < `checkout binding env` < `saved session deltas` < `invocation --env` < `tool-owned` (`PATH`, `REMOTE_RUN_ID`, `REMOTE_RUN_HOST`, `REMOTE_RUN_WORKSPACE`, `REMOTE_RUN_ARTIFACTS`, and `CUDA_VISIBLE_DEVICES` on GPU hosts).
- **GPUs**: `--no-gpu` / `--gpus` > `checkout binding` > `repo binding` > `[]` (CPU default).

---

## 2. Sync & Pull Guarantees

- **Zero language-specific `BUILTIN_IGNORE` or `BUILTIN_PRESERVE`**: Tracked files in `results/`, `outputs/`, etc. are synced normally unless explicitly excluded.
- **Hard exclusions only**: `.git`, `.ssh`, `.aws`, `.env`, `.env.local`, `.env.*.local`, `.remote-run.toml`, control markers, and any in-repo XDG config/bindings directories.
- **Single sender manifest**: Git checkouts use `git ls-files` (tracked + non-ignored untracked + initialized submodules) and send exact NUL-delimited paths to `rsync` without `-r` or `--delete`.
- **Ghost cleanup**: Previously uploaded files/symlinks are removed only when absent locally and still matching their recorded remote snapshot; newly excluded or newly gitignored files still present locally are never deleted remotely.
- **Type-neutral atomic `pull`**: No `.pt`/`.ckpt` suffix blacklist. Pull filters by recorded `[pull].max_file_mib` and `[pull].exclude`; `--artifact REL` fetches an exact artifact bypassing filters; `--path REL` fetches workspace or registered shared-dir files without `rsync -k` symlink traversal; publication is atomic.

---

## 3. CLI & MCP Command Reference

| CLI Subcommand | MCP Tool | Purpose |
| --- | --- | --- |
| `remote-run hosts` | `remote_hosts` | List configured hosts (env values redacted) |
| `remote-run init <target>` | — | Probe SSH target and write `hosts.toml` entry |
| `remote-run bind --scope repo\|checkout ...` | `remote_bind` | Create/update private XDG binding |
| `remote-run unbind --scope repo\|checkout` | `remote_unbind` | Remove private XDG binding |
| `remote-run config` | `remote_config` | Inspect merged workspace, PATH layers, env origins, note, GPUs, sync/pull policies |
| `remote-run doctor --check-cmd ...` | `remote_doctor` | Strictly read-only check of SSH, tools, PATH layers, bare/literal commands, GPUs |
| `remote-run sync [--dry-run]` | `remote_sync` | Manifest sync with untracked budget guard and ghost cleanup |
| `remote-run exec [-- CMD \| --shell STR]` | `remote_exec` | Sync (optional) and run `argv` xor `shell` (CPU default; `--gpus` opt-in) |
| `remote-run status [RUN_ID]` | `remote_status` | List recent runs or inspect status + UTF-8-aligned log slice (pass `--host` from receipt or `--repo`) |
| `remote-run cancel RUN_ID` | `remote_cancel` | Cancel running/queued job and wait for process group termination (pass `--host` from receipt or `--repo`) |
| `remote-run pull RUN_ID DEST` | `remote_pull` | Atomically pull logs, eligible artifacts, `--artifact`, or `--path` (pass `--host` from receipt or `--repo`) |
| `remote-run session list\|reset` | `remote_session` | Inspect or reset opt-in persistent shell sessions |
| `remote-run prune [--days N] [--apply] [--verbose]` | `remote_prune` | Dry-run or delete old terminal runs with `skipped_unmanaged_count` summary (pass `--host` or `--repo`) |
