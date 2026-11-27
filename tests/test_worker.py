"""Unit tests for standalone zero-dependency remote_worker.py (v2)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import remote_worker as rw


def make_exec(path: Path, body: str = "#!/bin/sh\necho ok\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


class RemoteWorkerTests(unittest.TestCase):
    def test_standalone_zero_local_imports(self):
        """remote_worker.py must be executable in isolation with only stdlib."""
        with tempfile.TemporaryDirectory() as tmp:
            isolated = Path(tmp) / "worker.py"
            isolated.write_text((ROOT / "remote_worker.py").read_text(encoding="utf-8"), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-I", "-c",
                 "import importlib.util, sys; spec=importlib.util.spec_from_file_location('w', sys.argv[1]); "
                 "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.PROTOCOL_VERSION)",
                 str(isolated)],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(proc.stdout.strip(), "2")

    def test_pattern_matching_semantics(self):
        self.assertEqual(rw.validate_pattern("docs/generated/"), "docs/generated/")
        self.assertEqual(rw.validate_pattern("**/*.trace"), "**/*.trace")
        with self.assertRaises(ValueError):
            rw.validate_pattern("/abs")
        with self.assertRaises(ValueError):
            rw.validate_pattern("!negate")
        with self.assertRaises(ValueError):
            rw.validate_pattern("a/../b")
        with self.assertRaises(ValueError):
            rw.validate_pattern("foo**bar")

        self.assertTrue(rw.match_pattern("docs/generated/a.html", "docs/generated/"))
        self.assertTrue(rw.match_pattern("docs/generated", "docs/generated/"))
        self.assertFalse(rw.match_pattern("docs/generated2/a.html", "docs/generated/"))
        self.assertTrue(rw.match_pattern("a.trace", "**/*.trace"))
        self.assertTrue(rw.match_pattern("sub/dir/b.trace", "**/*.trace"))
        self.assertFalse(rw.match_pattern("sub/dir/b.txt", "**/*.trace"))

    def test_utf8_log_slicing_boundaries_and_tiny_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run.log"
            # Each CJK char is 3 bytes in UTF-8; emoji is 4 bytes.
            text = "ab中文字🚀xy"
            raw = text.encode("utf-8")
            log.write_bytes(raw)

            # Page size 4 splits '中' (bytes 2..4); should return 'ab' and next_offset=2.
            p1 = rw.read_log_slice(log, offset=0, tail_bytes=4)
            self.assertEqual(p1["text"], "ab")
            self.assertEqual(p1["next_offset"], 2)
            self.assertTrue(p1["truncated"])

            # Tiny page size 2 starting at 3-byte char '中' must advance by completing codepoint.
            p2 = rw.read_log_slice(log, offset=2, tail_bytes=2)
            self.assertEqual(p2["text"], "中")
            self.assertEqual(p2["next_offset"], 5)

            # Collect remaining pages sequentially and verify exact round-trip without U+FFFD.
            chunks = [p1["text"], p2["text"]]
            off = p2["next_offset"]
            while off < len(raw):
                page = rw.read_log_slice(log, offset=off, tail_bytes=3)
                self.assertGreater(page["next_offset"], off)
                chunks.append(page["text"])
                off = page["next_offset"]
            self.assertEqual("".join(chunks), text)
            self.assertNotIn("\ufffd", "".join(chunks))

            # Active run ending mid-sequence holds back trailing incomplete bytes.
            partial_log = Path(tmp) / "partial.log"
            partial_log.write_bytes(b"ok" + "中".encode("utf-8")[:2])
            pa = rw.read_log_slice(partial_log, offset=0, tail_bytes=16, active=True)
            self.assertEqual(pa["text"], "ok")
            self.assertEqual(pa["next_offset"], 2)
            # When active run has ONLY an incomplete byte sequence at EOF, wait without advancing.
            pa2 = rw.read_log_slice(partial_log, offset=2, tail_bytes=16, active=True)
            self.assertEqual(pa2["text"], "")
            self.assertEqual(pa2["next_offset"], 2)

            # Tail mode skips leading continuation bytes instead of emitting replacement char.
            ptail = rw.read_log_slice(log, offset=None, tail_bytes=5)
            self.assertNotIn("\ufffd", ptail["text"])
            self.assertEqual(ptail["text"], "xy")

    def test_path_layers_and_probe_executable_bare_and_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workdir = base / "ws"
            workdir.mkdir()
            d_inv = base / "inv_bin"
            d_proj = workdir / "build" / "bin"
            d_host = base / "host_bin"
            d_ssh = base / "ssh_bin"
            for d in (d_inv, d_host, d_ssh):
                d.mkdir()
            make_exec(d_inv / "tool", "#!/bin/sh\necho inv\n")
            make_exec(d_host / "tool", "#!/bin/sh\necho host\n")
            make_exec(d_ssh / "tool", "#!/bin/sh\necho ssh\n")
            make_exec(d_host / "only_host")
            non_exec = base / "not_exec"
            non_exec.write_text("hi")
            non_exec.chmod(0o644)

            layers = [
                {"source": "invocation", "entry": str(d_inv), "allow_pending": False},
                {"source": "project", "entry": "build/bin", "allow_pending": True},
                {"source": "host", "entry": str(d_host), "allow_pending": False},
            ]
            eff, rep = rw.resolve_path_layers(layers, str(d_ssh), workdir=workdir, strict=True)
            parts = eff.split(os.pathsep)
            self.assertEqual(parts, [str(d_inv), str(d_proj), str(d_host), str(d_ssh)])
            self.assertEqual(rep[1]["status"], "pending")

            # Bare name resolves to highest priority (invocation).
            p_bare = rw.probe_executable("tool", eff, cwd=workdir)
            self.assertTrue(p_bare["ok"])
            self.assertEqual(p_bare["resolved"], str(d_inv / "tool"))

            # Literal absolute executable succeeds; non-executable fails cleanly.
            p_abs = rw.probe_executable(str(d_host / "only_host"), eff, cwd=workdir)
            self.assertTrue(p_abs["ok"])
            self.assertEqual(p_abs["kind"], "literal_path")

            p_noexec = rw.probe_executable(str(non_exec), eff, cwd=workdir)
            self.assertFalse(p_noexec["ok"])
            self.assertEqual(p_noexec["reason"], "not_executable")

            # Relative path works when cwd is provided.
            make_exec(workdir / "scripts" / "run.sh")
            p_rel = rw.probe_executable("scripts/run.sh", eff, cwd=workdir)
            self.assertTrue(p_rel["ok"])
            self.assertEqual(p_rel["resolved"], str(workdir / "scripts" / "run.sh"))

            # Strict mode rejects missing explicit host/invocation dirs.
            with self.assertRaises(ValueError):
                rw.resolve_path_layers(
                    [{"source": "host", "entry": str(base / "does-not-exist"), "allow_pending": False}],
                    str(d_ssh), workdir=workdir, strict=True,
                )

    def test_doctor_is_read_only_and_checks_commands_and_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "rr-v2"
            bin_dir = base / "bin"
            for tool in ("bash", "tmux", "rsync"):
                make_exec(bin_dir / tool)
            make_exec(
                bin_dir / "nvidia-smi",
                "#!/bin/sh\nprintf '0, GPU-aaaa, RTX 4090, 24564 MiB\\n1, GPU-bbbb, RTX 4090, 24564 MiB\\n'\n",
            )
            make_exec(bin_dir / "cargo")
            payload = {
                "protocol_version": 2,
                "action": "doctor",
                "host": {"name": "box", "root": str(root), "path": [str(bin_dir)], "gpu": {"devices": [0, 1]}},
                "path_layers": [{"source": "host", "entry": str(bin_dir), "allow_pending": False}],
                "check_cmds": ["cargo", str(bin_dir / "cargo"), "missing_tool"],
            }
            res = rw.control(payload)
            # Must NOT create root directory (strictly read-only).
            self.assertFalse(root.exists())
            self.assertEqual(res["root_state"], "pending")
            self.assertEqual(res["status"], "checks_failed")
            self.assertTrue(res["_exec_failed"])
            self.assertEqual(res["failed_commands"], ["missing_tool"])
            self.assertEqual(len(res["gpu"]["observed"]), 2)
            self.assertEqual(res["gpu"]["observed"][0]["uuid"], "GPU-aaaa")

    def test_gpu_allocation_sorted_uuid_locks_and_requested_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            make_exec(
                bin_dir / "nvidia-smi",
                "#!/bin/sh\nprintf '0, GPU-zzzz, Card0, 16000 MiB\\n1, GPU-aaaa, Card1, 16000 MiB\\n'\n",
            )
            indices, uuids, lock_keys = rw.resolve_gpu_allocation([1, 0], [0, 1], str(bin_dir))
            self.assertEqual(indices, [1, 0])
            # CUDA_VISIBLE_DEVICES preserves requested order (1 then 0 -> GPU-aaaa,GPU-zzzz).
            self.assertEqual(uuids, ["GPU-aaaa", "GPU-zzzz"])
            # Lock acquisition order is sorted by UUID regardless of request order.
            _, uuids_rev, lock_keys_rev = rw.resolve_gpu_allocation([0, 1], [0, 1], str(bin_dir))
            self.assertEqual(uuids_rev, ["GPU-zzzz", "GPU-aaaa"])
            self.assertEqual(lock_keys, lock_keys_rev)
            self.assertEqual(lock_keys, ["gpu-GPU-aaaa", "gpu-GPU-zzzz"])

    def test_pull_type_neutral_size_exclude_and_explicit_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            job = base / "run1"
            art = job / "artifacts"
            art.mkdir(parents=True)
            (job / "state.json").write_text("{}")
            (job / "run.log").write_text("hi\n")
            # .pt file under size limit must be INCLUDED (no CHECKPOINT_SUFFIXES hardcoding!).
            (art / "weights.pt").write_bytes(b"small")
            (art / "debug.trace").write_bytes(b"trace")
            (art / "huge.bin").write_bytes(b"x" * (2 * 1024 * 1024))
            wdir = base / "ws"
            wdir.mkdir()

            jfiles, wfiles, skipped = rw.eligible_pull_files(
                job, str(wdir), max_file_mib=1, exclude_patterns=["**/*.trace"],
            )
            self.assertIn("artifacts/weights.pt", jfiles)
            self.assertNotIn("artifacts/debug.trace", jfiles)
            self.assertNotIn("artifacts/huge.bin", jfiles)
            reasons = {s["path"]: s["reason"] for s in skipped}
            self.assertEqual(reasons["artifacts/debug.trace"], "excluded:**/*.trace")
            self.assertEqual(reasons["artifacts/huge.bin"], "exceeds_1mib")

            # Explicit --artifact bypasses size and exclude filters for that file only.
            jfiles2, _, skipped2 = rw.eligible_pull_files(
                job, str(wdir), max_file_mib=1, exclude_patterns=["**/*.trace"],
                explicit_artifacts=["huge.bin"],
            )
            self.assertEqual(jfiles2, ["state.json", "run.log", "artifacts/huge.bin"])
            self.assertEqual(skipped2, [])

    def test_prune_unmanaged_summary_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rr-v2"
            rw.ensure_root_owned(root)
            runs = root / "runs"
            rw.private_directory(runs)
            for i in range(25):
                (runs / f"legacy-{i:02d}").mkdir()
            res = rw.prune_runs(root, "box", 14.0, False, os.environ.get("PATH", ""), verbose=False)
            self.assertEqual(res["skipped_unmanaged_count"], 25)
            self.assertEqual(res["skipped_by_reason"], {"unrecognized_metadata": 25})
            self.assertEqual(len(res["skipped_unmanaged"]), 20)
            self.assertTrue(res["skipped_unmanaged_truncated"])

            res_v = rw.prune_runs(root, "box", 14.0, False, os.environ.get("PATH", ""), verbose=True)
            self.assertEqual(len(res_v["skipped_unmanaged"]), 25)
            self.assertFalse(res_v["skipped_unmanaged_truncated"])

    def test_worker_execution_env_precedence_no_python_magic_and_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "rr-v2"
            rw.ensure_root_owned(root)
            wdir = root / "projects" / "demo" / "workspaces" / "default"
            (wdir / "sub").mkdir(parents=True)
            job = root / "runs" / "run-1"
            (job / "artifacts").mkdir(parents=True)
            out_file = job / "artifacts" / "out.json"

            # Verify: no implicit PYTHONUNBUFFERED=1, literal env precedence, GPU-managed CPU run hides CUDA.
            req = {
                "host": "box",
                "workdir": str(wdir),
                "shared_dir": "",
                "rel_subdir": "",
                "project_id": "demo",
                "workspace_id": "default",
                "gpu_managed": True,
                "gpus": [],
                "gpu_uuids": [],
                "gpu_lock_keys": [],
                "locks_dir": str(root / "locks"),
                "effective_path": os.environ.get("PATH", "/usr/bin:/bin"),
                "trusted_path": os.environ.get("PATH", "/usr/bin:/bin"),
                "ssh_base_env": {"SSH_VAR": "base", "OVERRIDE": "ssh"},
                "config_env": {"OVERRIDE": "cfg", "CFG_VAR": "1"},
                "invocation_env": {"OVERRIDE": "inv"},
                "session": "dev",
                "session_key": rw.session_key_for(str(wdir), "dev"),
                "argv": [],
                "shell": (
                    f'cd sub && export PERSISTED=hello && '
                    f'{sys.executable} -c "import json,os; '
                    f'open({str(out_file)!r}, \'w\').write(json.dumps(dict(os.environ)))"'
                ),
            }
            rw.write_json(job / "request.json", req)
            rw.write_json(job / "state.json", {"run_id": "run-1", "status": "queued", "waiting_for_lock": False})
            rw.worker(root, job)

            st = rw.read_json(job / "state.json")
            self.assertEqual(st["status"], "completed")
            self.assertEqual(st["exit_code"], 0)
            env_seen = json.loads(out_file.read_text())
            self.assertNotIn("PYTHONUNBUFFERED", env_seen)
            self.assertEqual(env_seen["OVERRIDE"], "inv")
            self.assertEqual(env_seen["CFG_VAR"], "1")
            self.assertEqual(env_seen["SSH_VAR"], "base")
            self.assertEqual(env_seen["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(env_seen["REMOTE_RUN_ID"], "run-1")
            self.assertEqual(env_seen["REMOTE_RUN_WORKSPACE"], str(wdir))

            # Second job in same session inherits cwd ('sub') and PERSISTED, while invocation_env overrides session.
            job2 = root / "runs" / "run-2"
            (job2 / "artifacts").mkdir(parents=True)
            out2 = job2 / "artifacts" / "out2.json"
            req2 = dict(req, invocation_env={"PERSISTED": "from_inv"}, shell="", argv=[
                sys.executable, "-c",
                f"import json,os; open({str(out2)!r}, 'w').write(json.dumps({{'cwd': os.getcwd(), 'p': os.environ.get('PERSISTED')}}))"
            ])
            rw.write_json(job2 / "request.json", req2)
            rw.write_json(job2 / "state.json", {"run_id": "run-2", "status": "queued", "waiting_for_lock": False})
            rw.worker(root, job2)
            st2 = rw.read_json(job2 / "state.json")
            self.assertEqual(st2["status"], "completed")
            d2 = json.loads(out2.read_text())
            self.assertEqual(d2["cwd"], str((wdir / "sub").resolve()))
            self.assertEqual(d2["p"], "from_inv")

    def test_shared_links_registration_and_safe_unlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rr-v2"
            rw.ensure_root_owned(root)
            ws_info = {
                "project_id": "demo",
                "workspace_id": "default",
                "managed": True,
                "workdir": str(root / "projects" / "demo" / "workspaces" / "default"),
                "shared_dir": str(root / "projects" / "demo" / "shared"),
                "root_commit": "abc12345",
            }
            p1 = rw.prepare_remote_workspace(root, ws_info, ["cache"], trusted_path=os.environ.get("PATH", ""))
            wdir = Path(p1["workdir"])
            sdir = Path(p1["shared_dir"])
            link = wdir / "cache"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (sdir / "cache").resolve())
            (sdir / "cache" / "keep.txt").write_text("data")

            # Removing 'cache' from shared_dirs unlinks the registered symlink but retains backing data.
            rw.prepare_remote_workspace(root, ws_info, [], trusted_path=os.environ.get("PATH", ""))
            self.assertFalse(link.exists() or link.is_symlink())
            self.assertTrue((sdir / "cache" / "keep.txt").is_file())

    def test_regressions_utf8_binary_missing_log_and_invalid_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.log"
            r_bin = rw.read_log_slice(missing, binary=True)
            self.assertEqual(r_bin["data"], "")
            self.assertNotIn("text", r_bin)
            r_raw = rw.read_log_slice(missing, raw_log=True)
            self.assertEqual(r_raw["text"], "")
            self.assertEqual(r_raw["data"], "")

            # Invalid byte following a 3-byte lead must NOT eat the next valid ASCII char.
            corrupt = Path(tmp) / "corrupt.log"
            corrupt.write_bytes(b"\xe4Xok")
            p1 = rw.read_log_slice(corrupt, offset=0, tail_bytes=1)
            self.assertEqual(p1["next_offset"], 1)
            p2 = rw.read_log_slice(corrupt, offset=1, tail_bytes=10)
            self.assertEqual(p2["text"], "Xok")

            # Tail slice of 3 pure continuation bytes strips all 3 cleanly instead of leaving replacement.
            cont = Path(tmp) / "cont.log"
            cont.write_bytes(b"aaaa\x80\x80\x80")
            pt = rw.read_log_slice(cont, offset=None, tail_bytes=3)
            self.assertEqual(pt["text"], "")
            self.assertEqual(pt["next_offset"], 7)

    def test_regressions_idempotent_retry_and_detached_submit_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "rr-v2"
            rw.ensure_root_owned(root)
            wdir = root / "projects" / "demo" / "workspaces" / "default"
            rw.prepare_remote_workspace(
                root,
                {"project_id": "demo", "workspace_id": "default", "managed": True, "workdir": str(wdir), "shared_dir": ""},
                [], trusted_path=os.environ.get("PATH", ""),
            )
            rw.reconcile_synced_manifest(root, str(wdir), {}, [], [])
            bin_dir = base / "bin"
            # Fake tmux that starts the command in background in a new session & tracks pid.
            pid_file = base / "fake_tmux.pid"
            make_exec(
                bin_dir / "tmux",
                f"#!/bin/sh\n"
                f"if [ \"$1\" = \"new-session\" ]; then\n"
                f"  shift 6\n"
                f"  if [ \"$1\" = \"--\" ]; then shift; fi\n"
                f"  \"$@\" </dev/null >/dev/null 2>&1 &\n"
                f"  echo $! > {pid_file}\n"
                f"  exit 0\n"
                f"elif [ \"$1\" = \"has-session\" ]; then\n"
                f"  [ -f {pid_file} ] && kill -0 $(cat {pid_file}) 2>/dev/null\n"
                f"  exit $?\n"
                f"elif [ \"$1\" = \"kill-session\" ]; then\n"
                f"  exit 0\n"
                f"fi\n"
                f"exit 0\n",
            )
            make_exec(bin_dir / "bash", "#!/bin/sh\nexec /bin/bash \"$@\"\n")
            make_exec(bin_dir / "nvidia-smi", "#!/bin/sh\nprintf '0, GPU-1111, Card0, 8000 MiB\\n'\n")
            host_cfg = {
                "name": "box",
                "root": str(root),
                "path": [str(bin_dir)],
                "gpu": {"devices": [0]},
            }
            worker_src = (ROOT / "remote_worker.py").read_text(encoding="utf-8")
            req = {
                "host": "box",
                "workdir": str(wdir),
                "shared_dir": "",
                "rel_subdir": "",
                "project_id": "demo",
                "workspace_id": "default",
                "gpus": [0],
                "path_layers": [{"source": "host", "entry": str(bin_dir), "allow_pending": False}],
                "config_env": {},
                "invocation_env": {},
                "session": "",
                "session_key": "",
                "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
                "shell": "",
            }
            job = root / "runs" / "run-live-1"
            trusted = rw.build_trusted_host_path(os.environ.get("PATH", ""), [str(bin_dir)])
            r1 = rw.remote_submit(root, job, host_cfg, req, worker_src, trusted, {"PATH": os.environ.get("PATH", "")})
            self.assertFalse(r1["reused"])

            # Idempotent retry MUST succeed even if nvidia-smi is broken now AND reuse_only=True is passed!
            make_exec(bin_dir / "nvidia-smi", "#!/bin/sh\nexit 1\n")
            retry_req = dict(req, reuse_only=True)
            r2 = rw.remote_submit(root, job, host_cfg, retry_req, worker_src, trusted, {"PATH": os.environ.get("PATH", "")})
            self.assertTrue(r2["reused"])

            # Wait briefly for worker to start running, then cancel cleanly.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                st = rw.state_of(job, root, trusted)
                if st["status"] == "running":
                    break
                time.sleep(0.05)
            cancelled = rw.remote_cancel(root, job, trusted)
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertTrue(cancelled["cancellation_requested"])

    def test_regressions_dry_run_digest_and_pinned_pull_symlink_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rr-v2"
            rw.ensure_root_owned(root)
            ws_info = {
                "project_id": "demo", "workspace_id": "default", "managed": True,
                "workdir": str(root / "projects" / "demo" / "workspaces" / "default"),
                "shared_dir": str(root / "projects" / "demo" / "shared"),
                "root_commit": "abc12345",
            }
            prep = rw.prepare_remote_workspace(root, ws_info, [], trusted_path=os.environ.get("PATH", ""))
            wdir = Path(prep["workdir"])
            f = wdir / "a.txt"
            f.write_text("v1")
            st = f.stat()
            # Record manifest with mismatched sha256 but matching stat version -> dry-run MUST NOT claim it would prune!
            rw.write_json(wdir / rw.SYNCED_MANIFEST, {
                "schema_version": 2,
                "updated_at": rw.now(),
                "files": {"a.txt": {"version": rw.file_version(st), "sha256": "0" * 64}},
            })
            dry = rw.reconcile_synced_manifest(root, str(wdir), {}, ["a.txt"], [], dry_run=True)
            self.assertEqual(dry["pruned_ghost_files"], [])

            # Arbitrary symlink inside workspace must be rejected by pull.
            outside = Path(tmp) / "secret.txt"
            outside.write_text("secret")
            (wdir / "evil_link").symlink_to(outside)
            job = root / "runs" / "r-pull"
            job.mkdir(parents=True)
            (job / "state.json").write_text('{"status": "completed"}')
            (job / "run.log").write_text("")
            with self.assertRaisesRegex(ValueError, "arbitrary symlink"):
                rw.eligible_pull_files(job, str(wdir), extra_paths=["evil_link"], remote_root=root)

    def test_regressions_partial_gpu_holder_and_corrupt_prune_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rr-v2"
            rw.ensure_root_owned(root)
            runs = root / "runs"
            rw.private_directory(runs)
            # Partial GPU holder (still queued waiting on 2nd lock, already holding 1st lock)
            holder = runs / "run-holder"
            holder.mkdir()
            rw.write_json(holder / "request.json", {
                "host": "box", "locks_dir": str(root / "locks"), "gpu_lock_keys": ["gpu-A", "gpu-B"],
            })
            rw.write_json(holder / "state.json", {
                "run_id": "run-holder", "status": "queued", "waiting_for_lock": True,
                "held_gpu_locks": ["gpu-A"], "worker_pid": os.getpid(),
                "worker_identity": rw.process_identity(os.getpid()),
            })
            waiter = runs / "run-waiter"
            waiter.mkdir()
            holders = rw.queue_holder_runs(root, waiter, {
                "locks_dir": str(root / "locks"), "gpu_lock_keys": ["gpu-A"],
            })
            self.assertEqual(holders, ["run-holder"])

            # Corrupt run with non-string status is skipped cleanly in prune_runs without crashing.
            bad = runs / "run-bad"
            bad.mkdir()
            rw.write_json(bad / rw.RUN_MARKER, {"schema_version": 2, "run_id": "run-bad"})
            rw.write_json(bad / "request.json", {"host": "box"})
            rw.write_json(bad / "state.json", {"run_id": "run-bad", "status": 123})
            old = time.time() - 86400 * 30
            os.utime(bad, (old, old))
            res = rw.prune_runs(root, "box", 14.0, False, os.environ.get("PATH", ""))
            self.assertEqual(res["runs_count"], 0)
            bad_entry = next(e for e in res["skipped_unmanaged"] if e["path"] == str(bad))
            self.assertEqual(bad_entry["reason"], "unrecognized_metadata")

            # Unhashable status (list) must not crash prune_runs.
            bad2 = runs / "run-bad-list"
            bad2.mkdir()
            rw.write_json(bad2 / rw.RUN_MARKER, {"schema_version": 2, "run_id": "run-bad-list"})
            rw.write_json(bad2 / "request.json", {"host": "box"})
            rw.write_json(bad2 / "state.json", {"run_id": "run-bad-list", "status": ["bad"]})
            os.utime(bad2, (old, old))
            res2 = rw.prune_runs(root, "box", 14.0, False, os.environ.get("PATH", ""))
            bad2_entry = next(e for e in res2["skipped_unmanaged"] if e["path"] == str(bad2))
            self.assertEqual(bad2_entry["reason"], "unrecognized_metadata")

            # Receiver parent symlink check rejects symlinks before rsync can write through them.
            ws_info = {
                "project_id": "demo", "workspace_id": "ws-lnk", "managed": True,
                "workdir": str(root / "projects" / "demo" / "workspaces" / "ws-lnk"),
                "shared_dir": str(root / "projects" / "demo" / "shared"),
                "root_commit": "abc12345",
            }
            prep = rw.prepare_remote_workspace(root, ws_info, [], trusted_path=os.environ.get("PATH", ""))
            wdir = Path(prep["workdir"])
            outside_dir = Path(tmp) / "outside_dir"
            outside_dir.mkdir()
            (wdir / "sub").symlink_to(outside_dir)
            with self.assertRaisesRegex(ValueError, "receiver parent path.*symlink"):
                rw.prepare_remote_workspace(
                    root, ws_info, [], trusted_path=os.environ.get("PATH", ""), manifest_parents=["sub"],
                )

    def test_pull_stage_sweep_ignores_unmanaged_and_submit_blocks_unsynced_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rr-v2"
            rw.ensure_root_owned(root)
            # 1. Stale pull stage sweep must NEVER delete unmanaged stage-* dirs without PULL_STAGE_MARKER.
            stages = root / "pull-stages"
            rw.private_directory(stages)
            unmanaged = stages / "stage-foreign"
            unmanaged.mkdir()
            (unmanaged / "keep.txt").write_text("do-not-delete")
            old = time.time() - 7200
            os.utime(unmanaged, (old, old))
            owned = stages / "stage-owned"
            owned.mkdir()
            rw.write_json(owned / rw.PULL_STAGE_MARKER, {"schema_version": 2, "stage_id": "stage-owned"})
            os.utime(owned, (old, old))
            rw._cleanup_stale_pull_stages(root)
            self.assertTrue((unmanaged / "keep.txt").is_file())
            self.assertFalse(owned.exists())

            # 2. remote_submit MUST refuse a managed workspace with sync_completed=False (e.g. interrupted sync).
            ws_info = {
                "project_id": "demo", "workspace_id": "ws-unsynced", "managed": True,
                "workdir": str(root / "projects" / "demo" / "workspaces" / "ws-unsynced"),
                "shared_dir": str(root / "projects" / "demo" / "shared"),
                "root_commit": "abc12345",
            }
            rw.prepare_remote_workspace(root, ws_info, [], trusted_path=os.environ.get("PATH", ""), for_sync=True)
            with self.assertRaisesRegex(ValueError, "sync_completed=false"):
                rw.remote_submit(
                    root,
                    root / "runs" / "run-unsynced",
                    {"name": "box", "root": str(root)},
                    {
                        "host": "box",
                        "workdir": ws_info["workdir"],
                        "shared_dir": ws_info["shared_dir"],
                        "project_id": "demo",
                        "workspace_id": "ws-unsynced",
                        "root_commit": "abc12345",
                        "argv": ["true"],
                        "shell": "",
                    },
                    worker_src="",
                    trusted_path=os.environ.get("PATH", ""),
                    ssh_base_env={},
                )

            # 3. reuse_only=True MUST refuse launching a new run if an existing run was pruned concurrently.
            with self.assertRaisesRegex(ValueError, "removed while reusing"):
                rw.remote_submit(
                    root,
                    root / "runs" / "run-pruned-mid-retry",
                    {"name": "box", "root": str(root)},
                    {
                        "host": "box",
                        "workdir": ws_info["workdir"],
                        "shared_dir": ws_info["shared_dir"],
                        "project_id": "demo",
                        "workspace_id": "ws-unsynced",
                        "root_commit": "abc12345",
                        "reuse_only": True,
                        "argv": ["true"],
                        "shell": "",
                    },
                    worker_src="",
                    trusted_path=os.environ.get("PATH", ""),
                    ssh_base_env={},
                )

    def test_worker_kills_sigterm_resistant_leader_on_metadata_failure_before_releasing_gpu_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "rr-v2"
            rw.ensure_root_owned(root)
            wdir = root / "projects" / "demo" / "workspaces" / "default"
            wdir.mkdir(parents=True)
            job = root / "runs" / "run-sigterm-ignore"
            (job / "artifacts").mkdir(parents=True)
            pid_file = base / "child.pid"
            req = {
                "host": "box",
                "workdir": str(wdir),
                "shared_dir": "",
                "rel_subdir": "",
                "project_id": "demo",
                "workspace_id": "default",
                "gpu_managed": True,
                "gpus": [0],
                "gpu_uuids": ["GPU-aaaa"],
                "gpu_lock_keys": ["gpu-GPU-aaaa"],
                "locks_dir": str(root / "locks"),
                "effective_path": os.environ.get("PATH", "/usr/bin:/bin"),
                "trusted_path": os.environ.get("PATH", "/usr/bin:/bin"),
                "ssh_base_env": {},
                "config_env": {},
                "invocation_env": {},
                "session": "",
                "session_key": "",
                "argv": [
                    sys.executable, "-c",
                    f"import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)",
                ],
                "shell": "",
            }
            rw.write_json(job / "request.json", req)
            rw.write_json(job / "state.json", {"run_id": "run-sigterm-ignore", "status": "queued", "waiting_for_lock": False})
            orig_write_json = rw.write_json

            def fail_on_running_write(path, data):
                if path == job / "state.json" and isinstance(data, dict) and data.get("status") == "running":
                    # Wait briefly until the SIGTERM-ignoring child writes its pid file, then fail state write.
                    deadline = time.monotonic() + 3.0
                    while not pid_file.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    raise OSError("simulated disk full on state.json write")
                return orig_write_json(path, data)

            with mock.patch.object(rw, "write_json", side_effect=fail_on_running_write):
                with self.assertRaisesRegex(OSError, "simulated disk full"):
                    rw.worker(root, job)

            # Leader MUST already be dead before worker() exited and released GPU locks!
            self.assertTrue(pid_file.exists())
            child_pid = int(pid_file.read_text().strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

            # _read_proc_stat_fields handles non-UTF-8 truncated kernel comm bytes cleanly.
            fake_stat = base / "fake_stat"
            fake_stat.write_bytes(b"1234 (model-loaderX-\xce) S 1 1234 1234 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 999\n")
            fields = rw._read_proc_stat_fields(fake_stat)
            self.assertEqual(fields[0], "S")
            self.assertEqual(fields[2], "1234")
            self.assertEqual(fields[19], "999")


if __name__ == "__main__":
    unittest.main()
