"""Focused regressions for the release audit."""
import grp
from hashlib import sha256
import io
import os
from pathlib import Path
import pwd
import shlex
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import remote_run as rr
import remote_run_mcp as mcp


class AuditRegressions(unittest.TestCase):
    def test_mcp_rejects_invalid_arguments_before_launch(self):
        with mock.patch.object(mcp, "run_cli") as cli:
            result = mcp.call_tool("remote_prune", {"apply": "false"})
            self.assertTrue(result["isError"])
            cli.assert_not_called()

    def test_status_and_doctor_leave_a_fresh_root_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "remote"
            host = {"name": "box", "root": str(root), "device": "cpu"}
            with mock.patch.object(rr.shutil, "which", return_value="/bin/true"):
                self.assertEqual(rr.control({"action": "doctor", "host": host})["status"], "ready")
            self.assertFalse(root.exists())
            with self.assertRaisesRegex(ValueError, "No such run"):
                rr.control({"action": "status", "host": host, "run_id": "missing"})
            self.assertFalse(root.exists())

    def test_only_owned_regular_file_versions_can_be_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "remote"
            workdir = root / "work"
            workdir.mkdir(parents=True)
            rr.write_json(workdir / rr.WORKSPACE_MARKER, {"schema_version": 1})
            for name in ("keep.txt", "gone.txt", "old.txt"):
                (workdir / name).write_text("uploaded")
            rr.write_json(workdir / rr.SYNCED_MANIFEST, {"schema_version": 1, "files": ["old.txt"]})
            digest = sha256(b"uploaded").hexdigest()
            rr.reconcile_synced_manifest(root, str(workdir), {"keep.txt": digest, "gone.txt": digest}, ["old.txt"], [])
            (workdir / "keep.txt").write_text("new remote data")
            result = rr.reconcile_synced_manifest(root, str(workdir), {}, ["gone.txt", "keep.txt", "old.txt"], [])
            self.assertEqual(result["pruned_ghost_files"], ["gone.txt"])
            self.assertEqual((workdir / "keep.txt").read_text(), "new remote data")
            self.assertEqual((workdir / "old.txt").read_text(), "uploaded")

    def test_literal_preserve_and_new_gitignore_rule_on_real_rsync(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, target = base / "local", base / "remote"
            source.mkdir()
            target.mkdir()
            preserved = target / "cache[one]"
            preserved.mkdir()
            (preserved / "keep.txt").write_text("cache")
            (target / "scratch.log").write_text("ignored")
            (source / ".gitignore").write_text("scratch.log\n")
            config = rr.empty_project_config()
            config["sync"]["preserve"] = ["cache[one]"]
            rr.run(["rsync", "-rlt", "--delete-after", *rr.build_rsync_filters(config, {}),
                    f"{source}/", f"{target}/"])
            self.assertEqual((preserved / "keep.txt").read_text(), "cache")
            self.assertEqual((target / "scratch.log").read_text(), "ignored")

    def test_prune_leaves_a_replaced_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "remote"
            rr.ensure_root_owned(root)
            runs = root / "runs"
            rr.private_directory(runs)
            job = runs / "job"
            job.mkdir()
            rr.write_json(job / rr.RUN_MARKER, {"schema_version": 1, "run_id": job.name})
            rr.write_json(job / "state.json", {"run_id": job.name, "status": "completed"})
            rr.write_json(job / "request.json", {"host": "box"})

            def replace_job(_):
                job.rename(root / "already-removed")
                job.mkdir()
                rr.write_json(job / "state.json", {"run_id": job.name, "status": "running"})
                return 0

            with mock.patch.object(rr.subprocess, "run", return_value=SimpleNamespace(returncode=1)), \
                 mock.patch.object(rr, "tmux_program", return_value="/unused/tmux"), \
                 mock.patch.object(rr, "dir_size_bytes", side_effect=replace_job):
                result = rr.prune_runs(root, {"name": "box"}, 0, True)
            self.assertEqual(result["runs_count"], 0)
            self.assertEqual(rr.read_json(job / "state.json")["status"], "running")

    def test_pull_reserves_local_destination_at_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "results"
            args = SimpleNamespace(destination=dest, force=False, max_file_mb=10, path=[],
                                   all_artifacts=False, run_id="job", paths_only=False)

            def metadata(*unused):
                dest.mkdir()
                (dest / "keep.txt").write_text("first pull")
                return {"job_files": [], "workdir_files": []}

            with mock.patch.object(rr, "ssh_call", side_effect=metadata), \
                 mock.patch.object(rr, "rsync_protect_args", return_value=[]), \
                 mock.patch.object(rr, "build_ssh_argv", return_value=["unused"]), \
                 mock.patch.object(rr, "run") as run:
                with self.assertRaises(FileExistsError):
                    rr.handle_pull(args, {"root": "/tmp/remote", "ssh": "unused"})
                run.assert_not_called()
            self.assertEqual((dest / "keep.txt").read_text(), "first pull")

    def test_workspace_pull_refuses_control_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "remote"
            job = root / "runs" / "job"
            job.mkdir(parents=True)
            rr.write_json(job / "request.json", {"host": "box"})
            with self.assertRaisesRegex(ValueError, "private state"):
                rr.eligible_pull_files(job, str(root), remote_root=root,
                                       extra_paths=["runs/job/request.json"], paths_only=True)

    def test_prepare_refuses_synchronizing_over_a_custom_locks_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "remote"
            workdir = base / "work"
            locks = workdir / "locks"
            for dry_run in (False, True):
                with self.assertRaisesRegex(ValueError, "overlaps the workdir"):
                    rr.prepare_remote_workspace(root, {"workdir": str(workdir)}, [],
                                                dry_run=dry_run, locks_dir=locks)
            self.assertFalse(workdir.exists())

    def test_cancel_never_signals_a_pid_without_a_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "job"
            job.mkdir()
            running = {"status": "running", "worker_pid": 1234, "child_pgid": 1235}
            rr.write_json(job / "state.json", running)
            with mock.patch.object(rr, "state_of", return_value=running), \
                 mock.patch.object(rr, "process_identity", return_value=["current", "2", "1235", "1235"]), \
                 mock.patch.object(rr, "run_process_alive", return_value=False), \
                 mock.patch.object(rr, "tmux_program", return_value="/unused/tmux"), \
                 mock.patch.object(rr.subprocess, "run"), \
                 mock.patch.object(rr.os, "kill") as kill, mock.patch.object(rr.os, "killpg") as killpg:
                result = rr.remote_cancel(root, job)
            self.assertTrue(result["cancellation_requested"])
            kill.assert_not_called()
            killpg.assert_not_called()

    def test_follow_keeps_a_multibyte_boundary_and_terminal_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            expected = "x" * 65535 + "波\r100%\nsecond line"
            log.write_bytes(expected.encode("utf-8"))

            def status(_host, _action, data, **kwargs):
                return {
                    "status": "completed",
                    "log": rr.read_log_slice(log, offset=data["offset"], tail_bytes=data["tail_bytes"], binary=True),
                }

            stdout = io.StringIO()
            with mock.patch.object(rr, "ssh_call", side_effect=status) as ssh, mock.patch.object(sys, "stdout", stdout):
                self.assertEqual(rr.follow_run({}, "job")["log_bytes"], log.stat().st_size)
            self.assertEqual(ssh.call_count, 2)
            self.assertEqual(stdout.getvalue(), expected)

    def test_workspace_lease_serializes_overlapping_local_control_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "remote"
            work = root / "work"
            host = {"name": "box", "ssh": "unused", "root": str(root)}
            control_command = rr.control_command
            started, locked = threading.Event(), threading.Event()
            errors = []

            def local_command(*args):
                argv, payload = control_command(*args)
                return [sys.executable, *shlex.split(argv[-1])[1:]], payload

            def second():
                started.set()
                try:
                    with rr.workspace_lease(host, str(work / "subdir")):
                        locked.set()
                except Exception as exc:
                    errors.append(exc)

            with mock.patch.object(rr, "control_command", side_effect=local_command), \
                 mock.patch.object(rr, "build_ssh_argv", return_value=["unused"]):
                with rr.workspace_lease(host, str(work)):
                    contender = threading.Thread(target=second)
                    contender.start()
                    acquired_while_held = started.wait(2) and locked.wait(0.2)
                contender.join(3)
            self.assertFalse(errors)
            self.assertFalse(acquired_while_held)
            self.assertTrue(locked.is_set())
            self.assertFalse(contender.is_alive())

    def test_directory_chain_allows_user_private_group_and_rejects_shared_writable(self):
        pw_self = SimpleNamespace(pw_name="alice", pw_uid=1000, pw_gid=1000)
        pw_other_diff_gid = SimpleNamespace(pw_name="bob", pw_uid=1001, pw_gid=1001)
        pw_other_same_gid = SimpleNamespace(pw_name="charlie", pw_uid=1002, pw_gid=1000)
        gr_upg = SimpleNamespace(gr_name="alice", gr_mem=[])
        gr_shared_name = SimpleNamespace(gr_name="users", gr_mem=[])
        gr_extra_mem = SimpleNamespace(gr_name="alice", gr_mem=["bob"])

        rr.is_single_user_group.cache_clear()
        try:
            with mock.patch.object(pwd, "getpwuid", return_value=pw_self), \
                 mock.patch.object(grp, "getgrgid", return_value=gr_upg), \
                 mock.patch.object(pwd, "getpwall", return_value=[pw_self, pw_other_diff_gid]):
                self.assertTrue(rr.is_single_user_group(1000, 1000))
            rr.is_single_user_group.cache_clear()
            with mock.patch.object(pwd, "getpwuid", return_value=pw_self), \
                 mock.patch.object(grp, "getgrgid", return_value=gr_shared_name), \
                 mock.patch.object(pwd, "getpwall", return_value=[pw_self]):
                self.assertFalse(rr.is_single_user_group(1000, 1000))
            rr.is_single_user_group.cache_clear()
            with mock.patch.object(pwd, "getpwuid", return_value=pw_self), \
                 mock.patch.object(grp, "getgrgid", return_value=gr_extra_mem), \
                 mock.patch.object(pwd, "getpwall", return_value=[pw_self]):
                self.assertFalse(rr.is_single_user_group(1000, 1000))
            rr.is_single_user_group.cache_clear()
            with mock.patch.object(pwd, "getpwuid", return_value=pw_self), \
                 mock.patch.object(grp, "getgrgid", return_value=gr_upg), \
                 mock.patch.object(pwd, "getpwall", return_value=[pw_self, pw_other_same_gid]):
                self.assertFalse(rr.is_single_user_group(1000, 1000))
        finally:
            rr.is_single_user_group.cache_clear()

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            os.chmod(base, 0o755)  # Prevent TemporaryDirectory's 0700 from short-circuiting private_parent
            home_dir = base / "home"
            home_dir.mkdir()
            os.chmod(home_dir, 0o750)
            local_share = home_dir / ".local" / "share" / "remote-run"
            local_share.mkdir(parents=True)
            for p in (home_dir / ".local", home_dir / ".local" / "share"):
                os.chmod(p, 0o775)
            os.chmod(local_share, 0o700)

            with mock.patch.object(rr, "is_single_user_group", return_value=True):
                rr.check_directory_chain(local_share, "Remote root", trusted=True)

            with mock.patch.object(rr, "is_single_user_group", return_value=False):
                with self.assertRaisesRegex(ValueError, "writable by another user"):
                    rr.check_directory_chain(local_share, "Remote root", trusted=True)

            # Non-sticky world-writable ancestor fails with actionable chmod +t hint; sticky passes
            mount = base / "shared_mount"
            mount.mkdir()
            sub = mount / "jobs"
            sub.mkdir()
            os.chmod(sub, 0o700)
            os.chmod(mount, 0o777)
            with self.assertRaisesRegex(ValueError, "chmod \\+t"):
                rr.check_directory_chain(sub, "Remote root", trusted=True)
            os.chmod(mount, 0o1777)
            rr.check_directory_chain(sub, "Remote root", trusted=True)

    def test_queued_gpu_lock_reports_blocking_runs_and_queue_next_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "remote"
            runs = root / "runs"
            locks_main = root / "locks"
            locks_other = root / "other-locks"
            holder = runs / "rr-holder"
            other_locks_holder = runs / "rr-other-locks"
            malformed = runs / "rr-malformed"
            waiter = runs / "rr-waiter"
            for d in (holder, other_locks_holder, malformed, waiter):
                d.mkdir(parents=True)
            rr.write_json(holder / "request.json", {
                "host": "desktop", "device": "cuda:0", "physical_device": "cuda:0",
                "locks_dir": str(locks_main), "use_gpu": True, "gpu_slots": ["gpu-0"],
                "workdir": str(root / "w1"),
            })
            rr.write_json(holder / "state.json", {
                "run_id": "rr-holder", "status": "running", "waiting_for_lock": False,
                "worker_pid": os.getpid(), "started_at": rr.now(),
            })
            rr.write_json(other_locks_holder / "request.json", {
                "host": "desktop", "device": "cuda:0", "physical_device": "cuda:0",
                "locks_dir": str(locks_other), "use_gpu": True, "gpu_slots": ["gpu-0"],
                "workdir": str(root / "w-other"),
            })
            rr.write_json(other_locks_holder / "state.json", {
                "run_id": "rr-other-locks", "status": "running", "waiting_for_lock": False,
                "worker_pid": os.getpid(), "started_at": rr.now(),
            })
            rr.write_json(malformed / "request.json", {
                "host": "desktop", "device": "cuda:0", "physical_device": "cuda:0",
                "locks_dir": str(locks_main), "use_gpu": True, "gpu_slots": [["gpu-0"]],
                "workdir": str(root / "w-bad"),
            })
            rr.write_json(malformed / "state.json", {
                "run_id": "rr-malformed", "status": "running", "waiting_for_lock": False,
                "worker_pid": os.getpid(), "started_at": rr.now(),
            })
            rr.write_json(waiter / "request.json", {
                "host": "desktop", "device": "cuda:0", "physical_device": "cuda:0",
                "locks_dir": str(locks_main), "use_gpu": True, "gpu_slots": ["gpu-0"],
                "workdir": str(root / "w2"),
            })
            rr.write_json(waiter / "state.json", {
                "run_id": "rr-waiter", "status": "queued", "waiting_for_lock": True,
                "worker_pid": os.getpid(), "submitted_at": rr.now(),
            })
            with mock.patch.object(rr, "run_process_alive", return_value=True), \
                 mock.patch.object(rr, "tmux_program", return_value="/unused/tmux"), \
                 mock.patch.object(rr.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
                st = rr.state_of(waiter, root)
                cancel_res = rr.remote_cancel(root, waiter)
            self.assertEqual(st["blocked_by_runs"], ["rr-holder"])
            self.assertEqual(cancel_res["blocked_by_runs"], ["rr-holder"])
            msg = rr.continuation({"name": "desktop"}, "rr-waiter", state=st)
            self.assertIn("Queued waiting for GPU lock", msg)
            self.assertIn("rr-holder", msg)

    def test_worker_does_not_mark_session_lock_wait_as_gpu_lock_wait(self):
        import fcntl
        import time

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "remote"
            ws = root / "ws"
            ws.mkdir(parents=True)
            sdir = root / "sessions"
            sdir.mkdir(parents=True)
            skey = rr.session_key_for(str(ws), "dev")
            job = root / "runs" / "rr-sess-wait"
            (job / "artifacts").mkdir(parents=True)
            rr.write_json(job / "request.json", {
                "host": "desktop", "device": "cuda:0", "physical_device": "cuda:0",
                "gpu_slot": "", "gpu_slots": ["gpu-0"], "locks_dir": str(root / "locks"),
                "use_gpu": True, "env_name": "none", "python": "", "auto_venv": False,
                "path_prepend": [], "env_static": {}, "env_vars": {}, "shared": {},
                "workdir": str(ws), "shared_dir": "", "rel_subdir": "",
                "project_id": "p", "workspace_id": "main", "session": "dev", "session_key": skey,
                "command": ["true"], "shell": "",
            })
            rr.write_json(job / "state.json", {
                "run_id": "rr-sess-wait", "status": "queued", "waiting_for_lock": False,
                "submitted_at": rr.now(),
            })

            errors = []
            with mock.patch.object(rr.signal, "signal"):
                with rr.open_lock(sdir / f"{skey}.lock") as held_slock:
                    fcntl.flock(held_slock, fcntl.LOCK_EX)
                    t = threading.Thread(target=lambda: errors.append(rr.worker(root, job)))
                    t.start()
                    deadline = time.monotonic() + 2.0
                    seen_pid = False
                    while time.monotonic() < deadline:
                        cur = rr.read_json(job / "state.json")
                        if "worker_pid" in cur:
                            seen_pid = True
                            break
                        time.sleep(0.02)
                    self.assertTrue(seen_pid)
                    time.sleep(0.1)
                    during_session_wait = rr.read_json(job / "state.json")
                    self.assertEqual(during_session_wait["status"], "queued")
                    self.assertFalse(during_session_wait["waiting_for_lock"])
                t.join(5.0)
            self.assertFalse(t.is_alive())
            self.assertEqual(errors, [None])
            final_state = rr.read_json(job / "state.json")
            self.assertEqual(final_state["status"], "completed")
            self.assertFalse(final_state["waiting_for_lock"])

    def test_failed_submit_and_interrupt_timing_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            args = SimpleNamespace(
                shell="echo hi", command=[], wait=15.0, tail_bytes=8192, raw_log=False,
                follow=False, no_sync=True, no_project=False, _no_project_mode=False,
                _in_git=False, remote_dir="", workspace="", env="", shared=[], env_var=[],
                device="", no_gpu=True, session="", run_id="rr-ghost-check", _config_path=None,
            )
            host = {
                "name": "desktop", "ssh": "unused", "root": str(Path(directory) / "remote"),
                "locks_dir": str(Path(directory) / "remote/locks"), "device": "cpu",
                "default_env": "", "path_prepend": [], "envs": {}, "shared": {},
            }
            stderr = io.StringIO()
            with mock.patch.object(rr, "workspace_lease", return_value=mock.MagicMock(__enter__=lambda s: None, __exit__=lambda *a: False)), \
                 mock.patch.object(rr, "ssh_call", side_effect=RuntimeError("No remote workspace")), \
                 mock.patch.object(sys, "stderr", stderr):
                with self.assertRaisesRegex(RuntimeError, "No remote workspace"):
                    rr.handle_exec(args, host, repo, "", rr.empty_project_config())
            self.assertEqual(stderr.getvalue(), "")

            stderr_sub = io.StringIO()
            with mock.patch.object(rr, "workspace_lease", return_value=mock.MagicMock(__enter__=lambda s: None, __exit__=lambda *a: False)), \
                 mock.patch.object(rr, "ssh_call", side_effect=KeyboardInterrupt()), \
                 mock.patch.object(sys, "stderr", stderr_sub):
                with self.assertRaises(SystemExit) as cm:
                    rr.handle_exec(args, host, repo, "", rr.empty_project_config())
            self.assertEqual(cm.exception.code, 130)
            self.assertIn("rr-ghost-check", stderr_sub.getvalue())
            self.assertIn("unconfirmed", stderr_sub.getvalue())
            self.assertIn("cancel rr-ghost-check", stderr_sub.getvalue())

            stderr_pre = io.StringIO()
            with mock.patch.object(rr, "workspace_lease", side_effect=KeyboardInterrupt()), \
                 mock.patch.object(sys, "stderr", stderr_pre):
                with self.assertRaises(SystemExit) as cm2:
                    rr.handle_exec(args, host, repo, "", rr.empty_project_config())
            self.assertEqual(cm2.exception.code, 130)
            self.assertNotIn("rr-ghost-check", stderr_pre.getvalue())

    def test_opaque_linked_worktree_uses_checkout_name_with_stable_hash(self):
        self.assertTrue(rr.is_opaque_worktree_name("1073435-060f401f-8538-4acf-8a2b-af9fbd590505"))
        self.assertTrue(rr.is_opaque_worktree_name("060f401f-8538-4acf-8a2b-af9fbd590505"))
        self.assertTrue(rr.is_opaque_worktree_name("123456789"))
        self.assertTrue(rr.is_opaque_worktree_name("0123456789abcdef0123"))
        self.assertFalse(rr.is_opaque_worktree_name("soil-physics-exp"))
        self.assertFalse(rr.is_opaque_worktree_name("wt1"))

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            common = (base / "repo" / ".git").resolve()
            opaque_wt_dir = common / "worktrees" / "1073435-060f401f-8538-4acf-8a2b-af9fbd590505"
            named_wt_dir = common / "worktrees" / "feat-box"
            checkout_named = (base / "app-main").resolve()
            checkout_opaque = (base / "99887766").resolve()
            for p in (opaque_wt_dir, named_wt_dir, checkout_named, checkout_opaque):
                p.mkdir(parents=True)

            host = {"root": "/data/jobs"}
            cfg = rr.empty_project_config()
            cfg["project"] = "proj-x"

            def fake_git(git_dir):
                def _query(argv, cwd=None):
                    if argv[:2] == ["rev-list", "--max-parents=0"]:
                        return True, "deadbeef1234"
                    if argv == ["rev-parse", "HEAD"]:
                        return True, "a" * 40
                    if argv == ["status", "--porcelain"]:
                        return True, ""
                    if argv == ["symbolic-ref", "--short", "HEAD"]:
                        return True, "feat/box"
                    if argv == ["rev-parse", "--git-dir"]:
                        return True, str(git_dir)
                    if argv == ["rev-parse", "--git-common-dir"]:
                        return True, str(common)
                    return False, ""
                return _query

            expected_opaque_tag = sha256(b"worktrees/1073435-060f401f-8538-4acf-8a2b-af9fbd590505").hexdigest()[:12]
            with mock.patch.object(rr, "git_query", side_effect=fake_git(opaque_wt_dir)):
                ws_opaque = rr.resolve_workspace(checkout_named, cfg, host, in_git=True)
            self.assertEqual(ws_opaque["workspace_id"], f"wt-app-main-{expected_opaque_tag}")

            with mock.patch.object(rr, "git_query", side_effect=fake_git(opaque_wt_dir)):
                ws_both_opaque = rr.resolve_workspace(checkout_opaque, cfg, host, in_git=True)
            self.assertEqual(
                ws_both_opaque["workspace_id"],
                f"wt-1073435-060f401f-8538-4acf-8a2b-af9fbd5905-{expected_opaque_tag}",
            )

            expected_named_tag = sha256(b"worktrees/feat-box").hexdigest()[:12]
            with mock.patch.object(rr, "git_query", side_effect=fake_git(named_wt_dir)):
                ws_named = rr.resolve_workspace(checkout_named, cfg, host, in_git=True)
            self.assertEqual(ws_named["workspace_id"], f"wt-feat-box-{expected_named_tag}")

    def test_pull_reports_workdir_destination_only_when_workdir_files_present(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            host = {"root": "/tmp/remote", "ssh": "unused"}

            dest_no_wd = base / "pull-only-job"
            args1 = SimpleNamespace(
                destination=dest_no_wd, force=False, max_file_mb=10.0, path=[],
                all_artifacts=False, run_id="rr-1", paths_only=False,
            )
            with mock.patch.object(rr, "ssh_call", return_value={"job_files": [], "workdir_files": []}), \
                 mock.patch.object(rr, "rsync_protect_args", return_value=[]), \
                 mock.patch.object(rr, "build_ssh_argv", return_value=["unused"]), \
                 mock.patch.object(rr, "run"):
                res1 = rr.handle_pull(args1, host)
            self.assertNotIn("workdir_destination", res1)

            dest_with_wd = base / "pull-with-wd"
            args2 = SimpleNamespace(
                destination=dest_with_wd, force=False, max_file_mb=10.0, path=["out/res.tsv"],
                all_artifacts=False, run_id="rr-2", paths_only=True,
            )
            with mock.patch.object(rr, "ssh_call", return_value={
                     "job_files": [], "workdir_files": ["out/res.tsv"], "workdir": "/tmp/remote/ws",
                 }), \
                 mock.patch.object(rr, "rsync_protect_args", return_value=[]), \
                 mock.patch.object(rr, "build_ssh_argv", return_value=["unused"]), \
                 mock.patch.object(rr, "run"):
                res2 = rr.handle_pull(args2, host)
            self.assertEqual(res2["workdir_destination"], str(dest_with_wd.resolve() / "workspace"))
