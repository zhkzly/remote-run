"""Focused regressions for the release audit."""
from hashlib import sha256
import io
from pathlib import Path
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
