"""Unit, worker, local rsync, and MCP stdio protocol tests for remote-run."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import remote_run as rr


def git_init_commit(repo: Path, branch: str = "main") -> str:
    rr.run(["git", "init", "-q", "-b", branch], cwd=repo)
    (repo / "train.py").write_text("print('ok')\n")
    rr.run(["git", "add", "train.py"], cwd=repo)
    rr.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "init"],
        cwd=repo,
    )
    return rr.run(["git", "rev-parse", "HEAD"], cwd=repo)


class RemoteRunTests(unittest.TestCase):
    def test_symlink_bin_and_mcp_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "hosts.toml"
            cfg.write_text('schema_version = 1\ndefault_host = "b"\n[hosts.b]\nssh = "s"\nroot = "/data/r"\n')
            cli_link = Path(tmp) / "remote-run-link"
            cli_link.symlink_to(ROOT / "bin" / "remote-run")
            res = subprocess.run([str(cli_link), "--config", str(cfg), "hosts"], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(res.stdout)["default_host"], "b")

            mcp_link = Path(tmp) / "remote-run-mcp-link"
            mcp_link.symlink_to(ROOT / "bin" / "remote-run-mcp")
            req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
            mres = subprocess.run([str(mcp_link)], input=req + "\n", capture_output=True, text=True, check=True)
            init_resp = json.loads(mres.stdout.strip())
            self.assertEqual(init_resp["result"]["protocolVersion"], "2025-03-26")

    def test_strict_hosts_schema_and_path_validation(self):
        with self.assertRaisesRegex(ValueError, "filesystem root"):
            rr.remote_path("/")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "hosts.toml"
            cfg.write_text(
                'schema_version = 1\ndefault_host = "box"\n'
                '[hosts.box]\nssh = "box"\nroot = "/data/jobs"\ntypo_key = 123\n'
            )
            with self.assertRaisesRegex(ValueError, "Unknown key.*typo_key"):
                rr.load_hosts_config(cfg)

            cfg.write_text(
                'schema_version = 1\ndefault_host = "box"\n'
                '[hosts.box]\nssh = "box"\nroot = "/data/jobs"\ndefault_env = "missing"\n'
            )
            with self.assertRaisesRegex(ValueError, "not in envs"):
                rr.load_hosts_config(cfg)

            cfg.write_text(
                'schema_version = 1\ndefault_host = "box"\n'
                '[hosts.box]\nssh = "box"\nroot = "/data/jobs"\npath_prepend = ["/home/u/.local/bin"]\ndefault_env = "venv"\n'
            )
            _, hosts = rr.load_hosts_config(cfg)
            self.assertEqual(hosts["box"]["path_prepend"], ["/home/u/.local/bin"])
            self.assertEqual(rr.resolve_env(hosts["box"], "venv")["auto_venv"], True)
            self.assertEqual(rr.resolve_env(hosts["box"], "none")["python"], "")

    def test_strict_project_schema_and_config_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".remote-run.toml").write_text(
                'schema_version = 1\n[sync]\nignore = ["docs/"]\nbogus = true\n'
            )
            with self.assertRaisesRegex(ValueError, "Unknown key.*bogus"):
                rr.load_project_config(repo)

            (repo / ".remote-run.toml").write_text(
                'schema_version = 1\n[sync]\nignore = ["bad\\nline"]\n'
            )
            with self.assertRaisesRegex(ValueError, "clean non-empty"):
                rr.load_project_config(repo)

            cfg = Path(tmp) / "hosts.toml"
            cfg.write_text('schema_version = 1\ndefault_host = "d"\n[hosts.d]\nssh = "s"\nroot = "/data/r"\n')
            (repo / ".remote-run.toml").write_text('schema_version = 1\nenv = "missing_env"\n')
            bad = subprocess.run(
                [sys.executable, "-B", str(ROOT / "remote_run.py"), "--config", str(cfg), "config", "--repo", str(repo)],
                capture_output=True, text=True,
            )
            self.assertEqual(bad.returncode, 1)
            self.assertIn("missing_env", bad.stderr)

            (repo / ".remote-run.toml").write_text('schema_version = 1\nworkspace = "default"\n')
            good = subprocess.run(
                [sys.executable, "-B", str(ROOT / "remote_run.py"), "--config", str(cfg), "config", "--repo", str(repo)],
                capture_output=True, text=True, check=True,
            )
            cdata = json.loads(good.stdout)
            self.assertEqual(cdata["workspace"]["workspace_id"], "default")
            self.assertIn(".git/", cdata["effective_ignore"])
            self.assertIn("/results/", cdata["effective_ignore"])

    def test_git_subdir_normalization_move_and_worktree_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            p1 = base / "parent1"
            p2 = base / "parent2"
            p1.mkdir()
            p2.mkdir()
            repo = p1 / "my-app"
            repo.mkdir()
            sub = repo / "src" / "pkg"
            sub.mkdir(parents=True)
            (repo / ".remote-run.toml").write_text('schema_version = 1\nhost = "alt"\n')
            git_init_commit(repo, "main")

            root_found, rel_sub, in_git = rr.resolve_repo_context(sub)
            self.assertTrue(in_git)
            self.assertEqual(root_found, repo.resolve())
            self.assertEqual(rel_sub, "src/pkg")
            self.assertEqual(rr.load_project_config(root_found)["host"], "alt")

            host = {"root": "/data/jobs"}
            proj_cfg = rr.load_project_config(root_found)
            ws1 = rr.resolve_workspace(root_found, proj_cfg, host, rel_subdir=rel_sub, in_git=True)
            self.assertTrue(ws1["project_id"].startswith("my-app-"))

            moved = p2 / "my-app"
            repo.rename(moved)
            ws2 = rr.resolve_workspace(moved, proj_cfg, host, in_git=True)
            self.assertEqual(ws1["workdir"], ws2["workdir"])

            wt1 = base / "wt1"
            wt2 = base / "wt2"
            rr.run(["git", "worktree", "add", "-q", "-b", "feat/a-b", str(wt1)], cwd=moved)
            rr.run(["git", "worktree", "add", "-q", "-b", "feat/a/b", str(wt2)], cwd=moved)
            w_a = rr.resolve_workspace(wt1, proj_cfg, host, in_git=True)
            w_b = rr.resolve_workspace(wt2, proj_cfg, host, in_git=True)
            self.assertEqual(w_a["project_id"], w_b["project_id"])
            self.assertNotEqual(w_a["workspace_id"], w_b["workspace_id"])
            self.assertNotEqual(w_a["workdir"], w_b["workdir"])

    def test_non_git_collision_and_remote_dir_share_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d1 = base / "a" / "exp"
            d2 = base / "b" / "exp"
            d1.mkdir(parents=True)
            d2.mkdir(parents=True)
            host = {"root": "/data/jobs"}
            ws1 = rr.resolve_workspace(d1, rr.load_project_config(d1), host)
            ws2 = rr.resolve_workspace(d2, rr.load_project_config(d2), host)
            self.assertNotEqual(ws1["workdir"], ws2["workdir"])

            (d1 / ".remote-run.toml").write_text('schema_version = 1\n[share]\nproject = ["cache"]\n')
            with self.assertRaisesRegex(ValueError, "Cannot combine remote_dir"):
                rr.resolve_workspace(d1, rr.load_project_config(d1), host, remote_dir_override="/data/custom")

    def test_remote_submit_idempotent_reuse_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            workdir = root / "ws"
            workdir.mkdir()
            src = (ROOT / "remote_run.py").read_text()
            for fb in (False, True):
                rid = f"reuse-{fb}"
                job = root / "runs" / rid
                req = {
                    "host": "local", "device": "cpu", "physical_device": "cpu", "gpu_slot": "",
                    "locks_dir": str(root / "locks"), "use_gpu": False, "env_name": "none",
                    "python": "", "auto_venv": False, "path_prepend": [], "env_static": {},
                    "env_vars": {}, "shared": {}, "workdir": str(root / ("missing" if fb else "ws")),
                    "shared_dir": "", "rel_subdir": "", "fallback_root": fb,
                    "project_id": "p", "workspace_id": "main", "session": "", "session_key": "",
                    "command": ["echo", "hi"], "shell": "",
                }
                job.mkdir(parents=True)
                saved = rr.normalize_request(root, req)
                rr.write_json(job / "request.json", saved)
                rr.write_json(job / "state.json", {"run_id": rid, "status": "completed"})
                res = rr.remote_submit(root, job, {"root": str(root)}, req, src)
                self.assertTrue(res["reused"])
                with self.assertRaisesRegex(ValueError, "different request"):
                    rr.remote_submit(root, job, {"root": str(root)}, {**req, "command": ["echo", "other"]}, src)

    def test_local_worker_execution_and_session_edge_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            ws = root / "ws"
            ws.mkdir(parents=True)
            (ws / "sub").mkdir()
            # Create fake project .venv/bin/python
            vbin = ws / ".venv" / "bin"
            vbin.mkdir(parents=True)
            (vbin / "python").symlink_to(sys.executable)

            def run_worker(rid, *, command=None, shell="", session="", auto_venv=False, python_cfg=""):
                job = root / "runs" / rid
                (job / "artifacts").mkdir(parents=True)
                skey = rr.session_key_for(str(ws), session) if session else ""
                req = {
                    "host": "local", "device": "cpu", "physical_device": "cpu", "gpu_slot": "",
                    "locks_dir": str(root / "locks"), "use_gpu": False, "env_name": "venv" if auto_venv else "none",
                    "python": python_cfg, "auto_venv": auto_venv, "path_prepend": [], "env_static": {},
                    "env_vars": {}, "shared": {}, "workdir": str(ws), "shared_dir": "", "rel_subdir": "",
                    "project_id": "p", "workspace_id": "main", "session": session, "session_key": skey,
                    "command": command or [], "shell": shell,
                }
                rr.write_json(job / "request.json", req)
                rr.write_json(job / "state.json", {"run_id": rid, "status": "queued", "submitted_at": rr.now()})
                rr.worker(root, job)
                return rr.read_json(job / "state.json")

            # auto_venv activates ws/.venv/bin
            st_venv = run_worker("r-venv", command=["python", "-c", "import os; print(os.environ['VIRTUAL_ENV'])"], auto_venv=True)
            self.assertEqual(st_venv["status"], "completed")

            # Session multiline env, cd, unset, failed cmd non-pollution, deleted dir fallback
            st1 = run_worker("s1", session="dev", shell="cd sub && export MULTI=$'line1\\nline2' && export DROP=yes")
            self.assertEqual(st1["status"], "completed")
            st_fail = run_worker("s-fail", session="dev", shell="cd .. && export MULTI=broken && exit 4")
            self.assertEqual(st_fail["status"], "failed")
            self.assertEqual(st_fail["exit_code"], 4)
            st2 = run_worker("s2", session="dev", shell='test "$(basename "$PWD")" = "sub" && unset DROP')
            self.assertEqual(st2["status"], "completed")
            sdata = rr.read_json(root / "sessions" / (rr.session_key_for(str(ws), "dev") + ".json"))
            self.assertEqual(sdata["env"]["MULTI"], "line1\nline2")
            self.assertNotIn("DROP", sdata["env"])

            # Delete sub/ -> session falls back to ws cleanly
            (ws / "sub").rmdir()
            st3 = run_worker("s3", session="dev", shell='test "$PWD" = "$REMOTE_RUN_CODE"')
            self.assertEqual(st3["status"], "completed")

            # Python binary with only python3 sibling (no 'python' in that bin dir) must NOT use /usr/bin/python
            pyonly_dir = Path(tmp) / "pyonly"
            pyonly_dir.mkdir()
            py3_link = pyonly_dir / "python3"
            py3_link.symlink_to(sys.executable)
            st_py = run_worker("r-py", command=["python", "-c", "import sys; print(sys.executable)"], python_cfg=str(py3_link))
            self.assertEqual(st_py["status"], "completed")

    def test_eligible_pull_files_matrix_and_dangerous_dest_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / "job"
            (job / "artifacts").mkdir(parents=True)
            (job / "state.json").write_text("{}")
            (job / "artifacts" / "metrics_[0].json").write_text("{}")
            (job / "artifacts" / "model.pt").write_text("pt")
            (job / "artifacts" / "big.txt").write_bytes(b"x" * 2048)

            j_def, _, skip_def = rr.eligible_pull_files(job, str(root), max_file_mb=0.001)
            self.assertIn("artifacts/metrics_[0].json", j_def)
            self.assertEqual({s["path"] for s in skip_def}, {"artifacts/model.pt", "artifacts/big.txt"})

            j_all, _, skip_all = rr.eligible_pull_files(job, str(root), all_artifacts=True, max_file_mb=0.001)
            self.assertEqual(skip_all, [])
            self.assertIn("artifacts/model.pt", j_all)
            self.assertIn("artifacts/big.txt", j_all)

            workdir = root / "ws"
            shared_dir = root / "shared"
            outside = root / "outside.txt"
            outside.write_text("secret")
            workdir.mkdir()
            (shared_dir / "cache").mkdir(parents=True)
            (shared_dir / "cache" / "ok.txt").write_text("shared_ok")
            (workdir / "cache").symlink_to(shared_dir / "cache")
            (workdir / "escape").symlink_to(outside)

            _, wfiles, _ = rr.eligible_pull_files(
                job, str(workdir), shared_dir=str(shared_dir), extra_paths=["cache/ok.txt"]
            )
            self.assertEqual(wfiles, ["cache/ok.txt"])
            with self.assertRaisesRegex(ValueError, "cannot be a file symlink|escapes workspace"):
                rr.eligible_pull_files(job, str(workdir), shared_dir=str(shared_dir), extra_paths=["escape"])
            with self.assertRaisesRegex(ValueError, "finite"):
                rr.eligible_pull_files(job, str(workdir), max_file_mb=float("inf"))

            # Dangerous dest guard
            class Args:
                destination = Path(".")
                force = True
            with self.assertRaisesRegex(ValueError, "Refusing"):
                rr.handle_pull(Args(), {})
            Args.destination = Path("/")
            with self.assertRaisesRegex(ValueError, "Refusing"):
                rr.handle_pull(Args(), {})

    def test_local_rsync_anchored_results_and_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            src = base / "local src"
            dst = base / "remote dst"
            (src / "src" / "results").mkdir(parents=True)
            (src / "src" / "results" / "parser.py").write_text("OK = 1\n")
            (src / "results").mkdir()
            (src / "results" / "local_only.txt").write_text("skip")
            (dst / ".venv").mkdir(parents=True)
            (dst / ".venv" / "keep").write_text("keep")
            (dst / "outputs").mkdir(parents=True)
            (dst / "outputs" / "keep.txt").write_text("remote_keep")
            (src / "outputs").mkdir()
            (src / "outputs" / "keep.txt").write_text("local_should_not_overwrite")
            (dst / "reports").mkdir()
            (dst / "reports" / "unregistered.txt").write_text("must_survive_default_sync")
            (dst / "stale.py").write_text("old")
            (dst / rr.ROOT_MARKER).write_text("{}")
            (dst / rr.WORKSPACE_MARKER).write_text("{}")

            proj_cfg = rr.empty_project_config()
            filters = rr.build_rsync_filters(proj_cfg, {})
            # 1. Default additive sync: no --delete, remote outputs protected from local overwrite
            rr.run(["rsync", "-rlptz", "--protect-args", *filters, f"{src}/", f"{dst}/"])
            self.assertTrue((dst / "src" / "results" / "parser.py").is_file())
            self.assertFalse((dst / "results").exists())
            self.assertEqual((dst / "outputs" / "keep.txt").read_text(), "remote_keep")
            self.assertTrue((dst / "reports" / "unregistered.txt").is_file())
            self.assertTrue((dst / "stale.py").is_file())

            # 2. Dry-run --delete + explicit --delete preserves markers and preserve dirs
            dry = rr.run(["rsync", "-rlptz", "--delete", "-ni", "--itemize-changes", "--protect-args", *filters, f"{src}/", f"{dst}/"])
            _, would_del = rr.parse_rsync_itemize(dry)
            self.assertIn("stale.py", would_del)
            self.assertNotIn(rr.WORKSPACE_MARKER, would_del)
            rr.run(["rsync", "-rlptz", "--delete", "--protect-args", *filters, f"{src}/", f"{dst}/"])
            self.assertTrue((dst / ".venv" / "keep").is_file())
            self.assertTrue((dst / "outputs" / "keep.txt").is_file())
            self.assertTrue((dst / rr.WORKSPACE_MARKER).is_file())
            self.assertTrue((dst / rr.ROOT_MARKER).is_file())
            self.assertFalse((dst / "stale.py").exists())

    def test_machine_agnostic_ownership_topology_and_prune_safety(self):
        # Local repo cannot be / or $HOME
        with self.assertRaisesRegex(ValueError, "Refusing"):
            rr.resolve_repo_context(Path("/"))
        with self.assertRaisesRegex(ValueError, "Refusing"):
            rr.resolve_repo_context(Path.home())

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Remote root & workdir topology checks
            r = base / "remote_root"
            with self.assertRaisesRegex(ValueError, "unsafe remote root"):
                rr.validate_remote_root(Path("/"))
            with self.assertRaisesRegex(ValueError, "unsafe remote root"):
                rr.validate_remote_root(Path.home())
            with self.assertRaisesRegex(ValueError, "unsafe remote root"):
                rr.validate_remote_root(Path.home().parent)

            # Non-empty unmanaged root refused until init_root=True
            r.mkdir()
            (r / "user_file.txt").write_text("important")
            with self.assertRaisesRegex(ValueError, "init-root"):
                rr.ensure_root_owned(r)
            rr.ensure_root_owned(r, init_root=True)
            self.assertTrue((r / rr.ROOT_MARKER).is_file())

            # Workdir cannot equal root, contain root, or sit inside reserved dirs
            with self.assertRaisesRegex(ValueError, "equal or contain"):
                rr.validate_remote_workdir(r, r)
            with self.assertRaisesRegex(ValueError, "equal or contain"):
                rr.validate_remote_workdir(r, base)
            for bad_sub in (r / "runs" / "x", r / "locks" / "x", r / "sessions" / "x", r / "projects" / "p" / "shared"):
                with self.assertRaisesRegex(ValueError, "reserved|shared"):
                    rr.validate_remote_workdir(r, bad_sub)

            # Non-empty unmanaged workdir refused unless adopt=True; cross-project mismatch refused
            unmanaged_w = base / "existing_user_dir"
            unmanaged_w.mkdir()
            (unmanaged_w / "keep.txt").write_text("user_data")
            ws_a = {"project_id": "proj-a", "workspace_id": "main", "root_commit": "1111", "managed": False,
                    "workdir": str(unmanaged_w), "shared_dir": ""}
            with self.assertRaisesRegex(ValueError, "unmanaged remote directory"):
                rr.prepare_remote_workspace(r, ws_a, [])
            rr.prepare_remote_workspace(r, ws_a, [], adopt=True)
            self.assertTrue((unmanaged_w / rr.WORKSPACE_MARKER).is_file())
            ws_b = {**ws_a, "project_id": "proj-b"}
            with self.assertRaisesRegex(ValueError, "different project"):
                rr.prepare_remote_workspace(r, ws_b, [], adopt=True)

            # Prune skips unknown directories under runs/ even with apply=True, days=0
            user_exp = r / "runs" / "user_own_exp"
            user_exp.mkdir(parents=True)
            (user_exp / "weights.pt").write_text("precious")
            # Non-dict JSON metadata in runs/ is also skipped cleanly
            weird_dir = r / "runs" / "weird_json"
            weird_dir.mkdir(parents=True)
            (weird_dir / "state.json").write_text("[]")
            (weird_dir / "request.json").write_text("[]")
            owned_run = r / "runs" / "rr-old-1"
            owned_run.mkdir(parents=True)
            rr.write_json(owned_run / "request.json", {"host": "h"})
            rr.write_json(owned_run / "state.json", {"run_id": "rr-old-1", "status": "completed"})
            # Dead running run with old mtime: dry-run must preserve mtime & state.json, --apply must delete
            dead_running = r / "runs" / "rr-dead-running"
            dead_running.mkdir(parents=True)
            rr.write_json(dead_running / "request.json", {"host": "h", "device": "cpu", "workdir": str(r)})
            dead_st = {"run_id": "rr-dead-running", "status": "running", "started_at": "2020-01-01T00:00:00+00:00"}
            rr.write_json(dead_running / "state.json", dead_st)
            old_ts = 1_577_836_800.0
            os.utime(dead_running, (old_ts, old_ts))

            host = {"name": "h", "root": str(r), "locks_dir": str(r / "locks"), "device": "cpu"}
            with mock.patch.object(rr.subprocess, "run", return_value=mock.Mock(returncode=1)):
                dry_pruned = rr.control({"action": "prune", "host": host, "days": 14.0, "apply": False})
                self.assertEqual([x["run_id"] for x in dry_pruned["runs"]], ["rr-dead-running"])
                self.assertEqual(rr.read_json(dead_running / "state.json")["status"], "running")
                self.assertEqual(dead_running.stat().st_mtime, old_ts)
                # Ensure bad_sub under projects/ container is also refused
                for bad_proj in (r / "projects", r / "projects" / "p", r / "projects" / "p" / "workspaces"):
                    with self.assertRaisesRegex(ValueError, "workspaces"):
                        rr.validate_remote_workdir(r, bad_proj)
                # Apply with days=14 deletes rr-dead-running while keeping recent rr-old-1
                pruned14 = rr.control({"action": "prune", "host": host, "days": 14.0, "apply": True})
                self.assertEqual([x["run_id"] for x in pruned14["runs"]], ["rr-dead-running"])
                self.assertFalse(dead_running.exists())
                # Apply with days=0 deletes rr-old-1
                pruned = rr.control({"action": "prune", "host": host, "days": 0.0, "apply": True})
            self.assertEqual(pruned["runs_count"], 1)
            self.assertFalse(owned_run.exists())
            self.assertTrue((user_exp / "weights.pt").is_file())
            self.assertTrue(any("user_own_exp" in s["path"] for s in pruned["skipped_unmanaged"]))
            self.assertTrue(any("weird_json" in s["path"] for s in pruned["skipped_unmanaged"]))

    def test_dry_run_read_only_and_exec_no_sync_unmanaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # 1. Dry-run on uninitialized root + missing managed workdir creates zero files/dirs
            r = base / "fresh_root"
            w = r / "projects" / "p1" / "workspaces" / "main"
            s = r / "projects" / "p1" / "shared"
            ws_new = {"project_id": "p1", "workspace_id": "main", "root_commit": "abc", "managed": True,
                      "workdir": str(w), "shared_dir": str(s)}
            host = {"name": "h", "root": str(r), "locks_dir": str(r / "locks"), "device": "cpu"}
            prep = rr.control({
                "action": "prepare_workspace", "host": host, "workspace": ws_new,
                "share_project": ["data"], "dry_run": True,
            })
            self.assertFalse(prep["workspace_exists"])
            self.assertFalse(r.exists())

            # 2. Dry-run + adopt on existing non-empty unmanaged dir writes NO marker and NO submit.lock
            unmanaged = base / "existing_dir"
            unmanaged.mkdir()
            (unmanaged / "note.txt").write_text("keep")
            ws_unm = {"project_id": "p1", "workspace_id": "main", "root_commit": "", "managed": False,
                      "workdir": str(unmanaged), "shared_dir": ""}
            prep_adopt_dry = rr.control({
                "action": "prepare_workspace", "host": host, "workspace": ws_unm,
                "share_project": [], "adopt": True, "dry_run": True,
            })
            self.assertTrue(prep_adopt_dry["workspace_exists"])
            self.assertFalse((unmanaged / rr.WORKSPACE_MARKER).exists())
            self.assertFalse(r.exists())

            # 3. sync_workspace(..., dry_run=True) on missing workdir uses local empty tmpdir cleanly
            local_repo = base / "repo"
            local_repo.mkdir()
            (local_repo / "app.py").write_text("print(1)\n")
            proj_cfg = rr.empty_project_config()
            with mock.patch.object(rr, "ssh_call", return_value={
                "workdir": str(w), "shared_mounted": [], "run_exists": False,
                "workspace_exists": False, "active_runs_warning": [],
            }):
                sres = rr.sync_workspace(
                    local_repo, proj_cfg, {"name": "h", "ssh": "unused"}, ws_new, dry_run=True,
                )
            self.assertIn("app.py", sres["would_update"])
            self.assertEqual(sres["would_delete"], [])
            self.assertFalse(r.exists())

            # 4. exec --no-sync --remote-dir <non-empty unmanaged> (for_sync=False) succeeds without writing marker
            rr.ensure_root_owned(r, init_root=True)
            prep_nosync = rr.prepare_remote_workspace(r, ws_unm, [], for_sync=False)
            self.assertEqual(prep_nosync["workdir"], str(unmanaged.resolve()))
            self.assertFalse((unmanaged / rr.WORKSPACE_MARKER).exists())

            # 5. session list & reset --remote-dir from $HOME and from [share].project repo work cleanly
            cfg = base / "hosts.toml"
            remote_root = base / "remote_root_ctrl"
            cfg.write_text(f'schema_version = 1\ndefault_host = "h"\n[hosts.h]\nssh = "s"\nroot = "{remote_root}"\n')
            share_repo = base / "share_repo"
            share_repo.mkdir()
            (share_repo / ".remote-run.toml").write_text('schema_version = 1\n[share]\nproject = ["cache"]\n')

            for cwd_dir in (Path.home(), share_repo):
                with mock.patch.object(rr.Path, "cwd", return_value=cwd_dir):
                    with mock.patch.object(rr, "ssh_call", return_value={"sessions": []}) as mssh:
                        with mock.patch.object(sys, "argv", [
                            "remote-run", "--config", str(cfg), "session", "list", "--remote-dir", "/data/custom",
                        ]):
                            out, _, _ = rr.cli()
                    self.assertEqual(out, {"sessions": []})
                    self.assertEqual(mssh.call_args.args[2]["workdir"], "/data/custom")
                    with mock.patch.object(rr, "ssh_call", return_value={"reset": True}) as mssh:
                        with mock.patch.object(sys, "argv", [
                            "remote-run", "--config", str(cfg), "session", "reset", "dev", "--remote-dir", "/data/custom",
                        ]):
                            out, _, _ = rr.cli()
                    self.assertEqual(out, {"reset": True})
                    self.assertEqual(mssh.call_args.args[2], {"subaction": "reset", "workdir": "/data/custom", "name": "dev"})

            # 6. exec --no-sync from $HOME & [share].project repo, and unsafe .remote-run.toml remote_dir guard
            def fake_ssh(_host, action, data):
                if action == "prepare_workspace":
                    return rr.control({"action": "prepare_workspace", "host": {"name": "h", "root": str(remote_root), "locks_dir": str(remote_root / "locks"), "device": "cpu"}, **data})
                if action == "submit":
                    return {"run_id": data["run_id"], "status": "completed", "request": data["request"]}
                raise AssertionError(f"Unexpected action: {action}")

            rr.ensure_root_owned(remote_root, init_root=True)
            with mock.patch.object(rr.Path, "cwd", return_value=Path.home()):
                with mock.patch.object(rr, "ssh_call", side_effect=fake_ssh) as mssh:
                    with mock.patch.object(sys, "argv", [
                        "remote-run", "--config", str(cfg), "exec", "--no-sync", "--no-gpu", "--wait", "0", "--", "nvidia-smi",
                    ]):
                        out, _, _ = rr.cli()
                    self.assertEqual(out["status"], "completed")
                    with mock.patch.object(sys, "argv", [
                        "remote-run", "--config", str(cfg), "exec", "--no-sync", "--remote-dir", str(unmanaged), "--wait", "0", "--", "ls",
                    ]):
                        out, _, _ = rr.cli()
                    self.assertEqual(out["status"], "completed")
                # exec WITH sync from $HOME must still fail
                with mock.patch.object(sys, "argv", [
                    "remote-run", "--config", str(cfg), "exec", "--", "echo", "hi",
                ]):
                    with self.assertRaisesRegex(ValueError, "Refusing"):
                        rr.cli()

            # From [share].project repo (both implicit cwd and explicit --repo)
            for argv_extra in ([], ["--repo", str(share_repo)]):
                with mock.patch.object(rr.Path, "cwd", return_value=share_repo):
                    with mock.patch.object(rr, "ssh_call", side_effect=fake_ssh) as mssh:
                        with mock.patch.object(sys, "argv", [
                            "remote-run", "--config", str(cfg), "exec", *argv_extra,
                            "--no-sync", "--remote-dir", str(unmanaged), "--wait", "0", "--", "ls",
                        ]):
                            out, _, _ = rr.cli()
                        self.assertEqual(out["status"], "completed")
                        prep_call = [c for c in mssh.call_args_list if c.args[1] == "prepare_workspace"][0]
                        self.assertEqual(prep_call.args[2]["share_project"], [])
                        self.assertFalse(prep_call.args[2]["for_sync"])

            # .remote-run.toml with unsafe remote_dir must still be rejected by exec --no-sync
            bad_repo = base / "bad_rdir_repo"
            bad_repo.mkdir()
            (bad_repo / ".remote-run.toml").write_text(f'schema_version = 1\nremote_dir = "{remote_root / "runs"}"\n')
            with mock.patch.object(rr, "ssh_call", side_effect=fake_ssh):
                with mock.patch.object(sys, "argv", [
                    "remote-run", "--config", str(cfg), "exec", "--repo", str(bad_repo), "--no-sync", "--wait", "0", "--", "ls",
                ]):
                    with self.assertRaisesRegex(ValueError, "reserved control dir"):
                        rr.cli()

            # 7. prune dry-run on brand-new uninitialized root creates zero files/dirs
            uninit_root = base / "brand_new_prune_root"
            dry_empty = rr.control({"action": "prune", "host": {"name": "h", "root": str(uninit_root), "locks_dir": str(uninit_root / "locks"), "device": "cpu"}, "days": 14.0, "apply": False})
            self.assertEqual(dry_empty["runs_count"], 0)
            self.assertFalse(uninit_root.exists())

    def test_read_log_slice_corrupt_resilience_and_status_repo_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logp = root / "run.log"
            logp.write_bytes(b"a\r10%\r100%\nline2\n")
            s1 = rr.read_log_slice(logp, offset=0, tail_bytes=8)
            self.assertEqual(s1["offset"], 0)
            self.assertEqual(s1["next_offset"], 8)
            self.assertTrue(s1["truncated"])
            s2 = rr.read_log_slice(logp, offset=s1["next_offset"], tail_bytes=100, raw_log=True)
            self.assertFalse(s2["truncated"])
            self.assertEqual(s2["next_offset"], logp.stat().st_size)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                rr.read_log_slice(logp, tail_bytes=-1)

            # Corrupt run & session resilience
            runs_dir = root / "runs"
            (runs_dir / "bad-run").mkdir(parents=True)
            (runs_dir / "bad-run" / "state.json").write_text("{corrupt")
            (runs_dir / "bad-run" / "request.json").write_text("{}")
            (root / "sessions").mkdir()
            (root / "sessions" / "bad.json").write_text("{corrupt")
            (root / "sessions" / "list_json.json").write_text("[]")
            rr.ensure_root_owned(root, init_root=True)
            host = {"name": "h", "root": str(root), "locks_dir": str(root / "locks"), "device": "cpu"}
            listed = rr.control({"action": "list", "host": host, "limit": 10})
            self.assertEqual(listed["runs"][0]["status"], "corrupt")
            slisted = rr.control({"action": "session", "subaction": "list", "host": host})
            self.assertEqual([s["status"] for s in slisted["sessions"]], ["corrupt", "corrupt"])

            # status --repo picks project host
            cfg = root / "hosts.toml"
            cfg.write_text(
                'schema_version = 1\ndefault_host = "primary"\n'
                '[hosts.primary]\nssh = "s1"\nroot = "/data/p"\n'
                '[hosts.secondary]\nssh = "s2"\nroot = "/data/s"\n'
            )
            other = root / "other"
            other.mkdir()
            (other / ".remote-run.toml").write_text('schema_version = 1\nhost = "secondary"\n')
            # Trigger status --repo -> fails trying to SSH s2 (not s1!)
            proc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "remote_run.py"), "--config", str(cfg), "status", "--repo", str(other)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 1)

    def test_mcp_stdio_protocol_hardening(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "hosts.toml"
            cfg.write_text(
                'schema_version = 1\ndefault_host = "d"\n'
                '[hosts.d]\nssh = "s"\nroot = "/data/r"\ndevice = "cuda:0"\n'
            )
            reqs = [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled"}),
                "not json",
                json.dumps([1, 2, 3]),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "remote_hosts", "arguments": {"config": str(cfg)}}}),
                json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                            "params": {"name": "remote_config", "arguments": {"config": str(cfg), "repo": str(ROOT)}}}),
                json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                            "params": {"name": "remote_exec", "arguments": {"config": str(cfg), "command": ["echo", "hi"]}}}),
                json.dumps({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                            "params": {"name": "remote_cancel", "arguments": {}}}),
                json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                            "params": {"name": "remote_pull", "arguments": {"run_id": "r1", "destination": "--force"}}}),
                json.dumps({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                            "params": {"name": "remote_status", "arguments": {"config": str(cfg), "limit": None}}}),
                json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                            "params": {"name": "remote_exec",
                                       "arguments": {"config": str(cfg), "repo": str(ROOT), "command": ["echo"], "wait": 1e309}}}),
            ]
            proc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "remote_run_mcp.py")],
                input="\n".join(reqs) + "\n",
                capture_output=True, text=True, check=True,
            )
            lines = [json.loads(line) for line in proc.stdout.strip().splitlines()]
            self.assertEqual(len(lines), 11)
            self.assertEqual(lines[0]["result"]["protocolVersion"], "2025-03-26")
            self.assertEqual(lines[1]["error"]["code"], -32700)
            self.assertEqual(lines[2]["error"]["code"], -32600)
            tools = {t["name"]: t for t in lines[3]["result"]["tools"]}
            self.assertIn("remote_prune", tools)
            self.assertIn("annotations", tools["remote_exec"])
            self.assertTrue(tools["remote_exec"]["annotations"]["destructiveHint"])
            self.assertTrue(tools["remote_pull"]["annotations"]["destructiveHint"])
            sync_props = tools["remote_sync"]["inputSchema"]["properties"]
            self.assertIn("dry_run", sync_props)
            self.assertNotIn("delete", sync_props)
            self.assertNotIn("adopt", sync_props)
            self.assertTrue(lines[6]["result"]["isError"])  # missing repo on remote_exec
            self.assertTrue(lines[7]["result"]["isError"])  # missing run_id on remote_cancel
            self.assertTrue(lines[8]["result"]["isError"])  # --force flag injection blocked
            # lines[9] was limit=None -> fails only on SSH 's' (not TypeError!)
            self.assertNotIn("TypeError", lines[9]["result"]["content"][0]["text"])
            self.assertTrue(lines[10]["result"]["isError"])  # non-finite wait OverflowError -> isError


if __name__ == "__main__":
    unittest.main()
