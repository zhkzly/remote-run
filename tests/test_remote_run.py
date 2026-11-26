"""Unit, worker, local rsync, and MCP stdio protocol tests for remote-run."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
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

            # Minimal hosts.toml (only ssh required; root defaults to ~/.local/share/remote-run, device defaults to cpu)
            cfg.write_text('schema_version = 1\n[hosts.minbox]\nssh = "ubuntu@192.168.1.50"\nport = 22345\n')
            dhost, mhosts = rr.load_hosts_config(cfg)
            self.assertEqual(dhost, "minbox")
            self.assertEqual(mhosts["minbox"]["port"], 22345)
            self.assertEqual(mhosts["minbox"]["root"], rr.DEFAULT_REMOTE_ROOT)
            self.assertEqual(mhosts["minbox"]["device"], "cpu")
            self.assertEqual(mhosts["minbox"]["gpu_slots"], [])

            # Safe ssh / path / device / port / sha256 validation
            for bad_ssh in ("-oProxyCommand=evil", "user@host:2222", "bad host"):
                with self.assertRaises(ValueError):
                    rr.ssh_target(bad_ssh)
            for bad_path in ("~", "~/", "~/.", "~bob/dir", "/data/../etc", "relative/path"):
                with self.assertRaises(ValueError):
                    rr.remote_path(bad_path)
            self.assertEqual(rr.remote_path("~/jobs/"), "~/jobs")
            self.assertEqual(rr.expand_remote_path("~/jobs/"), str(Path.home() / "jobs"))

            with self.assertRaisesRegex(ValueError, "duplicate"):
                rr.validate_device("cuda:0,0")
            self.assertEqual(rr.device_gpu_slots("cuda:2,0"), ["gpu-0", "gpu-2"])
            self.assertEqual(rr.device_gpu_slots("cpu"), [])

            cfg.write_text(
                'schema_version = 1\ndefault_host = "box"\n'
                '[hosts.box]\nssh = "box"\nroot = "/data/jobs"\nport = 70000\n'
            )
            with self.assertRaisesRegex(ValueError, "1 and 65535"):
                rr.load_hosts_config(cfg)

            cfg.write_text(
                'schema_version = 1\ndefault_host = "box"\n'
                '[hosts.box]\nssh = "box"\nroot = "/data/jobs"\n'
                f'[hosts.box.shared.ds]\npath = "/data/ds"\nsha256 = "{"A" * 64}"\n'
            )
            _, shosts = rr.load_hosts_config(cfg)
            self.assertEqual(shosts["box"]["shared"]["ds"]["sha256"], "a" * 64)

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
            self.assertEqual(ws1["workspace_id"], "default")
            self.assertFalse(ws1["is_linked_worktree"])

            moved = p2 / "my-app"
            repo.rename(moved)
            rr.run(["git", "checkout", "-qb", "local-feature"], cwd=moved)
            ws2 = rr.resolve_workspace(moved, proj_cfg, host, in_git=True)
            self.assertEqual(ws1["workdir"], ws2["workdir"])

            wt1 = base / "wt1"
            wt2 = base / "wt2"
            rr.run(["git", "worktree", "add", "-q", "-b", "feat/a-b", str(wt1)], cwd=moved)
            rr.run(["git", "worktree", "add", "-q", "-b", "feat/a/b", str(wt2)], cwd=moved)
            w_a = rr.resolve_workspace(wt1, proj_cfg, host, in_git=True)
            w_b = rr.resolve_workspace(wt2, proj_cfg, host, in_git=True)
            self.assertTrue(w_a["is_linked_worktree"])
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
            (job / "artifacts" / "firmware.bin").write_bytes(b"ELF")
            (job / "artifacts" / "model.pt").write_text("pt")
            (job / "artifacts" / "big.txt").write_bytes(b"x" * 2048)

            j_def, _, skip_def = rr.eligible_pull_files(job, str(root), max_file_mb=0.001)
            self.assertIn("artifacts/metrics_[0].json", j_def)
            self.assertIn("artifacts/firmware.bin", j_def)
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
            po_j, po_w, po_s = rr.eligible_pull_files(
                job, str(workdir), shared_dir=str(shared_dir), extra_paths=["cache/ok.txt"], paths_only=True
            )
            self.assertEqual(po_j, [])
            self.assertEqual(po_w, ["cache/ok.txt"])
            self.assertEqual(po_s, [])
            with self.assertRaisesRegex(ValueError, "--paths-only"):
                rr.eligible_pull_files(job, str(workdir), paths_only=True)
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
                    self.assertEqual(out["request"]["project_id"], "")
                    self.assertEqual(out["request"]["workspace_id"], "")
                    self.assertEqual(out["request"]["shared_dir"], "")
                    with mock.patch.object(sys, "argv", [
                        "remote-run", "--config", str(cfg), "exec", "--no-sync", "--remote-dir", str(unmanaged), "--wait", "0", "--", "ls",
                    ]):
                        out, _, _ = rr.cli()
                    self.assertEqual(out["status"], "completed")
                    self.assertEqual(out["request"]["project_id"], "")
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
            by_id = {item.get("id"): item for item in lines if item.get("id") is not None}
            errs = [item for item in lines if item.get("id") is None]
            self.assertEqual(len(errs), 2)
            self.assertEqual(errs[0]["error"]["code"], -32700)
            self.assertEqual(errs[1]["error"]["code"], -32600)
            self.assertEqual(by_id[1]["result"]["protocolVersion"], "2025-03-26")
            self.assertEqual(by_id[1]["result"]["serverInfo"]["version"], "0.5.0")
            tools = {t["name"]: t for t in by_id[2]["result"]["tools"]}
            self.assertIn("remote_prune", tools)
            self.assertIn("repo", tools["remote_doctor"]["inputSchema"]["properties"])
            self.assertIn("check_cmds", tools["remote_doctor"]["inputSchema"]["properties"])
            self.assertIn("device", tools["remote_exec"]["inputSchema"]["properties"])
            self.assertIn("paths_only", tools["remote_pull"]["inputSchema"]["properties"])
            self.assertIn("annotations", tools["remote_exec"])
            self.assertTrue(tools["remote_exec"]["annotations"]["destructiveHint"])
            self.assertTrue(tools["remote_pull"]["annotations"]["destructiveHint"])
            sync_props = tools["remote_sync"]["inputSchema"]["properties"]
            self.assertIn("dry_run", sync_props)
            self.assertNotIn("delete", sync_props)
            self.assertNotIn("adopt", sync_props)
            self.assertFalse(by_id[3]["result"]["isError"])  # remote_hosts
            self.assertFalse(by_id[4]["result"]["isError"])  # remote_config
            self.assertTrue(by_id[5]["result"]["isError"])   # missing repo on remote_exec
            self.assertTrue(by_id[6]["result"]["isError"])   # missing run_id on remote_cancel
            self.assertTrue(by_id[7]["result"]["isError"])   # --force flag injection blocked
            # id=8 was limit=None -> fails only on SSH 's' (not TypeError!)
            self.assertNotIn("TypeError", by_id[8]["result"]["content"][0]["text"])
            self.assertTrue(by_id[9]["result"]["isError"])   # non-finite wait OverflowError -> isError

    def test_mcp_parallel_tool_calls_no_hol_blocking(self):
        import io
        import remote_run_mcp as mcp

        barrier = threading.Barrier(2, timeout=2.0)
        def concurrent_call(_name, _args):
            barrier.wait()
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}

        reqs = "\n".join([
            json.dumps({"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "remote_hosts", "arguments": {}}}),
            json.dumps({"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "remote_hosts", "arguments": {}}}),
        ]) + "\n"
        out_buf = io.StringIO()
        with mock.patch.object(mcp, "call_tool", side_effect=concurrent_call), \
             mock.patch.object(sys, "stdin", io.StringIO(reqs)), \
             mock.patch.object(sys, "stdout", out_buf):
            mcp.main()
        resps = {json.loads(line)["id"]: json.loads(line) for line in out_buf.getvalue().strip().splitlines()}
        self.assertEqual(set(resps), {10, 11})
        self.assertFalse(resps[10]["result"]["isError"])
        self.assertFalse(resps[11]["result"]["isError"])

    def test_doctor_status_wait_gpu_isolation_git_env_and_permissions(self):
        import fcntl

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            r = base / "root"
            rr.ensure_root_owned(r, init_root=True)
            host = {"name": "h", "root": str(r), "locks_dir": str(r / "locks"), "device": "cuda:0"}

            # 1 & 2. doctor refuses empty python for probes, and batched doctor reports missing_imports cleanly
            with mock.patch.object(rr.shutil, "which", return_value="/usr/bin/true"):
                with self.assertRaisesRegex(ValueError, "no host Python"):
                    rr.control({"action": "doctor", "host": {**host, "device": "cpu"}, "env_name": "venv", "python": "", "check_cuda": True})
                with mock.patch.object(rr, "run", side_effect=[
                    "0, Mock GPU, 12288 MiB",
                    "Python 3.12.0",
                    json.dumps({"imports": {"json": True, "nonexistent_mod_xyz": False}, "cuda_probe": {"cuda": False, "count": 0}}),
                ]):
                    dres = rr.control({
                        "action": "doctor", "host": host, "env_name": "py", "python": sys.executable,
                        "check_imports": ["json", "nonexistent_mod_xyz"], "check_cuda": True,
                    })
            self.assertEqual(dres["status"], "missing_imports")
            self.assertTrue(dres["_exec_failed"])
            self.assertEqual(dres["missing_imports"], ["nonexistent_mod_xyz"])
            self.assertTrue(dres["imports"]["json"])

            # 3. server-side status wait runs WITHOUT acquiring submit.lock
            job = r / "runs" / "rr-wait-test"
            job.mkdir(parents=True)
            rr.write_json(job / "request.json", {"host": "h", "device": "cpu", "workdir": str(r)})
            rr.write_json(job / "state.json", {"run_id": "rr-wait-test", "status": "completed"})
            (job / "run.log").write_text("done\n")
            with (r / "submit.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                st_res = rr.control({"action": "status", "host": host, "run_id": "rr-wait-test", "wait_sec": 0.1})
            self.assertEqual(st_res["status"], "completed")

            # 4. --no-gpu hides CUDA, git provenance env vars injected, user mkdir keeps natural umask, sessions stay 0700
            repo = base / "repo"
            repo.mkdir()
            head_sha = git_init_commit(repo, "main")
            (repo / "dirty.txt").write_text("x\n")
            ws_info = rr.resolve_workspace(repo, rr.empty_project_config(), {"root": str(r)}, in_git=True)
            self.assertEqual(ws_info["git_commit"], head_sha)
            self.assertEqual(ws_info["git_branch"], "main")
            self.assertTrue(ws_info["git_dirty"])

            wjob = r / "runs" / "rr-worker-check"
            (wjob / "artifacts").mkdir(parents=True)
            wdir = r / "ws_check"
            wdir.mkdir()
            skey = rr.session_key_for(str(wdir), "dev")
            req = {
                "host": "h", "device": "cuda:0", "physical_device": "cuda:0", "gpu_slot": "gpu-0",
                "locks_dir": str(r / "locks"), "use_gpu": False, "env_name": "none",
                "python": "", "auto_venv": False, "path_prepend": [], "env_static": {},
                "env_vars": {"G": "{git_commit}:{git_branch}"}, "shared": {}, "workdir": str(wdir),
                "shared_dir": "", "rel_subdir": "", "project_id": "p", "workspace_id": "main",
                "git_commit": head_sha, "git_branch": "main", "git_dirty": True,
                "session": "dev", "session_key": skey,
                "shell": "mkdir -p out_dir && python3 -c \"import os; print(os.environ['REMOTE_RUN_DEVICE'], repr(os.environ['CUDA_VISIBLE_DEVICES']), os.environ['REMOTE_RUN_GIT_COMMIT'], os.environ['REMOTE_RUN_GIT_BRANCH'], os.environ['REMOTE_RUN_GIT_DIRTY'], os.environ['G'])\"",
                "command": [],
            }
            rr.write_json(wjob / "request.json", req)
            rr.write_json(wjob / "state.json", {"run_id": "rr-worker-check", "status": "queued", "submitted_at": rr.now()})
            old_umask = os.umask(0o022)
            try:
                wproc = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "remote_run.py"), "_worker", str(r), str(wjob)],
                    capture_output=True, text=True, check=True,
                )
            finally:
                os.umask(old_umask)
            self.assertEqual(rr.read_json(wjob / "state.json")["status"], "completed")
            self.assertIn(f"cpu '' {head_sha} main 1 {head_sha}:main", wproc.stdout)
            self.assertNotEqual((wdir / "out_dir").stat().st_mode & 0o777, 0o700)
            self.assertEqual((r / "sessions").stat().st_mode & 0o777, 0o700)
            self.assertEqual((r / "sessions" / f"{skey}.json").stat().st_mode & 0o777, 0o600)

            # 5. doctor --repo infers host, env, and [share].host from another cwd; --host override ignores cwd defaults
            cfg = base / "doc_hosts.toml"
            cfg.write_text(
                f'schema_version = 1\ndefault_host = "other"\n'
                f'[hosts.primary]\nssh = "s1"\nroot = "{r}"\ndefault_env = "venv"\n'
                f'[hosts.primary.envs.pytorch]\npython = "{sys.executable}"\n'
                f'[hosts.primary.shared.ds]\npath = "{r / "ds"}"\n'
                f'[hosts.other]\nssh = "s2"\nroot = "{r}"\n'
            )
            doc_repo = base / "doc_repo"
            doc_repo.mkdir()
            (doc_repo / ".remote-run.toml").write_text(
                'schema_version = 1\nhost = "primary"\nenv = "pytorch"\n[share]\nhost = ["data=ds"]\n'
            )
            with mock.patch.object(rr.Path, "cwd", return_value=Path.home()):
                with mock.patch.object(rr, "ssh_call", return_value={"status": "ready"}) as mssh:
                    with mock.patch.object(sys, "argv", [
                        "remote-run", "--config", str(cfg), "doctor", "--repo", str(doc_repo),
                    ]):
                        d_out, _, _ = rr.cli()
            self.assertEqual(d_out["status"], "ready")
            self.assertEqual(mssh.call_args.args[0]["name"], "primary")
            self.assertEqual(mssh.call_args.args[2]["env_name"], "pytorch")
            self.assertEqual(mssh.call_args.args[2]["shared"]["data"]["path"], str(r / "ds"))

            with mock.patch.object(rr.Path, "cwd", return_value=doc_repo):
                with mock.patch.object(rr, "ssh_call", return_value={"status": "ready"}) as mssh:
                    with mock.patch.object(sys, "argv", [
                        "remote-run", "--config", str(cfg), "doctor", "--host", "other",
                    ]):
                        d_other, _, _ = rr.cli()
            self.assertEqual(d_other["status"], "ready")
            self.assertEqual(mssh.call_args.args[0]["name"], "other")
            self.assertEqual(mssh.call_args.args[2]["shared"], {})

    def test_control_path_prepend_no_git_check_cmd_cpu_clean_and_runs_0700(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            r = base / "root"
            rr.ensure_root_owned(r, init_root=True)
            bin_dir = base / "custom_bin"
            bin_dir.mkdir()
            tool = bin_dir / "g++"
            tool.write_text("#!/bin/sh\necho ok\n")
            tool.chmod(0o755)
            # host without git in PATH still succeeds if bash, tmux, rsync exist
            fake_bin = base / "fake_sys_bin"
            fake_bin.mkdir()
            for cmd_name in ("bash", "tmux", "rsync"):
                p = fake_bin / cmd_name
                p.write_text("#!/bin/sh\nexit 0\n")
                p.chmod(0o755)
            host = {"name": "h", "root": str(r), "locks_dir": str(r / "locks"), "device": "cpu", "path_prepend": [str(bin_dir)]}
            with mock.patch.dict(os.environ, {"PATH": str(fake_bin)}, clear=False):
                doc_ok = rr.control({"action": "doctor", "host": host, "env_name": "none", "python": "", "check_cmds": ["g++"]})
                self.assertEqual(doc_ok["status"], "ready")
                self.assertEqual(doc_ok["commands"]["g++"], str(tool))
                doc_miss = rr.control({"action": "doctor", "host": host, "env_name": "none", "python": "", "check_cmds": ["g++", "nonexistent_tool_xyz"]})
                self.assertEqual(doc_miss["status"], "missing_commands")
                self.assertTrue(doc_miss["_exec_failed"])
                self.assertEqual(doc_miss["missing_commands"], ["nonexistent_tool_xyz"])

            # CPU worker does NOT inject CUDA_VISIBLE_DEVICES or CUBLAS_WORKSPACE_CONFIG, and runs/ is 0700
            job = r / "runs" / "rr-cpu-clean"
            wdir = r / "ws_cpu"
            src = (ROOT / "remote_run.py").read_text()
            req = {
                "host": "h", "device": "cpu", "physical_device": "cpu", "gpu_slot": "", "gpu_slots": [],
                "locks_dir": str(r / "locks"), "use_gpu": False, "env_name": "none",
                "python": "", "auto_venv": False, "path_prepend": [], "env_static": {},
                "env_vars": {}, "shared": {}, "workdir": str(wdir), "shared_dir": "", "rel_subdir": "",
                "project_id": "p", "workspace_id": "default", "session": "", "session_key": "",
                "shell": "python3 -c \"import os; print('CVD' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NO_CVD', 'CUBLAS' if 'CUBLAS_WORKSPACE_CONFIG' in os.environ else 'NO_CUBLAS')\"",
                "command": [],
            }
            with mock.patch.object(rr, "run", return_value=""), \
                 mock.patch.object(rr.subprocess, "run", return_value=mock.Mock(returncode=0)):
                sub_res = rr.remote_submit(r, job, host, req, src)
            self.assertEqual((r / "runs").stat().st_mode & 0o777, 0o700)
            self.assertEqual(sub_res["tmux_session"], rr.session_name(r, "rr-cpu-clean"))
            clean_env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")}
            wproc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "remote_run.py"), "_worker", str(r), str(job)],
                env=clean_env, capture_output=True, text=True, check=True,
            )
            self.assertIn("NO_CVD NO_CUBLAS", wproc.stdout)

    def test_ssh_control_master_rsync_fallback_and_init_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fake_home = base / "home"
            fake_home.mkdir()
            host = {"ssh": "box", "port": 2222}
            with mock.patch.object(rr.Path, "home", return_value=fake_home), \
                 mock.patch.dict(os.environ, {"REMOTE_RUN_SSH_CONTROL": "1"}):
                argv = rr.build_ssh_argv(host)
                self.assertIn("ControlMaster=auto", argv)
                self.assertIn("ControlPersist=60", argv)
                self.assertIn("-p", argv)
                self.assertIn("2222", argv)
                self.assertEqual((fake_home / ".cache" / "remote-run" / "ssh").stat().st_mode & 0o777, 0o700)
            with mock.patch.dict(os.environ, {"REMOTE_RUN_SSH_CONTROL": "0"}):
                argv_off = rr.build_ssh_argv(host)
                self.assertNotIn("ControlMaster=auto", argv_off)

            # macOS rsync fallback (no --protect-args)
            rr.rsync_supports_protect_args.cache_clear()
            with mock.patch.object(rr.subprocess, "run", return_value=mock.Mock(stdout="rsync 2.6.9", stderr="")):
                self.assertEqual(rr.rsync_protect_args(), [])
                self.assertEqual(rr.rsync_remote_spec("box", "/tmp/dir with space"), "box:'/tmp/dir with space/'")
            rr.rsync_supports_protect_args.cache_clear()

            # remote-run init creates, appends, refuses duplicate without --force, updates with --force, handles dotted IPs & preserves existing parent mode
            cfg = base / "cfg" / "hosts.toml"
            existing_parent = base / "shared_parent"
            existing_parent.mkdir(mode=0o755)
            os.chmod(existing_parent, 0o755)
            probe1 = {"status": "ok", "tools": {"bash": "/bin/bash", "tmux": "/usr/bin/tmux", "rsync": "/usr/bin/rsync"},
                      "missing_tools": [], "device": "cuda:0", "gpus": [{"index": 0, "name": "GPU"}], "resolved_root": "/home/u/.local/share/remote-run"}
            with mock.patch.object(rr, "ssh_call", return_value=probe1):
                with mock.patch.object(sys, "argv", ["remote-run", "--config", str(cfg), "init", "ubuntu@10.0.0.2", "--port", "2222", "--name", "lab"]):
                    out1, _, _ = rr.cli()
            self.assertEqual(out1["host"], "lab")
            self.assertEqual(out1["device"], "cuda:0")
            self.assertEqual(cfg.stat().st_mode & 0o777, 0o600)
            dh, hs = rr.load_hosts_config(cfg)
            self.assertEqual(dh, "lab")
            self.assertEqual(hs["lab"]["port"], 2222)
            self.assertEqual(hs["lab"]["ssh"], "ubuntu@10.0.0.2")

            # Dotted IP host without --name quotes TOML key properly and does not chmod pre-existing parent dir
            ip_cfg = existing_parent / "hosts.toml"
            with mock.patch.object(rr, "ssh_call", return_value=probe1):
                with mock.patch.object(sys, "argv", ["remote-run", "--config", str(ip_cfg), "init", "ubuntu@192.168.1.50"]):
                    out_ip, _, _ = rr.cli()
            self.assertEqual(out_ip["host"], "192.168.1.50")
            self.assertEqual(existing_parent.stat().st_mode & 0o777, 0o755)
            dh_ip, hs_ip = rr.load_hosts_config(ip_cfg)
            self.assertEqual(dh_ip, "192.168.1.50")
            self.assertIn("192.168.1.50", hs_ip)

            # Duplicate without --force raises
            with mock.patch.object(rr, "ssh_call", return_value=probe1):
                with mock.patch.object(sys, "argv", ["remote-run", "--config", str(cfg), "init", "ubuntu@10.0.0.2", "--name", "lab"]):
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        rr.cli()

            # Add second CPU host with --default, then force-update lab
            probe2 = {**probe1, "device": "cpu", "gpus": []}
            with mock.patch.object(rr, "ssh_call", return_value=probe2):
                with mock.patch.object(sys, "argv", ["remote-run", "--config", str(cfg), "init", "cpu_srv", "--default"]):
                    out2, _, _ = rr.cli()
            self.assertEqual(out2["default_host"], "cpu_srv")
            dh2, hs2 = rr.load_hosts_config(cfg)
            self.assertEqual(dh2, "cpu_srv")
            self.assertIn("lab", hs2)

    def test_owned_manifest_ghost_cleanup_and_ignore_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            remote_root = base / "remote_root"
            rr.ensure_root_owned(remote_root, init_root=True)
            repo = base / "repo"
            repo.mkdir()
            (repo / "src").mkdir()
            (repo / "src" / "main.go").write_text("package main\n")
            (repo / "src" / "old.go").write_text("package main\n")
            (repo / "docs").mkdir()
            (repo / "docs" / "guide.md").write_text("guide\n")

            host_local = {"name": "h", "ssh": "unused", "root": str(remote_root), "locks_dir": str(remote_root / "locks"), "device": "cpu", "path_prepend": []}

            def fake_ssh(_host, action, data):
                return rr.control({"action": action, "host": host_local, **data})

            def fake_local_rsync(cmd, **kwargs):
                if cmd[0] == "rsync" and "-e" in cmd:
                    src_dir = cmd[-2]
                    dst_spec = cmd[-1]
                    dst_dir = dst_spec.split(":", 1)[1].strip("'")
                    idx_e = cmd.index("-e")
                    clean_cmd = cmd[:idx_e] + cmd[idx_e + 2:-2] + [src_dir, dst_dir]
                    return real_run(clean_cmd, **kwargs)
                return real_run(cmd, **kwargs)

            real_run = rr.run
            proj_cfg = rr.load_project_config(repo)
            ws_info = rr.resolve_workspace(repo, proj_cfg, host_local)
            wdir = Path(ws_info["workdir"])

            with mock.patch.object(rr, "ssh_call", side_effect=fake_ssh), mock.patch.object(rr, "run", side_effect=fake_local_rsync):
                s1 = rr.sync_workspace(repo, proj_cfg, host_local, ws_info)
            self.assertNotIn("pruned_ghost_files", s1)
            self.assertTrue((wdir / "src" / "old.go").is_file())
            self.assertTrue((wdir / "docs" / "guide.md").is_file())
            self.assertTrue((wdir / rr.SYNCED_MANIFEST).is_file())

            # Create remote-owned file and protected outputs/, then delete src/old.go and ignore docs/ (still on disk locally!)
            (wdir / "remote_gen.txt").write_text("keep_me\n")
            (wdir / "outputs").mkdir()
            (wdir / "outputs" / "res.txt").write_text("keep_output\n")
            (repo / "src" / "old.go").unlink()
            (repo / "src" / "renamed.go").write_text("package main\n")
            (repo / ".remote-run.toml").write_text('schema_version = 1\n[sync]\nignore = ["docs/"]\n')
            proj_cfg2 = rr.load_project_config(repo)

            with mock.patch.object(rr, "ssh_call", side_effect=fake_ssh), mock.patch.object(rr, "run", side_effect=fake_local_rsync):
                dry = rr.sync_workspace(repo, proj_cfg2, host_local, ws_info, dry_run=True)
                self.assertEqual(dry.get("would_prune_ghost"), ["src/old.go"])
                s2 = rr.sync_workspace(repo, proj_cfg2, host_local, ws_info)
            self.assertEqual(s2["pruned_ghost_files"], ["src/old.go"])
            self.assertFalse((wdir / "src" / "old.go").exists())
            self.assertTrue((wdir / "src" / "renamed.go").is_file())
            # Newly ignored file still present locally MUST NOT be pruned!
            self.assertTrue((wdir / "docs" / "guide.md").is_file())
            self.assertTrue((wdir / "remote_gen.txt").is_file())
            self.assertTrue((wdir / "outputs" / "res.txt").is_file())

            # Defense-in-depth adversarial check on reconcile_synced_manifest
            outside = base / "secret.txt"
            outside.write_text("secret")
            (wdir / "sym_link").symlink_to(outside)
            rr.write_json(wdir / rr.SYNCED_MANIFEST, {
                "schema_version": 1, "updated_at": rr.now(),
                "files": ["outputs/res.txt", rr.WORKSPACE_MARKER, "../secret.txt", "sym_link"],
            })
            rec = rr.reconcile_synced_manifest(
                remote_root, str(wdir), [], ["outputs/res.txt", rr.WORKSPACE_MARKER, "../secret.txt", "sym_link"], rr.BUILTIN_PRESERVE,
            )
            self.assertEqual(rec["pruned_ghost_files"], [])
            self.assertTrue(outside.is_file())
            self.assertTrue((wdir / "outputs" / "res.txt").is_file())
            self.assertTrue((wdir / rr.WORKSPACE_MARKER).is_file())

    def test_multi_gpu_sorted_locking_and_concurrency_stress(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            ws = root / "ws"
            ws.mkdir(parents=True)
            rr.ensure_root_owned(root, init_root=True)
            events_log = base / "events.jsonl"
            file_lock = threading.Lock()

            def run_gpu_worker(rid, dev):
                job = root / "runs" / rid
                (job / "artifacts").mkdir(parents=True)
                slots = rr.device_gpu_slots(dev)
                req = {
                    "host": "h", "device": dev, "physical_device": dev,
                    "gpu_slot": ",".join(slots), "gpu_slots": slots,
                    "locks_dir": str(root / "locks"), "use_gpu": True, "env_name": "none",
                    "python": "", "auto_venv": False, "path_prepend": [], "env_static": {},
                    "env_vars": {}, "shared": {}, "workdir": str(ws), "shared_dir": "", "rel_subdir": "",
                    "project_id": "p", "workspace_id": "default", "session": "", "session_key": "",
                    "command": [
                        sys.executable, "-c",
                        (
                            "import fcntl, json, os, time; "
                            f"lp = {repr(str(events_log))}; "
                            "t0 = time.monotonic(); time.sleep(0.18); t1 = time.monotonic(); "
                            "rec = {'rid': os.environ['REMOTE_RUN_ID'], 'dev': os.environ['REMOTE_RUN_DEVICE'], "
                            "'cvd': os.environ.get('CUDA_VISIBLE_DEVICES', ''), 't0': t0, 't1': t1}; "
                            "fh = open(lp, 'a'); fcntl.flock(fh, fcntl.LOCK_EX); fh.write(json.dumps(rec)+'\\n'); fh.close()"
                        ),
                    ],
                    "shell": "",
                }
                rr.write_json(job / "request.json", req)
                rr.write_json(job / "state.json", {"run_id": rid, "status": "queued", "submitted_at": rr.now()})
                subprocess.run(
                    [sys.executable, "-B", str(ROOT / "remote_run.py"), "_worker", str(root), str(job)],
                    capture_output=True, text=True, check=True, timeout=15,
                )
                return rr.read_json(job / "state.json")

            specs = [("g0-a", "cuda:0"), ("g1-a", "cuda:1"), ("g01", "cuda:0,1"), ("g0-b", "cuda:0"), ("g1-b", "cuda:1")]
            results = {}
            threads = []
            for rid, dev in specs:
                def runner(r=rid, d=dev):
                    st = run_gpu_worker(r, d)
                    with file_lock:
                        results[r] = st
                t = threading.Thread(target=runner)
                threads.append(t)
                t.start()
            for t in threads:
                t.join(timeout=20)
                self.assertFalse(t.is_alive(), "Potential multi-GPU lock deadlock!")
            self.assertEqual({k: v["status"] for k, v in results.items()}, {r: "completed" for r, _ in specs})

            recs = {r["rid"]: r for r in (json.loads(line) for line in events_log.read_text().splitlines() if line.strip())}
            self.assertEqual(recs["g01"]["cvd"], "0,1")
            self.assertEqual(recs["g01"]["dev"], "cuda:0,1")
            # cuda:0,1 must NEVER overlap in time with any cuda:0 or cuda:1 run
            g01_t0, g01_t1 = recs["g01"]["t0"], recs["g01"]["t1"]
            for other in ("g0-a", "g1-a", "g0-b", "g1-b"):
                o_t0, o_t1 = recs[other]["t0"], recs[other]["t1"]
                overlap = min(g01_t1, o_t1) - max(g01_t0, o_t0)
                self.assertLessEqual(overlap, 0.01, f"g01 overlapped with {other} by {overlap}s!")

    def test_scale_prune_non_blocking_combined_submit_follow_and_mcp_stress(self):
        import fcntl
        import io
        import time
        import remote_run_mcp as mcp

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            rr.ensure_root_owned(root, init_root=True)
            host = {"name": "h", "root": str(root), "locks_dir": str(root / "locks"), "device": "cpu", "path_prepend": []}

            # Create 400 terminal runs + 1000 workspace files
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True)
            ws_dir = root / "projects" / "p1" / "workspaces" / "default"
            ws_dir.mkdir(parents=True)
            for i in range(400):
                jd = runs_dir / f"rr-term-{i:04d}"
                jd.mkdir()
                rr.write_json(jd / "request.json", {"host": "h", "device": "cpu", "workdir": str(ws_dir)})
                rr.write_json(jd / "state.json", {"run_id": jd.name, "status": "completed"})
            for j in range(1000):
                (ws_dir / f"f{j:04d}.txt").write_text("x")

            t0 = time.monotonic()
            active = rr.active_runs_for_workdir(root, str(ws_dir))
            t_active = time.monotonic() - t0
            self.assertEqual(active, [])
            self.assertLess(t_active, 2.0)

            # Prune runs WITHOUT acquiring submit.lock and returns workspace paths without bytes rglob
            with mock.patch.object(rr.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="")):
                with (root / "submit.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    p_res = rr.control({"action": "prune", "host": host, "days": 0.0, "apply": True})
            self.assertEqual(p_res["runs_count"], 400)
            self.assertEqual(p_res["workspaces"], [{"path": str(ws_dir)}])

            # Streaming follow_run over multi-chunk wake_on_output + combined submit wait_sec
            fjob = runs_dir / "rr-follow-1"
            fjob.mkdir(parents=True)
            rr.write_json(fjob / "request.json", {"host": "h", "device": "cpu", "workdir": str(ws_dir)})
            rr.write_json(fjob / "state.json", {"run_id": "rr-follow-1", "status": "running", "started_at": rr.now()})
            (fjob / "run.log").write_text("line1\n")

            calls = {"n": 0}
            def fake_follow_ssh(_host, action, data, **_kw):
                calls["n"] += 1
                if calls["n"] == 2:
                    (fjob / "run.log").write_text("line1\nline2\n")
                    rr.write_json(fjob / "state.json", {"run_id": "rr-follow-1", "status": "completed", "started_at": rr.now(), "finished_at": rr.now()})
                return rr.control({"action": action, "host": host, **data})

            cap = io.StringIO()
            with mock.patch.object(rr.subprocess, "run", return_value=mock.Mock(returncode=0)), \
                 mock.patch.object(rr, "ssh_call", side_effect=fake_follow_ssh), \
                 mock.patch.object(sys, "stdout", cap):
                f_res = rr.follow_run(host, "rr-follow-1")
            self.assertEqual(f_res["status"], "completed")
            self.assertEqual(cap.getvalue(), "line1\nline2\n")

            # 16-way concurrent MCP stress test
            cfg = base / "mcp_hosts.toml"
            cfg.write_text(f'schema_version = 1\ndefault_host = "h"\n[hosts.h]\nssh = "s"\nroot = "{root}"\n')
            reqs = "\n".join([
                *(
                    json.dumps({"jsonrpc": "2.0", "id": 100 + k, "method": "tools/call",
                                "params": {"name": "remote_hosts", "arguments": {"config": str(cfg)}}})
                    for k in range(16)
                ),
                json.dumps({"jsonrpc": "2.0", "id": 200, "method": "tools/call",
                            "params": {"name": "remote_exec", "arguments": {"config": str(cfg), "repo": str(ROOT), "env_vars": ["bad"]}}}),
                json.dumps({"jsonrpc": "2.0", "id": 201, "method": "tools/call",
                            "params": {"name": "remote_exec", "arguments": {"config": str(cfg), "repo": str(ROOT), "command": "echo"}}}),
            ]) + "\n"
            out_buf = io.StringIO()
            with mock.patch.object(sys, "stdin", io.StringIO(reqs)), mock.patch.object(sys, "stdout", out_buf):
                mcp.main()
            resps = {json.loads(line)["id"]: json.loads(line) for line in out_buf.getvalue().strip().splitlines()}
            self.assertEqual(set(resps), set(range(100, 116)) | {200, 201})
            self.assertTrue(all(not resps[k]["result"]["isError"] for k in range(100, 116)))
            self.assertTrue(resps[200]["result"]["isError"])
            self.assertTrue(resps[201]["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
