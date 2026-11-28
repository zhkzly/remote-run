"""CLI, MCP stdio protocol, and standalone-worker end-to-end tests for remote-run v2."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import remote_run_mcp as mcp


def make_exec(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TEMPLATE_DIR": "",
    }
    res = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True)
    return res.stdout.strip()


class CliMcpAndE2ETests(unittest.TestCase):
    def test_bin_launchers_and_mcp_protocol_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            xdg = t / "xdg"
            cfg_file = xdg / "remote-run" / "hosts.toml"
            cfg_file.parent.mkdir(parents=True)
            cfg_file.write_text('schema_version = 2\ndefault_host = "b"\n[hosts.b]\nssh = "s"\n')

            env = {**os.environ, "XDG_CONFIG_HOME": str(xdg), "REMOTE_RUN_SSH_CONTROL": "0"}
            cli_link = t / "remote-run-link"
            cli_link.symlink_to(ROOT / "bin" / "remote-run")
            ver = subprocess.run([str(cli_link), "--version"], env=env, capture_output=True, text=True, check=True)
            self.assertIn("2.0.0", ver.stdout)
            res = subprocess.run([str(cli_link), "hosts"], env=env, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            self.assertEqual(data["default_host"], "b")
            self.assertIsNone(data["hosts"]["b"]["gpu"])

            mcp_link = t / "remote-run-mcp-link"
            mcp_link.symlink_to(ROOT / "bin" / "remote-run-mcp")
            reqs = "\n".join([
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
                json.dumps({
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "remote_hosts", "arguments": {}},
                }),
                json.dumps({
                    "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {
                        "name": "remote_exec",
                        "arguments": {"repo": str(t), "argv": ["echo", "hi"], "shell": "echo hi"},
                    },
                }),
                json.dumps({
                    "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {
                        "name": "remote_exec",
                        "arguments": {"repo": str(t), "argv": [], "shell": "echo hi"},
                    },
                }),
            ]) + "\n"
            mres = subprocess.run([str(mcp_link)], input=reqs, env=env, capture_output=True, text=True, check=True)
            responses = {json.loads(line)["id"]: json.loads(line) for line in mres.stdout.splitlines() if line.strip()}
            self.assertEqual(responses[1]["result"]["protocolVersion"], "2025-06-18")
            self.assertEqual(responses[1]["result"]["serverInfo"]["version"], "2.0.0")

            tools = {tool["name"]: tool for tool in responses[2]["result"]["tools"]}
            self.assertTrue(tools["remote_doctor"]["annotations"]["readOnlyHint"])
            self.assertTrue(tools["remote_doctor"]["annotations"]["idempotentHint"])
            self.assertNotIn("check_imports", tools["remote_doctor"]["inputSchema"]["properties"])
            self.assertNotIn("check_cuda", tools["remote_doctor"]["inputSchema"]["properties"])
            self.assertIn("oneOf", tools["remote_exec"]["inputSchema"])

            self.assertFalse(responses[3]["result"]["isError"])
            self.assertEqual(responses[3]["result"]["structuredContent"]["default_host"], "b")
            # Mutual exclusion error for passing both argv and shell (even when argv is empty).
            self.assertTrue(responses[4]["result"]["isError"])
            self.assertIn("never both", responses[4]["result"]["content"][0]["text"])
            self.assertTrue(responses[5]["result"]["isError"])
            self.assertIn("never both", responses[5]["result"]["content"][0]["text"])

            # status --print-log without RUN_ID fails cleanly.
            bad_st = subprocess.run([str(cli_link), "status", "--print-log"], env=env, capture_output=True, text=True)
            self.assertEqual(bad_st.returncode, 1)
            self.assertIn("--print-log requires a RUN_ID", bad_st.stderr)

            # _service_rpc validates malformed JSON cleanly without traceback.
            bad_rpc = subprocess.run(
                [sys.executable, "-B", str(ROOT / "remote_run.py"), "_service_rpc"],
                input="not-json", env=env, capture_output=True, text=True,
            )
            self.assertEqual(bad_rpc.returncode, 1)
            self.assertIn("Invalid service RPC JSON", bad_rpc.stderr)

    def test_full_local_e2e_standalone_worker_sync_exec_status_pull_prune(self):
        """End-to-end test over a local fake SSH shim that runs the shipped remote_worker.py in isolation."""
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            xdg = t / "xdg"
            fake_bin = t / "fake_bin"
            remote_root = t / "remote_root"
            repo = t / "demo_repo"
            pull_dest = t / "pulled_out"

            # Fake SSH shim: executes the remote command locally with an isolated clean working dir,
            # proving ONLY the shipped remote_worker.py source is needed on the remote side!
            remote_cwd = t / "remote_empty_cwd"
            remote_cwd.mkdir()
            make_exec(
                fake_bin / "ssh",
                f"#!/usr/bin/env python3\n"
                f"import os, subprocess, sys\n"
                f"args = sys.argv[1:]\n"
                f"i = 0\n"
                f"while i < len(args):\n"
                f"    a = args[i]\n"
                f"    if a in ('-o', '-p', '-l', '-i'):\n"
                f"        i += 2\n"
                f"    elif a.startswith('-'):\n"
                f"        i += 1\n"
                f"    else:\n"
                f"        break\n"
                f"# args[i] is host, args[i+1:] is remote command\n"
                f"remote_cmd = ' '.join(args[i+1:])\n"
                f"env = dict(os.environ)\n"
                f"env.pop('PYTHONPATH', None)\n"
                f"p = subprocess.run(['/bin/sh', '-c', remote_cmd], cwd={str(remote_cwd)!r}, env=env)\n"
                f"sys.exit(p.returncode)\n",
            )
            # Fake tmux shim: runs detached command in background using setsid-like start_new_session.
            tmux_state = t / "tmux_pids"
            tmux_state.mkdir()
            make_exec(
                fake_bin / "tmux",
                f"#!/usr/bin/env python3\n"
                f"import os, pathlib, subprocess, sys\n"
                f"args = sys.argv[1:]\n"
                f"state = pathlib.Path({str(tmux_state)!r})\n"
                f"if args[:1] == ['new-session']:\n"
                f"    sess = args[args.index('-s') + 1]\n"
                f"    cwd = args[args.index('-c') + 1]\n"
                f"    cmd = args[args.index('--') + 1:]\n"
                f"    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
                f"    (state / sess).write_text(str(p.pid))\n"
                f"    sys.exit(0)\n"
                f"if args[:1] == ['has-session']:\n"
                f"    sess = args[args.index('-t') + 1].lstrip('=')\n"
                f"    pf = state / sess\n"
                f"    if not pf.is_file(): sys.exit(1)\n"
                f"    try:\n"
                f"        os.kill(int(pf.read_text().strip()), 0)\n"
                f"        sys.exit(0)\n"
                f"    except OSError:\n"
                f"        sys.exit(1)\n"
                f"if args[:1] == ['list-sessions']:\n"
                f"    for f in state.iterdir():\n"
                f"        try:\n"
                f"            os.kill(int(f.read_text().strip()), 0)\n"
                f"            print(f.name)\n"
                f"        except OSError:\n"
                f"            pass\n"
                f"    sys.exit(0)\n"
                f"sys.exit(0)\n",
            )

            # Configure hosts.toml
            cfg_file = xdg / "remote-run" / "hosts.toml"
            cfg_file.parent.mkdir(parents=True)
            cfg_file.write_text(
                f'schema_version = 2\n'
                f'default_host = "local-sim"\n'
                f'[hosts.local-sim]\n'
                f'ssh = "sim-host"\n'
                f'root = "{remote_root}"\n'
                f'path = ["{fake_bin}"]\n'
            )

            # Create a polyglot Git repo with tracked results/ and a custom script.
            repo.mkdir()
            git(repo, "init", "-q", "-b", "main")
            (repo / ".remote-run.toml").write_text(
                'schema_version = 2\n'
                'project = "e2e-demo"\n'
                'path = ["scripts"]\n'
                '[env]\n'
                'DEMO_MODE = "e2e"\n'
                '[workspace]\n'
                'shared = ["shared-cache"]\n'
                '[sync]\n'
                'preserve = ["build"]\n'
                'max_untracked_mib = 16\n'
                '[pull]\n'
                'max_file_mib = 5\n'
                'exclude = ["**/*.trace"]\n'
            )
            (repo / "results").mkdir()
            (repo / "results" / "input.txt").write_text("tracked-result-input\n")
            make_exec(
                repo / "scripts" / "run_demo",
                "#!/bin/sh\n"
                "set -eu\n"
                "echo \"mode=$DEMO_MODE msg=$1\"\n"
                "cat results/input.txt\n"
                "echo 'keep' > \"$REMOTE_RUN_ARTIFACTS/report.txt\"\n"
                "echo 'skip' > \"$REMOTE_RUN_ARTIFACTS/debug.trace\"\n"
                "echo 'shared-val' > shared-cache/item.txt\n",
            )
            git(repo, "add", ".remote-run.toml", "results/input.txt", "scripts/run_demo")
            git(repo, "-c", "user.name=T", "-c", "user.email=t@t", "commit", "-qm", "init")

            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                "XDG_CONFIG_HOME": str(xdg),
                "REMOTE_RUN_SSH_CONTROL": "0",
            }
            cli = [str(ROOT / "bin" / "remote-run")]

            # 1. Read-only doctor BEFORE any sync: must NOT create remote_root.
            doc = json.loads(subprocess.run(
                [*cli, "doctor", "--repo", str(repo), "--check-cmd", "sh"],
                env=env, capture_output=True, text=True, check=True,
            ).stdout)
            self.assertEqual(doc["status"], "ready")
            self.assertEqual(doc["root_state"], "pending")
            self.assertFalse(remote_root.exists())

            # 2. Exec (auto-sync + run script from project relative path `scripts/` with argument containing spaces).
            ex_proc = subprocess.run(
                [*cli, "exec", "--repo", str(repo), "--wait", "15", "--", "run_demo", "hello world"],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(ex_proc.returncode, 0, f"stdout={ex_proc.stdout}\nstderr={ex_proc.stderr}")
            ex = json.loads(ex_proc.stdout)
            self.assertEqual(ex["status"], "completed")
            self.assertEqual(ex["exit_code"], 0)
            self.assertIn("mode=e2e msg=hello world", ex["log"]["text"])
            self.assertIn("tracked-result-input", ex["log"]["text"])
            run_id = ex["run_id"]

            # 3. Pull artifacts + registered shared-dir file atomically (staged from pinned descriptors).
            pulled = json.loads(subprocess.run(
                [*cli, "pull", run_id, str(pull_dest), "--path", "shared-cache/item.txt"],
                env=env, capture_output=True, text=True, check=True,
            ).stdout)
            self.assertEqual(pulled["status"], "completed")
            self.assertTrue((pull_dest / "artifacts" / "report.txt").is_file())
            self.assertFalse((pull_dest / "artifacts" / "debug.trace").exists())
            self.assertEqual((pull_dest / "workspace" / "shared-cache" / "item.txt").read_text().strip(), "shared-val")
            # Remote pull stage must be cleaned up automatically.
            stages_dir = remote_root / "pull-stages"
            self.assertEqual(list(stages_dir.iterdir()) if stages_dir.is_dir() else [], [])

            # Non-force pull MUST refuse an existing empty directory (RENAME_NOREPLACE protection).
            empty_dest = t / "already_exists_empty"
            empty_dest.mkdir()
            refused = subprocess.run(
                [*cli, "pull", run_id, str(empty_dest)],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(refused.returncode, 1)
            self.assertIn("already exists", refused.stderr)

            # --force pull atomically exchanges merged tree preserving pre-existing files AND directory mode.
            (empty_dest / "preexisting.txt").write_text("keep-me\n")
            empty_dest.chmod(0o755)
            subprocess.run(
                [*cli, "pull", run_id, str(empty_dest), "--force"],
                env=env, capture_output=True, text=True, check=True,
            )
            self.assertEqual((empty_dest / "preexisting.txt").read_text(), "keep-me\n")
            self.assertEqual(empty_dest.stat().st_mode & 0o777, 0o755)
            self.assertTrue((empty_dest / "artifacts" / "report.txt").is_file())

            # --force pull MUST refuse writing through an existing symlink inside destination.
            outside_target = t / "outside_dir"
            outside_target.mkdir()
            sym_dest = t / "dest_with_sym"
            sym_dest.mkdir()
            (sym_dest / "artifacts").symlink_to(outside_target)
            sym_refused = subprocess.run(
                [*cli, "pull", run_id, str(sym_dest), "--force"],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(sym_refused.returncode, 1)
            self.assertIn("existing symlink", sym_refused.stderr)
            self.assertEqual(list(outside_target.iterdir()), [])

            # 4. Prune dry-run with days=0 sees our completed run cleanly.
            pr = json.loads(subprocess.run(
                [*cli, "prune", "--days", "0"],
                env=env, capture_output=True, text=True, check=True,
            ).stdout)
            self.assertEqual(pr["runs_count"], 1)
            self.assertEqual(pr["skipped_unmanaged_count"], 0)

            # 5. MCP rejects empty gpus=[] cleanly (caller must use no_gpu=true / clear_gpus=true).
            import remote_run_mcp as mcp
            with self.assertRaisesRegex(ValueError, "non-empty"):
                mcp.translate_tool("remote_exec", {"argv": ["true"], "no_sync": True, "gpus": []})
            with self.assertRaisesRegex(ValueError, "non-empty"):
                mcp.translate_tool("remote_bind", {"repo": str(repo), "gpus": []})


if __name__ == "__main__":
    unittest.main()
