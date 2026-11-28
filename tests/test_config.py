"""Unit tests for v2 three-layer configuration, bindings, hierarchy, and provenance."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from remote_run_local import config as cfg
from remote_run_local import service as svc
from remote_run_local import transport


def git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TEMPLATE_DIR": "",
    }
    res = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q", "-b", "main")
    (repo / "main.rs").write_text("fn main() {}\n")
    git(repo, "add", "main.rs")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "init")


class ConfigAndBindingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.xdg = self.tmp / "xdg"
        self.xdg.mkdir()
        self.enterContext(mock.patch.dict(os.environ, {
            "XDG_CONFIG_HOME": str(self.xdg),
            "REMOTE_RUN_SSH_CONTROL": "0",
        }))
        self.hosts_file = self.xdg / "remote-run" / "hosts.toml"
        self.hosts_file.parent.mkdir(parents=True)

    def test_v1_rejected_and_minimal_cpu_host_accepted(self):
        self.hosts_file.write_text('schema_version = 1\n[hosts.box]\nssh = "box"\n')
        with self.assertRaisesRegex(ValueError, "schema_version must be integer 2.*Migrate"):
            cfg.load_hosts_config(self.hosts_file)

        # Legacy v1 .remote-run.toml with v1-only keys MUST report schema_version v1 migration hint FIRST (not Unknown keys).
        v1_repo = self.tmp / "v1-proj"
        v1_repo.mkdir()
        (v1_repo / ".remote-run.toml").write_text('schema_version = 1\nhost = "box"\n[share]\nhost = ["d=x"]\n')
        with self.assertRaisesRegex(ValueError, "schema_version must be integer 2.*remote-run bind"):
            cfg.load_project_config(v1_repo)

        # Minimal 3-line CPU host with only ssh configured.
        self.hosts_file.write_text('schema_version = 2\n[hosts.cpu-box]\nssh = "user@cpu-box"\n')
        default_host, hosts = cfg.load_hosts_config(self.hosts_file)
        self.assertIsNone(default_host)
        self.assertIn("cpu-box", hosts)
        self.assertIsNone(hosts["cpu-box"]["gpu"])
        self.assertEqual(hosts["cpu-box"]["root"], "~/.local/share/remote-run-v2")
        selected, origin = cfg.select_host_from_catalog(default_host, hosts)
        self.assertEqual(selected["name"], "cpu-box")
        self.assertEqual(origin, "sole_host")

        # Unknown keys in hosts.toml or project config fail fast.
        self.hosts_file.write_text('schema_version = 2\n[hosts.box]\nssh = "box"\ntoolsets = {}\n')
        with self.assertRaisesRegex(ValueError, "Unknown key.*toolsets"):
            cfg.load_hosts_config(self.hosts_file)

    def test_four_level_hierarchy_repo_and_checkout_bindings_and_worktree_inheritance(self):
        self.hosts_file.write_text(
            'schema_version = 2\n'
            'default_host = "cpu-box"\n'
            '[hosts.cpu-box]\n'
            'ssh = "cpu-ssh"\n'
            'path = ["/opt/cpu/bin"]\n'
            '[hosts.cpu-box.env]\n'
            'SHARED_KEY = "from-host"\n'
            'HOST_ONLY = "1"\n'
            '[hosts.gpu-box]\n'
            'ssh = "gpu-ssh"\n'
            'root = "/srv/rr-v2"\n'
            'path = ["/opt/gpu-global/bin"]\n'
            '[hosts.gpu-box.env]\n'
            'SHARED_KEY = "from-gpu-host"\n'
            '[hosts.gpu-box.gpu]\n'
            'devices = [0, 1]\n'
        )
        main_repo = self.tmp / "repo-main"
        init_git_repo(main_repo)
        (main_repo / ".remote-run.toml").write_text(
            'schema_version = 2\n'
            'project = "demo-proj"\n'
            'path = ["build/bin", "node_modules/.bin"]\n'
            '[env]\n'
            'SHARED_KEY = "from-project"\n'
            'PROJ_KEY = "p"\n'
            '[workspace]\n'
            'shared = ["tool-cache"]\n'
            '[sync]\n'
            'exclude = ["docs/generated/"]\n'
            'preserve = ["build", "target"]\n'
            'max_untracked_mib = 128\n'
            '[pull]\n'
            'max_file_mib = 5\n'
            'exclude = ["**/*.trace"]\n'
        )
        # Commit .remote-run.toml so linked worktrees receive the tracked project config!
        git(main_repo, "add", ".remote-run.toml")
        git(main_repo, "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "add config")
        # Create a linked Git worktree.
        wt_dir = self.tmp / "repo-wt-a"
        git(main_repo, "worktree", "add", "-q", "-b", "feat-a", str(wt_dir))

        # Bind repo scope -> ALL worktrees inherit host, gpus, note, path, env.
        svc.op_bind(
            main_repo, scope="repo", host="gpu-box", gpus=[0],
            note="Prebuilt toolchain ready",
            paths=["/opt/proj/bin"], env_vars={"SHARED_KEY": "from-repo-bind", "DATA_DIR": "/srv/data"},
            config=self.hosts_file,
        )
        # Bind checkout scope on linked worktree ONLY -> overrides workspace and gpus.
        svc.op_bind(
            wt_dir, scope="checkout", host="gpu-box", workspace="exp-a", gpus=[1],
            paths=["/opt/exp-a/bin"], env_vars={"SHARED_KEY": "from-co-bind"},
            config=self.hosts_file,
        )

        # Repo scope must reject workspace!
        with self.assertRaisesRegex(ValueError, "checkout-scoped"):
            svc.op_bind(main_repo, scope="repo", host="gpu-box", workspace="bad", config=self.hosts_file)

        # Resolve main checkout: workspace=default, gpus=[0], path order: cli > repo_bind > project > host
        res_main = cfg.resolve_effective_config(
            self.hosts_file, repo_dir=main_repo, cli_paths=["/opt/cli/bin"], cli_env={"CLI_KEY": "c"},
        )
        self.assertEqual(res_main["host"]["name"], "gpu-box")
        self.assertEqual(res_main["workspace"]["project_id"], "demo-proj")
        self.assertEqual(res_main["workspace"]["workspace_id"], "default")
        self.assertEqual(res_main["workspace"]["workdir"], "/srv/rr-v2/projects/demo-proj/workspaces/default")
        self.assertEqual(res_main["workspace"]["shared_dir"], "/srv/rr-v2/projects/demo-proj/shared")
        self.assertEqual(res_main["gpus"], [0])
        self.assertEqual(res_main["note"], "Prebuilt toolchain ready")
        self.assertEqual(
            [p["entry"] for p in res_main["path_layers"]],
            ["/opt/cli/bin", "/opt/proj/bin", "build/bin", "node_modules/.bin", "/opt/gpu-global/bin"],
        )
        self.assertEqual(res_main["config_env"]["SHARED_KEY"], "from-repo-bind")
        self.assertEqual(res_main["config_env"]["DATA_DIR"], "/srv/data")

        # Resolve linked worktree: inherits repo binding + overlays checkout binding.
        res_wt = cfg.resolve_effective_config(self.hosts_file, repo_dir=wt_dir)
        self.assertEqual(res_wt["workspace"]["workspace_id"], "exp-a")
        self.assertEqual(res_wt["gpus"], [1])
        self.assertEqual(res_wt["note"], "Prebuilt toolchain ready")
        self.assertEqual(res_wt["config_env"]["SHARED_KEY"], "from-co-bind")
        self.assertEqual(res_wt["config_env"]["DATA_DIR"], "/srv/data")
        self.assertEqual(
            [p["entry"] for p in res_wt["path_layers"]],
            ["/opt/exp-a/bin", "/opt/proj/bin", "build/bin", "node_modules/.bin", "/opt/gpu-global/bin"],
        )

        # Switching --host cpu-box MUST skip gpu-box host-bound fields (gpus/note/path/env),
        # while checkout workspace ID remains portable across hosts!
        res_other = cfg.resolve_effective_config(self.hosts_file, repo_dir=wt_dir, cli_host="cpu-box")
        self.assertEqual(res_other["host"]["name"], "cpu-box")
        self.assertEqual(res_other["gpus"], [])
        self.assertEqual(res_other["note"], "")
        self.assertNotIn("DATA_DIR", res_other["config_env"])
        self.assertEqual(res_other["config_env"]["SHARED_KEY"], "from-project")
        self.assertEqual(len(res_other["bindings"]["skipped"]), 2)
        self.assertEqual(res_other["workspace"]["workspace_id"], "exp-a")

        # Rebinding checkout to a different host resets old host-bound fields (gpus/path/env/note)
        # while preserving portable checkout workspace ID.
        rebound = svc.op_bind(wt_dir, scope="checkout", host="cpu-box", config=self.hosts_file)
        self.assertEqual(rebound["host"], "cpu-box")
        self.assertEqual(rebound["workspace"], "exp-a")
        self.assertIsNone(rebound["gpus"])
        self.assertEqual(rebound["path"], [])
        self.assertEqual(rebound["env_keys"], [])
        self.assertEqual(rebound["note"], "")

        # Changing host endpoint in hosts.toml invalidates old repo binding's host-bound fields!
        self.hosts_file.write_text(
            'schema_version = 2\n'
            'default_host = "gpu-box"\n'
            '[hosts.gpu-box]\n'
            'ssh = "different-ssh-endpoint"\n'
            'root = "/srv/rr-v2"\n'
            '[hosts.gpu-box.gpu]\n'
            'devices = [0, 1]\n'
        )
        res_changed_ep = cfg.resolve_effective_config(self.hosts_file, repo_dir=main_repo)
        self.assertNotIn("DATA_DIR", res_changed_ep["config_env"])
        self.assertEqual(res_changed_ep["gpus"], [])
        self.assertEqual(len(res_changed_ep["bindings"]["skipped"]), 1)
        self.assertIn("endpoint", res_changed_ep["bindings"]["skipped"][0]["reason"])

        # Redacted config view hides env values while exposing origins, note, and path layers.
        view = cfg.redact_config_view(res_wt)
        self.assertEqual(view["environment"]["env_keys"]["SHARED_KEY"]["value"], "<redacted>")
        self.assertEqual(view["environment"]["env_keys"]["SHARED_KEY"]["origin"], "checkout_binding")
        self.assertEqual(view["environment"]["note"], "Prebuilt toolchain ready")

    def test_ssh_auth_sock_missing_hint_on_255(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            hint = transport._ssh_auth_hint(["ssh", "box"], 255, "Permission denied (publickey,password).")
            self.assertIn("SSH_AUTH_SOCK", hint)
            self.assertIn("env_vars", hint)
        with mock.patch.dict(os.environ, {"SSH_AUTH_SOCK": str(self.tmp / "missing.sock")}, clear=True):
            hint2 = transport._ssh_auth_hint(["ssh", "box"], 255, "Permission denied (publickey).")
            self.assertIn("missing or unreachable socket", hint2)
        self.assertEqual(transport._ssh_auth_hint(["rsync", "a", "b"], 255, "Permission denied (publickey)."), "")

    def test_load_binding_requires_host_endpoint_for_host_bound_fields(self):
        bfile = cfg.binding_path("repo", "manual-check")
        bfile.parent.mkdir(parents=True, exist_ok=True)
        bfile.write_text('schema_version = 2\nhost = "gpu-box"\ngpus = [0]\n')
        with self.assertRaisesRegex(ValueError, "host_endpoint"):
            cfg.load_binding("repo", "manual-check")
        # Even empty host-specific keys (path = [], env = {}, note = "") require host + host_endpoint.
        bfile.write_text('schema_version = 2\npath = []\n')
        with self.assertRaisesRegex(ValueError, "host_endpoint"):
            cfg.load_binding("repo", "manual-check")

    def test_op_exec_rejects_workspace_with_no_project(self):
        with self.assertRaisesRegex(ValueError, "--workspace cannot be used with --no-project"):
            svc.op_exec(
                repo=None,
                no_project=True,
                no_sync=True,
                workspace="exp-a",
                argv=["true"],
                config=self.hosts_file,
            )


if __name__ == "__main__":
    unittest.main()
