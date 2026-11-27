"""Unit tests for v2 single-manifest transfer, budget guard, ghost cleanup, and pull."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from remote_run_local import transfer


def git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TEMPLATE_DIR": "",
    }
    res = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True)
    return res.stdout.strip()


class TransferManifestTests(unittest.TestCase):
    def test_tracked_results_and_outputs_are_never_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            git(repo, "init", "-q", "-b", "main")
            # Even if .gitignore mentions results/, tracked files in results/ and outputs/ MUST be included!
            (repo / ".gitignore").write_text("results/\n*.log\n")
            (repo / "results").mkdir()
            (repo / "results" / "golden.json").write_text('{"ok": true}\n')
            (repo / "outputs").mkdir()
            (repo / "outputs" / "fixture.txt").write_text("fixture\n")
            (repo / "results" / "ignored_untracked.tmp").write_text("skip me\n")
            (repo / "untracked.log").write_text("skip log\n")
            (repo / "new_src.rs").write_text("pub fn f() {}\n")
            (repo / ".env").write_text("SECRET=1\n")
            git(repo, "add", "-f", ".gitignore", "results/golden.json", "outputs/fixture.txt")
            git(repo, "-c", "user.name=T", "-c", "user.email=t@t", "commit", "-qm", "init")

            m = transfer.build_sync_manifest(
                repo, in_git=True,
                sync_policy={"exclude": [], "preserve": [], "shared": [], "max_untracked_mib": 256},
            )
            self.assertIn("results/golden.json", m["files"])
            self.assertIn("outputs/fixture.txt", m["files"])
            self.assertIn("new_src.rs", m["files"])
            self.assertNotIn("results/ignored_untracked.tmp", m["files"])
            self.assertNotIn("untracked.log", m["files"])
            self.assertNotIn(".env", m["files"])

    def test_untracked_budget_blocks_large_plain_dir_cache_without_language_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain-proj"
            plain.mkdir()
            (plain / "main.go").write_text("package main\n")
            cache = plain / "huge_cache"
            cache.mkdir()
            (cache / "blob.dat").write_bytes(b"0" * (2 * 1024 * 1024))

            # Budget of 1 MiB blocks transfer with dominant directory reported.
            with self.assertRaisesRegex(ValueError, r"exceeds max_untracked_mib=1 MiB.*huge_cache"):
                transfer.build_sync_manifest(
                    plain, in_git=False,
                    sync_policy={"exclude": [], "preserve": [], "shared": [], "max_untracked_mib": 1},
                )

            # Adding huge_cache/ to exclude allows transfer cleanly.
            m = transfer.build_sync_manifest(
                plain, in_git=False,
                sync_policy={"exclude": ["huge_cache/"], "preserve": [], "shared": [], "max_untracked_mib": 1},
            )
            self.assertEqual(m["files"], ["main.go"])

    def test_tracked_conflict_with_preserve_or_shared_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            git(repo, "init", "-q", "-b", "main")
            (repo / "build").mkdir()
            (repo / "build" / "file.txt").write_text("tracked")
            git(repo, "add", "build/file.txt")
            git(repo, "-c", "user.name=T", "-c", "user.email=t@t", "commit", "-qm", "init")

            with self.assertRaisesRegex(ValueError, "conflicts with reserved remote directory"):
                transfer.build_sync_manifest(
                    repo, in_git=True,
                    sync_policy={"exclude": [], "preserve": ["build"], "shared": [], "max_untracked_mib": 256},
                )

    def test_ghost_cleanup_never_deletes_newly_excluded_existing_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / "kept.py").write_text("1")
            (repo / "newly_excluded.txt").write_text("still local")
            # deleted.py is absent locally -> eligible ghost candidate.
            cands = transfer.prune_ghost_candidates(
                repo,
                manifest_files={"kept.py"},
                previous_owned=["kept.py", "newly_excluded.txt", "deleted.py", "build/gen.o"],
                sync_policy={"exclude": ["*.txt"], "preserve": ["build"], "shared": []},
            )
            self.assertEqual(cands, ["deleted.py"])

    def test_escaping_symlink_rejected_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "outside.txt"
            outside.write_text("nope")
            plain = base / "proj"
            plain.mkdir()
            (plain / "bad_link").symlink_to("../outside.txt")
            with self.assertRaisesRegex(ValueError, "escaping symlink"):
                transfer.build_sync_manifest(
                    plain, in_git=False,
                    sync_policy={"exclude": [], "preserve": [], "shared": [], "max_untracked_mib": 256},
                )

    def test_in_repo_xdg_bindings_excluded_and_leading_space_git_paths_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            git(repo, "init", "-q", "-b", "main")
            # File with leading space must NOT be stripped by git_query!
            (repo / " leading.txt").write_text("hi\n")
            git(repo, "add", " leading.txt")
            git(repo, "-c", "user.name=T", "-c", "user.email=t@t", "commit", "-qm", "init")
            # Private XDG config dir placed inside repo must NEVER be uploaded.
            xdg = repo / ".local-xdg"
            bind_file = xdg / "remote-run" / "bindings" / "repo-abc.toml"
            bind_file.parent.mkdir(parents=True)
            bind_file.write_text('schema_version = 2\nhost = "box"\n')
            old = os.environ.get("XDG_CONFIG_HOME")
            os.environ["XDG_CONFIG_HOME"] = str(xdg)
            try:
                m = transfer.build_sync_manifest(
                    repo, in_git=True,
                    sync_policy={"exclude": [], "preserve": [], "shared": [], "max_untracked_mib": 256},
                )
            finally:
                if old is None:
                    os.environ.pop("XDG_CONFIG_HOME", None)
                else:
                    os.environ["XDG_CONFIG_HOME"] = old
            self.assertIn(" leading.txt", m["files"])
            self.assertFalse(any("bindings" in f for f in m["files"]))


if __name__ == "__main__":
    unittest.main()
