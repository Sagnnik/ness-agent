from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test")

import ness_agent.workspace.git_context as git_context


class AutoGitSnapshotTests(unittest.TestCase):
    def test_returns_true_on_successful_commit(self) -> None:
        with mock.patch.object(git_context, "_git", side_effect=["(ok)", "committed"]):
            self.assertTrue(git_context.auto_git_snapshot())

    def test_returns_true_when_nothing_to_commit(self) -> None:
        with mock.patch.object(
            git_context,
            "_git",
            side_effect=["(ok)", "Error: git commit failed\nnothing to commit"],
        ):
            self.assertTrue(git_context.auto_git_snapshot())

    def test_returns_false_when_add_fails(self) -> None:
        with mock.patch.object(git_context, "_git", return_value="Error: git add failed"):
            self.assertFalse(git_context.auto_git_snapshot())

    def test_uses_no_verify_on_commit(self) -> None:
        with mock.patch.object(git_context, "_git", side_effect=["(ok)", "(ok)"]) as mock_git:
            git_context.auto_git_snapshot("agent: test")
        mock_git.assert_any_call(["commit", "-m", "agent: test", "--no-verify"], timeout=60)


class GitWorktreeSummaryTests(unittest.TestCase):
    def test_clean_tree(self) -> None:
        with mock.patch.object(git_context, "_git", side_effect=["main", "(ok)"]):
            self.assertEqual(git_context.git_worktree_summary(), "branch: main; working tree clean")

    def test_dirty_tree_truncates_paths(self) -> None:
        porcelain = "\n".join(f" M file{i}.py" for i in range(8))
        with mock.patch.object(git_context, "_git", side_effect=["dev", porcelain]):
            summary = git_context.git_worktree_summary()
        self.assertIn("branch: dev", summary)
        self.assertIn("8 changed file(s)", summary)
        self.assertIn("file0.py", summary)
        self.assertIn("(+3 more)", summary)

    def test_returns_empty_when_not_in_repo(self) -> None:
        with mock.patch.object(
            git_context,
            "_git",
            return_value="Error: git branch --show-current failed with exit 128",
        ):
            self.assertEqual(git_context.git_worktree_summary(), "")

    def test_detached_head(self) -> None:
        with mock.patch.object(git_context, "_git", side_effect=["(ok)", "(ok)"]):
            self.assertEqual(git_context.git_worktree_summary(), "branch: (detached); working tree clean")


if __name__ == "__main__":
    unittest.main()
