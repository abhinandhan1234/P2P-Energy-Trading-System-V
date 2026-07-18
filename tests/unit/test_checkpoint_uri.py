"""Unit tests for resolve_checkpoint_uri in Module 9 evaluate.py.

Covers:
  - relative filesystem path
  - absolute Windows-style path
  - absolute POSIX path
  - already-valid file:// URI (passthrough)
  - missing checkpoint directory → FileNotFoundError
  - Windows path formats on the current platform

Design reference: docs/module_9_evaluation_framework.md §1
"""

from __future__ import annotations

# standard library
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# third party
import pytest

# local
from p2p_energy_trading.evaluation.evaluate import resolve_checkpoint_uri

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"


def _is_file_uri(s: str) -> bool:
    """Return True when *s* is a well-formed file:// URI."""
    parsed = urlparse(s)
    return parsed.scheme == "file" and bool(parsed.path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveCheckpointUriRelativePath:
    """resolve_checkpoint_uri with a relative path."""

    def test_relative_path_returns_file_uri(self, tmp_path: Path) -> None:
        """A relative path that resolves to an existing dir becomes a file URI."""
        # Create the checkpoint directory under tmp_path so it exists.
        ckpt_dir = tmp_path / "checkpoint_000001"
        ckpt_dir.mkdir()

        # Change cwd so the relative path resolves correctly.
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = resolve_checkpoint_uri("checkpoint_000001")
        finally:
            os.chdir(old_cwd)

        assert _is_file_uri(result), f"Expected file URI, got: {result!r}"
        assert result.startswith("file:///"), (
            f"Expected file:/// prefix, got: {result!r}"
        )
        # The resolved path must contain the directory name.
        assert "checkpoint_000001" in result


class TestResolveCheckpointUriAbsolutePath:
    """resolve_checkpoint_uri with absolute filesystem paths."""

    def test_absolute_posix_path_returns_file_uri(self, tmp_path: Path) -> None:
        """An absolute POSIX path to an existing dir becomes a file URI."""
        ckpt_dir = tmp_path / "checkpoint_abs"
        ckpt_dir.mkdir()

        result = resolve_checkpoint_uri(str(ckpt_dir))

        assert _is_file_uri(result), f"Expected file URI, got: {result!r}"
        assert "checkpoint_abs" in result

    @pytest.mark.skipif(not _IS_WINDOWS, reason="Windows-style paths only on Windows")
    def test_absolute_windows_path_returns_file_uri(self, tmp_path: Path) -> None:
        """An absolute Windows path (drive letter) to an existing dir → file URI."""
        ckpt_dir = tmp_path / "checkpoint_win"
        ckpt_dir.mkdir()

        # Use the native Windows string representation (e.g. C:\...\checkpoint_win)
        win_path = str(ckpt_dir)
        result = resolve_checkpoint_uri(win_path)

        assert _is_file_uri(result), f"Expected file URI, got: {result!r}"
        assert "checkpoint_win" in result


class TestResolveCheckpointUriAlreadyUri:
    """resolve_checkpoint_uri with an already-valid file:// URI."""

    def test_valid_file_uri_returned_unchanged(self, tmp_path: Path) -> None:
        """An existing file:// URI must be returned as-is (no double conversion)."""
        ckpt_dir = tmp_path / "checkpoint_uri"
        ckpt_dir.mkdir()

        uri = ckpt_dir.as_uri()
        assert uri.startswith("file:///")

        result = resolve_checkpoint_uri(uri)

        assert result == uri, (
            f"Expected URI returned unchanged.\n  input:  {uri!r}\n  output: {result!r}"
        )

    def test_valid_file_uri_still_validated(self, tmp_path: Path) -> None:
        """A file:// URI pointing to a non-existent path raises FileNotFoundError."""
        missing_uri = (tmp_path / "does_not_exist").as_uri()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            resolve_checkpoint_uri(missing_uri)


class TestResolveCheckpointUriMissingDirectory:
    """resolve_checkpoint_uri raises FileNotFoundError for missing checkpoints."""

    def test_missing_relative_path_raises(self, tmp_path: Path) -> None:
        """A relative path to a non-existent directory raises FileNotFoundError."""
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
                resolve_checkpoint_uri("this_does_not_exist")
        finally:
            os.chdir(old_cwd)

    def test_missing_absolute_path_raises(self, tmp_path: Path) -> None:
        """An absolute path to a non-existent directory raises FileNotFoundError."""
        missing = tmp_path / "nonexistent_checkpoint"
        assert not missing.exists()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            resolve_checkpoint_uri(str(missing))

    def test_error_message_contains_resolved_path(self, tmp_path: Path) -> None:
        """FileNotFoundError message must include the resolved absolute path."""
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with pytest.raises(FileNotFoundError) as exc_info:
                resolve_checkpoint_uri("missing_ckpt")
        finally:
            os.chdir(old_cwd)

        # The message should reference "Checkpoint not found" per the spec.
        assert "Checkpoint not found" in str(exc_info.value)
        # And it should contain the directory name somewhere.
        assert "missing_ckpt" in str(exc_info.value)


class TestResolveCheckpointUriOutputFormat:
    """Verify the output URI is well-formed across scenarios."""

    def test_output_is_parseable_uri(self, tmp_path: Path) -> None:
        """Output must be parseable by urllib.parse.urlparse with scheme='file'."""
        ckpt_dir = tmp_path / "checkpoint_format"
        ckpt_dir.mkdir()

        result = resolve_checkpoint_uri(str(ckpt_dir))
        parsed = urlparse(result)

        assert parsed.scheme == "file"
        assert parsed.netloc == "" or parsed.netloc == "localhost"
        assert parsed.path  # non-empty path component

    def test_output_is_idempotent_when_fed_back(self, tmp_path: Path) -> None:
        """Passing the output URI back into resolve_checkpoint_uri is a no-op."""
        ckpt_dir = tmp_path / "checkpoint_idempotent"
        ckpt_dir.mkdir()

        first_pass = resolve_checkpoint_uri(str(ckpt_dir))
        second_pass = resolve_checkpoint_uri(first_pass)

        assert first_pass == second_pass, (
            "resolve_checkpoint_uri should be idempotent for file:// URIs.\n"
            f"  first:  {first_pass!r}\n"
            f"  second: {second_pass!r}"
        )

    def test_spaces_in_path_are_percent_encoded(self, tmp_path: Path) -> None:
        """Spaces in directory names must be percent-encoded in the resulting URI."""
        ckpt_dir = tmp_path / "checkpoint with spaces"
        ckpt_dir.mkdir()

        result = resolve_checkpoint_uri(str(ckpt_dir))

        assert " " not in result, f"Unencoded space found in URI: {result!r}"
        assert "%20" in result or "+" in result, (
            f"Expected percent-encoded space in URI: {result!r}"
        )
