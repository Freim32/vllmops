"""Tests for the editor resolver used by the TUI 'e' binding."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from vllmctl.tui.app import _resolve_editor


@pytest.fixture
def fake_editors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Plant fake editor binaries in a temp dir and put it first on PATH."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    for name in ("nano", "vim", "vi", "micro", "emacs", "myeditor"):
        ext = ".exe" if sys.platform == "win32" else ""
        path = bin_dir / f"{name}{ext}"
        path.write_text("#!/bin/sh\nexit 0\n")
        if sys.platform != "win32":
            path.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    return bin_dir


def test_resolve_editor_prefers_project_config_over_env(fake_editors: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del fake_editors
    monkeypatch.setenv("EDITOR", "nano")
    assert _resolve_editor("vim") == "vim"


def test_resolve_editor_uses_visual_before_editor(fake_editors: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del fake_editors
    monkeypatch.setenv("VISUAL", "micro")
    monkeypatch.setenv("EDITOR", "nano")
    assert _resolve_editor(None) == "micro"


def test_resolve_editor_falls_back_to_first_on_path(fake_editors: Path) -> None:
    """No project config, no env vars, pick the first sensible default."""
    del fake_editors
    assert _resolve_editor(None) == "nano"


def test_resolve_editor_returns_none_when_nothing_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert _resolve_editor(None) is None


def test_resolve_editor_skips_unreachable_configured_value(fake_editors: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Project config points to a non-existent binary; fall through."""
    del fake_editors, monkeypatch
    assert _resolve_editor("nonexistent-editor-xyz") == "nano"


def test_resolve_editor_preserves_args(fake_editors: Path) -> None:
    """A configured value with args is kept verbatim if its first token resolves."""
    del fake_editors
    if shutil.which("myeditor") is None:
        pytest.skip("fake editor missing")
    assert _resolve_editor("myeditor --wait --no-banner") == "myeditor --wait --no-banner"
