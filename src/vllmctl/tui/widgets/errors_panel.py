"""Centered error card shown when the selected model's YAML is invalid."""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class ErrorsPanel(Vertical):
    """Vertically-centered card with the error message, file path and recovery hint."""

    DEFAULT_CSS = """
    ErrorsPanel {
        height: 1fr;
        border: round $error;
        padding: 0;
    }
    ErrorsPanel:focus-within {
        border: round $primary;
    }
    ErrorsPanel > #error-card {
        align: center middle;
        height: 1fr;
        padding: 1 2;
    }
    ErrorsPanel #error-body {
        width: 100%;
        text-align: center;
        color: $error;
        text-style: bold;
        height: auto;
    }
    ErrorsPanel #error-path {
        width: 100%;
        text-align: center;
        color: $text-muted;
        height: auto;
        margin-top: 1;
    }
    ErrorsPanel #error-hint {
        width: 100%;
        text-align: center;
        color: $text-muted;
        height: auto;
        margin-top: 2;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._body = Static("", id="error-body")
        self._path = Static("", id="error-path")
        self._hint = Static("", id="error-hint")
        self.border_title = "Errors"

    def compose(self) -> ComposeResult:
        with Vertical(id="error-card"):
            yield self._body
            yield self._path
            yield self._hint

    def show_yaml_error(
        self,
        model_name: str,
        error: str,
        yaml_path: Path | None = None,
    ) -> None:
        self.border_title = f"Invalid YAML · {model_name}"
        self._body.update(escape(error))
        if yaml_path is not None:
            self._path.update(f"in {escape(str(yaml_path))}")
        else:
            self._path.update("")
        self._hint.update(
            "press 'e' to edit and fix\n"
            "if a process is still running, 'S' (stop) still works"
        )
