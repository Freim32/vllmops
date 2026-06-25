"""Tail viewer for a model's log file with a placeholder for empty state."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher, RichLog, Static

from vllmctl.tui.widgets.highlighter import VllmLogHighlighter, strip_process_prefix

_LOG_VIEW = "log-content"
_PLACEHOLDER_VIEW = "log-placeholder"


class LogViewer(Vertical):
    """Tail of a model's log file. Polls the offset on each tick."""

    DEFAULT_CSS = """
    LogViewer {
        height: 1fr;
        border: round $accent;
        padding: 0 1;
    }
    LogViewer:focus-within {
        border: round $primary;
    }
    LogViewer > ContentSwitcher {
        height: 1fr;
    }
    LogViewer #log-placeholder {
        align: center middle;
        height: 1fr;
        width: 100%;
    }
    LogViewer #log-placeholder-body {
        width: 100%;
        text-align: center;
        color: $text-muted;
        height: auto;
    }
    """

    def __init__(self, *, max_lines: int = 1000) -> None:
        super().__init__()
        self._log = RichLog(
            highlight=True,
            markup=False,
            max_lines=max_lines,
            wrap=True,
            id=_LOG_VIEW,
        )
        self._log.highlighter = VllmLogHighlighter()
        self._placeholder_body = Static("[dim]no model selected[/dim]", id="log-placeholder-body")
        self._path: Path | None = None
        self._offset = 0
        self.border_title = "Logs"

    def compose(self) -> ComposeResult:
        with ContentSwitcher(initial=_PLACEHOLDER_VIEW):
            yield self._log
            with Vertical(id=_PLACEHOLDER_VIEW):
                yield self._placeholder_body

    def on_mount(self) -> None:
        # Initial state is the placeholder with "no model selected" baked into
        # the Static at construction; `attach()` will swap in a per-model
        # message once the cursor lands somewhere.
        pass

    def attach(self, path: Path | None, label: str) -> None:
        """Switch to a new log file. Resets buffer and seeks to end."""
        self.border_title = f"Logs · {label}"
        self._log.clear()
        self._path = path
        if path is None:
            self._offset = 0
            self._show_placeholder("[dim]no model selected[/dim]")
            return
        # A zero-byte file is still treated as "no log yet"; `poll()` will swap
        # to the log view once the first byte lands.
        size = path.stat().st_size if path.is_file() else 0
        if size == 0:
            self._offset = 0
            self._show_placeholder("[bold]no log file yet[/bold]\n\n[dim]press 's' to start the model[/dim]")
            return
        self._offset = size
        tail = _read_last_bytes(path, max_bytes=8192)
        for line in tail.splitlines():
            self._log.write(strip_process_prefix(line))
        self._offset = path.stat().st_size
        self._show_log()

    def poll(self) -> None:
        """Read new bytes appended since last poll."""
        if self._path is None or not self._path.is_file():
            return
        try:
            size = self._path.stat().st_size
        except OSError:
            return
        if size == self._offset:
            return
        if size < self._offset:
            self._offset = 0
            self._log.clear()
        try:
            with self._path.open("rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read(size - self._offset)
                self._offset = handle.tell()
        except OSError:
            return
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines():
            self._log.write(strip_process_prefix(line))
        self._show_log()

    def _show_placeholder(self, message: str) -> None:
        self._switcher.current = _PLACEHOLDER_VIEW
        self._placeholder_body.update(message)

    def _show_log(self) -> None:
        self._switcher.current = _LOG_VIEW

    @property
    def _switcher(self) -> ContentSwitcher:
        return self.query_one(ContentSwitcher)


def _read_last_bytes(path: Path, *, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    read = min(size, max_bytes)
    try:
        with path.open("rb") as handle:
            handle.seek(size - read)
            data = handle.read(read)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")
