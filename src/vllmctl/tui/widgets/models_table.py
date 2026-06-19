"""Sidebar table listing the project's models."""

from __future__ import annotations

from collections.abc import Iterable

from rich.text import Text
from textual.widgets import DataTable

from vllmctl.service import CatalogEntry


class ModelsTable(DataTable):
    """Sidebar list of models with state and metrics port."""

    DEFAULT_CSS = """
    ModelsTable {
        height: 1fr;
        border: round $accent;
    }
    ModelsTable:focus {
        border: round $primary;
    }
    ModelsTable > .datatable--cursor {
        background: cyan;
        color: black;
        text-style: bold;
    }
    ModelsTable > .datatable--header {
        background: $boost;
        color: $text-muted;
    }
    """

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.border_title = "Models"
        self.add_column("State", width=11)
        self.add_column("Name", width=18)
        self.add_column("Port", width=7)

    def render_entries(self, entries: Iterable[CatalogEntry]) -> None:
        entries = list(entries)
        previous_key = (
            self.coordinate_to_cell_key(self.cursor_coordinate).row_key
            if self.row_count
            else None
        )
        previous_name = previous_key.value if previous_key is not None else None

        self.clear()
        for index, entry in enumerate(entries):
            if entry.is_broken:
                state = Text("✗ invalid", style="bold red")
                port = "-"
            else:
                status = entry.status
                assert status is not None
                if status.running:
                    state = Text("● running", style="bold green")
                elif status.stale_pid_file:
                    state = Text("● stale", style="bold yellow")
                else:
                    state = Text("○ stopped", style="bright_black")
                port = str(status.metrics_port) if status.metrics_port else "-"
            self.add_row(state, entry.name, port, key=entry.name)
            if previous_name == entry.name:
                self.move_cursor(row=index)
        self.refresh()

    @property
    def selected_model_name(self) -> str | None:
        if self.row_count == 0:
            return None
        key = self.coordinate_to_cell_key(self.cursor_coordinate).row_key
        return key.value if key is not None else None
