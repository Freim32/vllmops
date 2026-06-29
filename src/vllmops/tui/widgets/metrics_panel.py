"""vLLM-side headline metrics panel."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from vllmops.metrics import ModelMetricsSnapshot
from vllmops.tui.widgets._format import (
    format_int,
    format_percent,
    format_rate,
    format_seconds,
)


class MetricsPanel(Vertical):
    """vLLM-side metrics for the selected model.

    Renders a single `rich.Table` with three groups separated by empty rows:
    rate (throughput, latency), queue (running, waiting), cache (KV cache).
    """

    DEFAULT_CSS = """
    MetricsPanel {
        height: 1fr;
        border: round $accent;
        padding: 1 2;
    }
    MetricsPanel:focus-within {
        border: round $primary;
    }
    MetricsPanel > .metric-table {
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._table = Static("", classes="metric-table")
        self.border_title = "vLLM Metrics"

    def compose(self) -> ComposeResult:
        yield self._table

    def show_no_selection(self) -> None:
        self.border_title = "vLLM Metrics · (no model selected)"
        self._reset_lines()

    def show_pending(self) -> None:
        self.border_title = "vLLM Metrics · (waiting for first scrape...)"
        self._reset_lines()

    def show_stopped(self) -> None:
        self.border_title = "vLLM Metrics · (model not running)"
        self._reset_lines()

    def show_unreachable(self, message: str) -> None:
        del message
        self.border_title = "vLLM Metrics · (cannot reach /metrics, vLLM still loading?)"
        self._reset_lines()

    def render_metrics(self, snapshot: ModelMetricsSnapshot) -> None:
        if snapshot.last_error and snapshot.last_scrape_at:
            self.show_unreachable(snapshot.last_error)
            return
        suffix = " · last scrape failed" if snapshot.last_error else ""
        self.border_title = f"vLLM Metrics{suffix}"
        self._table.update(
            self._build_table(
                throughput=f"[bold]{format_rate(snapshot.throughput_tokens_per_s)}[/bold] tok/s",
                latency=f"[bold]{format_seconds(snapshot.e2e_latency_p95_seconds)}[/bold]",
                running=f"[bold]{format_int(snapshot.requests_running)}[/bold]",
                waiting=f"[bold]{format_int(snapshot.requests_waiting)}[/bold]",
                kv_cache=f"[bold]{format_percent(snapshot.gpu_cache_usage_percent)}[/bold]",
            )
        )

    def _reset_lines(self) -> None:
        empty = "[dim]-[/dim]"
        self._table.update(self._build_table(empty, empty, empty, empty, empty))

    def _build_table(
        self,
        throughput: str,
        latency: str,
        running: str,
        waiting: str,
        kv_cache: str,
    ) -> Table:
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 0),
            expand=True,
        )
        table.add_column("label", style="default", no_wrap=True)
        table.add_column("value", justify="right", no_wrap=True)
        table.add_row("Throughput", Text.from_markup(throughput))
        table.add_row("Latency p95", Text.from_markup(latency))
        table.add_row("", "")
        table.add_row("Running", Text.from_markup(running))
        table.add_row("Waiting", Text.from_markup(waiting))
        table.add_row("", "")
        table.add_row("KV cache used", Text.from_markup(kv_cache))
        return table
