"""GPU metrics panel aggregating across the model's CUDA devices."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from vllmops.gpu import GpuSnapshot
from vllmops.tui.widgets._format import (
    format_celsius,
    format_device_list,
    format_gpu_memory,
    format_percent,
    format_watts,
)


class GpuPanel(Vertical):
    """GPU stats aggregated across the selected model's CUDA devices.

    Fixed shape regardless of GPU count: same rows whether the model uses 1
    or 8 GPUs.
    """

    DEFAULT_CSS = """
    GpuPanel {
        height: 1fr;
        border: round $accent;
        padding: 1 2;
    }
    GpuPanel:focus-within {
        border: round $primary;
    }
    GpuPanel > .gpu-table {
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._table = Static("", classes="gpu-table")
        self.border_title = "GPU Metrics"

    def compose(self) -> ComposeResult:
        yield self._table

    def render_gpus(
        self,
        configured_indices: list[int] | None,
        gpus: list[GpuSnapshot],
    ) -> None:
        if configured_indices is None:
            self.border_title = "GPU Metrics"
            self._reset()
            return
        if not configured_indices:
            self.border_title = "GPU Metrics"
            self._reset(devices="[dim]no CUDA_VISIBLE_DEVICES set[/dim]")
            return
        if not gpus:
            self.border_title = "GPU Metrics"
            self._reset(devices="[yellow]nvidia-smi unavailable[/yellow]")
            return

        self.border_title = "GPU Metrics"

        utils = [g.utilization_percent for g in gpus if g.utilization_percent is not None]
        avg_util = sum(utils) / len(utils) if utils else None

        mem_used = [g.memory_used_mb for g in gpus if g.memory_used_mb is not None]
        mem_total = [g.memory_total_mb for g in gpus if g.memory_total_mb is not None]
        sum_mem_used = sum(mem_used) if mem_used else None
        sum_mem_total = sum(mem_total) if mem_total else None

        powers = [g.power_w for g in gpus if g.power_w is not None]
        sum_power = sum(powers) if powers else None

        temps = [g.temperature_c for g in gpus if g.temperature_c is not None]
        max_temp = max(temps) if temps else None

        device_indices = sorted(g.index for g in gpus)
        self._table.update(
            self._build_table(
                devices=f"[bold]{format_device_list(device_indices)}[/bold]",
                util=f"[bold]{format_percent(avg_util)}[/bold]",
                mem=f"[bold]{format_gpu_memory(sum_mem_used, sum_mem_total)}[/bold]",
                power=f"[bold]{format_watts(sum_power)}[/bold]",
                temp=f"[bold]{format_celsius(max_temp)}[/bold]",
            )
        )

    def _reset(self, *, devices: str = "[dim]-[/dim]") -> None:
        empty = "[dim]-[/dim]"
        self._table.update(self._build_table(devices, empty, empty, empty, empty))

    def _build_table(
        self,
        devices: str,
        util: str,
        mem: str,
        power: str,
        temp: str,
    ) -> Table:
        table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 0),
            expand=True,
        )
        table.add_column("label", style="default", no_wrap=True)
        table.add_column("value", justify="right", overflow="fold")
        table.add_row("Devices", Text.from_markup(devices))
        table.add_row("", "")
        table.add_row("Util (avg)", Text.from_markup(util))
        table.add_row("Mem (sum)", Text.from_markup(mem))
        table.add_row("", "")
        table.add_row("Power (sum)", Text.from_markup(power))
        table.add_row("Temp (max)", Text.from_markup(temp))
        return table
