"""Shared value-formatting helpers for metrics and GPU panels."""

from __future__ import annotations


def format_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:6.1f}"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 1:
        return f"{value * 1000:6.0f} ms"
    return f"{value:6.2f} s"


def format_int(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{int(value)}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:5.1f}%"


def format_gpu_memory(used_mb: float | None, total_mb: float | None) -> str:
    if used_mb is None or total_mb is None or total_mb <= 0:
        return "-"
    return f"{used_mb / 1024:.1f} / {total_mb / 1024:.1f} GB"


def format_watts(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:5.0f} W"


def format_celsius(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:4.0f} °C"


def gpu_memory_percent(used_mb: float | None, total_mb: float | None) -> float | None:
    if used_mb is None or total_mb is None or total_mb <= 0:
        return None
    return (used_mb / total_mb) * 100.0


def format_device_list(indices: list[int]) -> str:
    """Compact human-readable list of GPU indices.

    1 device       -> "GPU 0"
    2-3 devices    -> "GPU 0, GPU 1, GPU 2"
    contiguous N>=4 -> "GPU 0-7  (8 devices)"
    sparse N>=4    -> "GPU 0, ..., 7  (5 devices)"
    """
    if not indices:
        return "-"
    sorted_idx = sorted(set(indices))
    if len(sorted_idx) == 1:
        return f"GPU {sorted_idx[0]}"
    if len(sorted_idx) <= 3:
        return ", ".join(f"GPU {i}" for i in sorted_idx)
    contiguous = sorted_idx == list(range(sorted_idx[0], sorted_idx[-1] + 1))
    if contiguous:
        return f"GPU {sorted_idx[0]}-{sorted_idx[-1]}  ({len(sorted_idx)} devices)"
    return f"GPU {sorted_idx[0]}, ..., {sorted_idx[-1]}  ({len(sorted_idx)} devices)"
