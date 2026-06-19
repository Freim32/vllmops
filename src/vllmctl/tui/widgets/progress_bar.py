"""Inline unicode progress bar used by metrics panels."""

from __future__ import annotations


def render_pct_bar(pct: float | None, *, width: int = 14) -> str:
    """Colored unicode bar for a 0-100 percentage. Returns Rich markup.

    Threshold colors: green < 50, yellow 50-80, red >= 80.
    """
    if pct is None:
        return f"[dim]{'─' * width}[/dim]"
    pct = max(0.0, min(100.0, pct))
    filled = int(round((pct / 100.0) * width))
    empty = width - filled
    color = _bar_color(pct)
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


def _bar_color(pct: float) -> str:
    if pct < 50.0:
        return "green"
    if pct < 80.0:
        return "yellow"
    return "red"
