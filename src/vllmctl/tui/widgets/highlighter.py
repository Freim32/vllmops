"""Regex-based syntax highlighting for vLLM log lines."""

from __future__ import annotations

import re

from rich.highlighter import Highlighter
from rich.text import Text

_LOG_HIGHLIGHTS: list[tuple[re.Pattern[str], str]] = [
    # Log severity keywords
    (re.compile(r"\bERROR\b"), "bold red"),
    (re.compile(r"\bWARN(?:ING)?\b"), "bold yellow"),
    (re.compile(r"\bINFO\b"), "cyan"),
    (re.compile(r"\bDEBUG\b"), "dim"),
    (re.compile(r"\bTRACEBACK\b", re.IGNORECASE), "bold red"),
    # vLLM timestamps (MM-DD HH:MM:SS[.us])
    (re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "dim"),
    # Source-file references for .py, .c, .go, ... not just Python
    (re.compile(r"\[[\w./-]+\.[a-z]+:\d+\]"), "light_steel_blue"),
    # HTTP methods
    (re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b"), "green"),
    # URL path. Must follow whitespace or start of line so mid-word slashes
    # like `shape/config` or `tokens/streaming` don't get picked up.
    (re.compile(r"(?:^|(?<=\s))/[a-zA-Z][\w./_{}+-]+"), "blue"),
    # HTTP status codes (anchored to the standard `HTTP/x.y" NNN ` access-log shape)
    (re.compile(r'(?<=HTTP/\d\.\d"\s)2\d{2}\b'), "green"),
    (re.compile(r'(?<=HTTP/\d\.\d"\s)3\d{2}\b'), "cyan"),
    (re.compile(r'(?<=HTTP/\d\.\d"\s)4\d{2}\b'), "bold yellow"),
    (re.compile(r'(?<=HTTP/\d\.\d"\s)5\d{2}\b'), "bold red"),
    # Token throughput (vLLM's headline metric)
    (re.compile(r"\b\d+(?:\.\d+)?\s*tokens?/s\b"), "bold cyan"),
    # Memory sizes with binary/decimal units
    (re.compile(r"\b\d+(?:[.,]\d+)*\s*(?:GiB|MiB|KiB|TiB|GB|MB|KB|TB)\b"), "dim"),
    # Durations: explicit unit words, plus the bare `N s` shape used by torch/vLLM
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|ms|us|μs|ns)\b"), "dim"),
    (re.compile(r"\b\d+(?:\.\d+)?\s+s\b"), "dim"),
]

# vLLM (and Ray under the hood) prefixes log lines with the producer process
# name and id, e.g. "(APIServer pid=668) INFO ..." or
# "(VllmWorker rank=0 pid=672) ...". When one prefixed process captures the
# stdout of another, prefixes stack: "(EngineCore pid=N) (APIServer pid=M) ...".
_PROCESS_PREFIX = re.compile(r"^\([A-Za-z]\w*(?:\s+[\w.-]+=[^)\s]+)*\)\s+")


def strip_process_prefix(line: str) -> str:
    """Remove every leading `(ProcessName key=value ...)` block, if present."""
    while True:
        match = _PROCESS_PREFIX.match(line)
        if match is None:
            return line
        line = line[match.end() :]


class VllmLogHighlighter(Highlighter):
    """Rich highlighter that applies the vLLM log pattern set to a `Text`.

    Plug this into `RichLog(highlight=True)` and set the `highlighter`
    attribute. Each `pattern.highlight_regex(...)` call delegates to Rich's
    built-in span-styler, which mirrors the manual `finditer` + `stylize`
    loop without taking a dependency on themed group names.
    """

    def highlight(self, text: Text) -> None:
        for pattern, style in _LOG_HIGHLIGHTS:
            text.highlight_regex(pattern, style=style)


_HIGHLIGHTER = VllmLogHighlighter()


def highlight_log_line(line: str) -> Text:
    """Strip the process prefix and return the highlighted `Text`.

    Convenience for tests and any future caller that wants a styled `Text`
    (e.g. a CLI tail command); the TUI uses the `Highlighter` directly.
    """
    return _HIGHLIGHTER(strip_process_prefix(line))
