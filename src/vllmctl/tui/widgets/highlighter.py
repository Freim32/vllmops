"""Regex-based syntax highlighting for vLLM log lines."""

from __future__ import annotations

import re

from rich.text import Text

_LOG_HIGHLIGHTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bERROR\b"), "bold red"),
    (re.compile(r"\bWARN(?:ING)?\b"), "bold yellow"),
    (re.compile(r"\bINFO\b"), "cyan"),
    (re.compile(r"\bDEBUG\b"), "dim"),
    (re.compile(r"\bTRACEBACK\b", re.IGNORECASE), "bold red"),
    (re.compile(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "dim"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"), "dim"),
    (re.compile(r"\[[\w./-]+\.py:\d+\]"), "magenta"),
    (re.compile(r"\bpid=\d+"), "dim"),
    (re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b"), "green"),
    (re.compile(r"/v\d+/[\w/{}.+_-]+"), "bright_blue"),
]


def highlight_log_line(line: str) -> Text:
    """Build a styled `Text` from a log line.

    Uses `Text.stylize` so we never produce markup that Rich could misparse.
    """
    text = Text(line)
    for pattern, style in _LOG_HIGHLIGHTS:
        for match in pattern.finditer(line):
            text.stylize(style, match.start(), match.end())
    return text
