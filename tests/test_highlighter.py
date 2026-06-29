"""Tests for the log line highlighter (prefix stripping + style spans)."""

from __future__ import annotations

from vllmops.tui.widgets.highlighter import highlight_log_line, strip_process_prefix


def test_strip_process_prefix_removes_basic_prefix() -> None:
    line = "(APIServer pid=668) INFO 06-22 18:01:55 server up"
    assert strip_process_prefix(line) == "INFO 06-22 18:01:55 server up"


def test_strip_process_prefix_handles_multiple_kv_pairs() -> None:
    line = "(VllmWorker rank=0 pid=672) WARNING something"
    assert strip_process_prefix(line) == "WARNING something"


def test_strip_process_prefix_handles_underscored_name() -> None:
    line = "(EngineCore_DP0 pid=720) INFO starting"
    assert strip_process_prefix(line) == "INFO starting"


def test_strip_process_prefix_handles_no_kv_pairs() -> None:
    line = "(WorkerA) doing work"
    assert strip_process_prefix(line) == "doing work"


def test_strip_process_prefix_leaves_unmatched_lines_alone() -> None:
    line = "INFO no prefix here"
    assert strip_process_prefix(line) == "INFO no prefix here"


def test_strip_process_prefix_leaves_inline_parens_alone() -> None:
    """Parens later in the line stay untouched."""
    line = "INFO using (sampler=greedy) for request"
    assert strip_process_prefix(line) == "INFO using (sampler=greedy) for request"


def test_strip_process_prefix_strips_stacked_prefixes() -> None:
    """vLLM stacks prefixes when one prefixed process captures another's stdout."""
    line = "(EngineCore pid=3068) (APIServer pid=3014) INFO trigger received"
    assert strip_process_prefix(line) == "INFO trigger received"


def test_strip_process_prefix_handles_empty_line() -> None:
    assert strip_process_prefix("") == ""


def test_highlight_log_line_strips_prefix_in_output() -> None:
    """End-to-end: the styled Text reflects the stripped content, not the original."""
    line = "(APIServer pid=668) INFO 06-22 18:01:55 ready"
    text = highlight_log_line(line)
    assert text.plain == "INFO 06-22 18:01:55 ready"


def test_highlight_log_line_styles_error_keyword() -> None:
    text = highlight_log_line("ERROR something went wrong")
    spans = [(s.start, s.end, s.style) for s in text.spans]
    assert (0, 5, "bold red") in spans


def _spans(line: str) -> list[tuple[int, int, object]]:
    text = highlight_log_line(line)
    return [(s.start, s.end, s.style) for s in text.spans]


def test_http_status_2xx_styled_green() -> None:
    line = '"POST /v1/completions HTTP/1.1" 200 OK'
    assert any(end - start == 3 and style == "green" for start, end, style in _spans(line))


def test_http_status_4xx_styled_bold_yellow() -> None:
    line = '"GET /v1/missing HTTP/1.1" 404 Not Found'
    assert any(style == "bold yellow" for _start, _end, style in _spans(line))


def test_http_status_5xx_styled_bold_red() -> None:
    line = '"POST /v1/completions HTTP/1.1" 500 Internal Server Error'
    assert any(style == "bold red" for _start, _end, style in _spans(line))


def test_http_status_3xx_styled_cyan() -> None:
    line = '"GET /redoc HTTP/1.1" 301 Moved Permanently'
    assert any(style == "cyan" and (end - start == 3) for start, end, style in _spans(line))


def test_http_status_not_matched_outside_http_context() -> None:
    """A bare `200 OK` without the `HTTP/1.1"` anchor should NOT be re-colored."""
    line = "INFO loaded 200 shards"
    assert not any(style == "green" and (end - start == 3) for start, end, style in _spans(line))


def test_throughput_tokens_per_second_styled_bold_cyan() -> None:
    line = "Avg prompt throughput: 0.4 tokens/s, gen 11.8 tokens/s"
    spans = _spans(line)
    matches = [(s, e, style) for s, e, style in spans if style == "bold cyan"]
    assert len(matches) >= 2


def test_memory_size_styled_dim() -> None:
    line = "Model loading took 0.24 GiB memory"
    assert any(style == "dim" for _s, _e, style in _spans(line))


def test_duration_seconds_styled_dim() -> None:
    line = "Loading weights took 1.708 seconds"
    assert any(style == "dim" for _s, _e, style in _spans(line))


def test_duration_bare_s_with_space_styled_dim() -> None:
    line = "torch.compile took 1.48 s in total"
    assert any(style == "dim" for _s, _e, style in _spans(line))


def test_url_path_broadened_matches_non_v1_routes() -> None:
    line = "Route: /openapi.json, Methods: HEAD, GET"
    spans = _spans(line)
    assert any(style == "blue" for _s, _e, style in spans)


def test_url_path_matches_health_metrics_ping() -> None:
    line = "Route: /health · /metrics · /ping"
    blue_count = sum(1 for _s, _e, style in _spans(line) if style == "blue")
    assert blue_count == 3


def test_file_line_reference_styled_pale_blue() -> None:
    line = "INFO [api_server.py:583] starting"
    assert any(style == "light_steel_blue" for _s, _e, style in _spans(line))


def test_url_does_not_match_tokens_per_second() -> None:
    """`tokens/s` must NOT trigger the URL highlighter as a standalone `/s` match."""
    line = "Avg generation throughput: 11.8 tokens/s"
    text = highlight_log_line(line)
    for span in text.spans:
        substring = text.plain[span.start : span.end]
        assert substring != "/s"


def test_url_does_not_match_mid_word_slash() -> None:
    """A slash inside a word (e.g. `shape/config`) must not be picked up as URL."""
    line = "consider extending warmup to cover this shape/config."
    spans = _spans(line)
    assert not any(style == "blue" for _s, _e, style in spans)


def test_url_matches_at_line_start() -> None:
    """A URL at column 0 (no preceding whitespace) must still match."""
    line = "/v1/completions called"
    spans = _spans(line)
    assert any(style == "blue" for _s, _e, style in spans)
