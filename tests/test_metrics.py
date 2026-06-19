"""Tests for the vLLM /metrics scraper, parser, and in-memory history."""

from __future__ import annotations

import math

import pytest

from tests.conftest import MockVllmMetricsHandler
from vllmctl.metrics import (
    MetricsHistory,
    ModelMetricsSnapshot,
    TimeSeries,
    VllmUnreachableError,
    parse_prometheus_text,
    scrape_vllm_metrics,
    snapshot_from_history,
)

# --- parse_prometheus_text ---


def test_parse_simple_lines() -> None:
    text = """# HELP vllm:num_requests_running ...
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 5.0
vllm:num_requests_waiting 0.0
"""
    samples = list(parse_prometheus_text(text))
    by_name = {s.name: s for s in samples}
    assert by_name["vllm:num_requests_running"].value == 5.0
    assert by_name["vllm:num_requests_waiting"].value == 0.0
    assert all(s.labels == {} for s in samples)


def test_parse_with_labels() -> None:
    text = 'vllm:generation_tokens_total{model_name="llama"} 12345.0'
    samples = list(parse_prometheus_text(text))
    assert len(samples) == 1
    assert samples[0].name == "vllm:generation_tokens_total"
    assert samples[0].labels == {"model_name": "llama"}
    assert samples[0].value == 12345.0


def test_parse_multiple_labels_with_escapes() -> None:
    text = 'metric{a="1",b="value with \\"quote\\""} 3.14'
    samples = list(parse_prometheus_text(text))
    assert samples[0].labels == {"a": "1", "b": 'value with "quote"'}


def test_parse_skips_comments_and_blanks() -> None:
    text = "\n# HELP foo\n# TYPE foo gauge\n\nfoo 1.0\n"
    samples = list(parse_prometheus_text(text))
    assert len(samples) == 1
    assert samples[0].value == 1.0


def test_parse_skips_invalid_value() -> None:
    text = "metric_a 1.0\nmetric_b not_a_number\nmetric_c 2.0"
    samples = list(parse_prometheus_text(text))
    names = {s.name for s in samples}
    assert names == {"metric_a", "metric_c"}


def test_parse_handles_inf_and_nan() -> None:
    text = "metric{le=\"+Inf\"} +Inf\nmetric{le=\"NaN\"} NaN"
    samples = list(parse_prometheus_text(text))
    assert math.isinf(samples[0].value)
    assert math.isnan(samples[1].value)


def test_parse_histogram_buckets() -> None:
    text = """vllm:e2e_request_latency_seconds_bucket{le="0.1"} 5
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 10
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 12
vllm:e2e_request_latency_seconds_sum 3.7
vllm:e2e_request_latency_seconds_count 12"""
    samples = list(parse_prometheus_text(text))
    assert len(samples) == 5
    bucket_samples = [s for s in samples if s.name.endswith("_bucket")]
    assert len(bucket_samples) == 3


# --- scrape_vllm_metrics ---


def test_scrape_returns_parsed_samples(mock_vllm_metrics: tuple[str, type[MockVllmMetricsHandler]]) -> None:
    base_url, handler = mock_vllm_metrics
    handler.response_body = b"foo 1.0\nbar 2.0\n"
    samples = scrape_vllm_metrics(f"{base_url}/metrics")
    by_name = {s.name: s.value for s in samples}
    assert by_name == {"foo": 1.0, "bar": 2.0}


def test_scrape_unreachable_raises() -> None:
    with pytest.raises(VllmUnreachableError):
        scrape_vllm_metrics("http://127.0.0.1:1/metrics", timeout=0.2)


# --- TimeSeries ---


def test_timeseries_appends_within_capacity() -> None:
    ts = TimeSeries(capacity=3)
    for i in range(5):
        ts.append(float(i), float(i * 10))
    assert ts.values() == [20.0, 30.0, 40.0]


def test_timeseries_rate_simple_counter() -> None:
    ts = TimeSeries(capacity=10)
    ts.append(0.0, 0.0)
    ts.append(10.0, 100.0)
    ts.append(20.0, 200.0)
    rate = ts.rate(window_seconds=30.0)
    assert rate is not None
    assert rate == pytest.approx(10.0, abs=0.01)


def test_timeseries_rate_returns_none_with_one_sample() -> None:
    ts = TimeSeries(capacity=10)
    ts.append(0.0, 0.0)
    assert ts.rate() is None


def test_timeseries_rate_returns_none_on_counter_reset() -> None:
    ts = TimeSeries(capacity=10)
    ts.append(0.0, 100.0)
    ts.append(10.0, 50.0)  # decrease, process restart
    assert ts.rate() is None


def test_timeseries_latest() -> None:
    ts = TimeSeries(capacity=10)
    assert ts.latest() is None
    ts.append(1.0, 5.0)
    ts.append(2.0, 7.0)
    assert ts.latest() == 7.0


# --- MetricsHistory ingest + accessors ---


def _ingest_text(history: MetricsHistory, text: str, *, ts: float) -> None:
    history.ingest(list(parse_prometheus_text(text)), now=ts)


def test_history_latest_gauge() -> None:
    history = MetricsHistory(capacity=10)
    _ingest_text(history, "vllm:num_requests_running 4.0", ts=1.0)
    assert history.latest("vllm:num_requests_running") == 4.0


def test_history_rate_aggregates_across_label_sets() -> None:
    history = MetricsHistory(capacity=10)
    _ingest_text(
        history,
        'metric{shard="a"} 0\nmetric{shard="b"} 0',
        ts=0.0,
    )
    _ingest_text(
        history,
        'metric{shard="a"} 30\nmetric{shard="b"} 60',
        ts=10.0,
    )
    rate = history.rate("metric")
    # 30/10 + 60/10 = 9
    assert rate == pytest.approx(9.0)


def test_history_rate_history_returns_per_step_increments() -> None:
    history = MetricsHistory(capacity=10)
    _ingest_text(history, "counter 0", ts=0.0)
    _ingest_text(history, "counter 100", ts=10.0)
    _ingest_text(history, "counter 250", ts=20.0)
    rates = history.rate_history("counter")
    assert rates == [pytest.approx(10.0), pytest.approx(15.0)]


def test_history_records_error_and_clears_on_success() -> None:
    history = MetricsHistory()
    history.record_error("connection refused", now=1.0)
    assert history.last_error == "connection refused"
    assert history.last_scrape_at == 1.0
    history.ingest([], now=2.0)
    assert history.last_error is None
    assert history.last_scrape_at == 2.0


def test_histogram_quantile_basic_interpolation() -> None:
    history = MetricsHistory()
    text = """vllm:e2e_request_latency_seconds_bucket{le="0.1"} 0
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 50
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 95
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 100"""
    _ingest_text(history, text, ts=1.0)
    # p95 should fall between 0.5 and 1.0; with 95 in bucket le=1.0 and target=95 → exactly le=1.0
    p95 = history.histogram_quantile("vllm:e2e_request_latency_seconds", 0.95)
    assert p95 == pytest.approx(1.0)
    # p50: target=50 lands exactly at the le=0.5 bucket edge
    p50 = history.histogram_quantile("vllm:e2e_request_latency_seconds", 0.50)
    assert p50 is not None
    assert 0.1 < p50 <= 0.5


def test_histogram_quantile_returns_none_when_no_buckets() -> None:
    history = MetricsHistory()
    _ingest_text(history, "other_metric 1", ts=1.0)
    assert history.histogram_quantile("foo", 0.95) is None


# --- snapshot_from_history ---


def test_snapshot_populates_known_fields() -> None:
    history = MetricsHistory()
    text_t0 = """vllm:num_requests_running 3
vllm:num_requests_waiting 0
vllm:gpu_cache_usage_perc 0.42
vllm:generation_tokens_total 0"""
    text_t1 = """vllm:num_requests_running 3
vllm:num_requests_waiting 1
vllm:gpu_cache_usage_perc 0.50
vllm:generation_tokens_total 1000"""
    _ingest_text(history, text_t0, ts=0.0)
    _ingest_text(history, text_t1, ts=10.0)
    snap = snapshot_from_history(history)
    assert snap.requests_running == 3.0
    assert snap.requests_waiting == 1.0
    assert snap.gpu_cache_usage_percent == pytest.approx(50.0)
    assert snap.throughput_tokens_per_s == pytest.approx(100.0)


def test_snapshot_empty_when_no_data() -> None:
    snap = snapshot_from_history(MetricsHistory())
    assert snap == ModelMetricsSnapshot.empty()


def test_snapshot_passes_through_error() -> None:
    history = MetricsHistory()
    history.record_error("boom", now=1.0)
    snap = snapshot_from_history(history)
    assert snap.last_error == "boom"
    assert snap.last_scrape_at == 1.0


def test_gpu_cache_already_in_percent_is_passed_through() -> None:
    """Some vLLM versions emit a 0-100 value instead of 0-1."""
    history = MetricsHistory()
    _ingest_text(history, "vllm:gpu_cache_usage_perc 42", ts=1.0)
    snap = snapshot_from_history(history)
    assert snap.gpu_cache_usage_percent == 42.0


def test_kv_cache_metric_uses_new_name_in_vllm_07() -> None:
    """vLLM 0.7+ renamed gpu_cache_usage_perc to kv_cache_usage_perc."""
    history = MetricsHistory()
    _ingest_text(history, 'vllm:kv_cache_usage_perc{model_name="m"} 0.42', ts=1.0)
    snap = snapshot_from_history(history)
    assert snap.gpu_cache_usage_percent == pytest.approx(42.0)


def test_kv_cache_new_name_wins_over_old_when_both_present() -> None:
    """If a vLLM build inexplicably exposes both, prefer the modern spelling."""
    history = MetricsHistory()
    _ingest_text(
        history,
        'vllm:kv_cache_usage_perc{model_name="m"} 0.7\n'
        'vllm:gpu_cache_usage_perc{model_name="m"} 0.2',
        ts=1.0,
    )
    snap = snapshot_from_history(history)
    assert snap.gpu_cache_usage_percent == pytest.approx(70.0)
