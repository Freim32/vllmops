"""vLLM /metrics scraper with in-memory history.

Hits the model's HTTP port, parses the Prometheus exposition format, and
keeps a bounded ring buffer per metric so the TUI can compute rates and
quantiles without an external Prometheus instance.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field


class VllmUnreachableError(RuntimeError):
    """The model's HTTP endpoint did not respond."""


@dataclass(frozen=True)
class ParsedMetric:
    name: str
    labels: dict[str, str]
    value: float


def scrape_vllm_metrics(url: str, *, timeout: float = 1.5) -> list[ParsedMetric]:
    """GET `url` and parse the Prometheus exposition format."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise VllmUnreachableError(str(exc)) from exc
    return list(parse_prometheus_text(body))


def parse_prometheus_text(text: str) -> Iterator[ParsedMetric]:
    """Parse the Prometheus exposition format.

    Skips comments and blank lines. Trailing timestamps are ignored.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name_part, rest = line.split("{", 1)
                labels_str, _, value_part = rest.partition("}")
                labels = _parse_labels(labels_str)
                value_str = value_part.strip().split()[0]
            else:
                head, value_str = line.split(maxsplit=1)
                value_str = value_str.split()[0]
                name_part = head
                labels = {}
            value = _parse_value(value_str)
        except (ValueError, IndexError):
            continue
        if value is None:
            continue
        yield ParsedMetric(name=name_part.strip(), labels=labels, value=value)


def _parse_value(text: str) -> float | None:
    if text in ("+Inf", "Inf"):
        return float("inf")
    if text == "-Inf":
        return float("-inf")
    if text == "NaN":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_labels(text: str) -> dict[str, str]:
    """Parse `k1="v1",k2="v2"` with backslash-escaped quotes inside values."""
    result: dict[str, str] = {}
    if not text.strip():
        return result
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index] in " ,":
            index += 1
        if index >= length:
            break
        key_start = index
        while index < length and text[index] != "=":
            index += 1
        if index >= length:
            break
        key = text[key_start:index].strip()
        index += 1
        if index >= length or text[index] != '"':
            break
        index += 1
        value_chars: list[str] = []
        while index < length:
            ch = text[index]
            if ch == "\\" and index + 1 < length:
                next_ch = text[index + 1]
                value_chars.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"'}.get(next_ch, next_ch))
                index += 2
                continue
            if ch == '"':
                index += 1
                break
            value_chars.append(ch)
            index += 1
        if key:
            result[key] = "".join(value_chars)
    return result


@dataclass
class TimeSeries:
    """Bounded ring of (timestamp, value) for a single metric stream."""

    capacity: int = 360
    points: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=360))

    def __post_init__(self) -> None:
        if self.points.maxlen != self.capacity:
            self.points = deque(self.points, maxlen=self.capacity)

    def append(self, timestamp: float, value: float) -> None:
        self.points.append((timestamp, value))

    def values(self) -> list[float]:
        return [v for _, v in self.points]

    def latest(self) -> float | None:
        return self.points[-1][1] if self.points else None

    def rate(self, window_seconds: float = 60.0) -> float | None:
        """Counter rate over the last `window_seconds`. Returns None on counter reset."""
        if len(self.points) < 2:
            return None
        latest_ts, latest_value = self.points[-1]
        cutoff = latest_ts - window_seconds
        old: tuple[float, float] | None = None
        for ts, value in self.points:
            if ts >= cutoff:
                old = (ts, value)
                break
        if old is None or old == self.points[-1]:
            return None
        old_ts, old_value = old
        delta_t = latest_ts - old_ts
        delta_v = latest_value - old_value
        if delta_t <= 0 or delta_v < 0:
            return None
        return delta_v / delta_t


@dataclass
class MetricsHistory:
    """Per-model rolling history of all scraped metrics, keyed by (name, labels)."""

    capacity: int = 360
    series: dict[tuple[str, tuple[tuple[str, str], ...]], TimeSeries] = field(default_factory=dict)
    last_scrape_at: float | None = None
    last_error: str | None = None

    def ingest(self, samples: list[ParsedMetric], *, now: float | None = None) -> None:
        timestamp = now if now is not None else time.time()
        for sample in samples:
            key = (sample.name, _freeze_labels(sample.labels))
            ts = self.series.get(key)
            if ts is None:
                ts = TimeSeries(capacity=self.capacity)
                self.series[key] = ts
            ts.append(timestamp, sample.value)
        self.last_scrape_at = timestamp
        self.last_error = None

    def record_error(self, message: str, *, now: float | None = None) -> None:
        self.last_scrape_at = now if now is not None else time.time()
        self.last_error = message

    def latest(self, metric: str) -> float | None:
        for (name, _labels), ts in self.series.items():
            if name == metric:
                return ts.latest()
        return None

    def rate(self, metric: str, window_seconds: float = 60.0) -> float | None:
        """Sum the rates across all label sets for a counter."""
        rates: list[float] = []
        for (name, _labels), ts in self.series.items():
            if name == metric:
                value = ts.rate(window_seconds=window_seconds)
                if value is not None:
                    rates.append(value)
        if not rates:
            return None
        return sum(rates)

    def rate_history(self, metric: str, *, window_seconds: float = 60.0) -> list[float]:
        """Per-sample rate history for sparklines."""
        merged: dict[float, float] = {}
        for (name, _labels), ts in self.series.items():
            if name != metric:
                continue
            for timestamp, value in ts.points:
                merged[timestamp] = merged.get(timestamp, 0.0) + value
        if len(merged) < 2:
            return []
        ordered = sorted(merged.items())
        rates: list[float] = []
        for (prev_ts, prev_v), (cur_ts, cur_v) in zip(ordered, ordered[1:]):
            dt = cur_ts - prev_ts
            dv = cur_v - prev_v
            if dt <= 0 or dv < 0:
                rates.append(0.0)
            else:
                rates.append(dv / dt)
        cutoff = ordered[-1][0] - window_seconds * 6
        kept_count = sum(1 for ts, _ in ordered[1:] if ts >= cutoff)
        return rates[-kept_count:] if kept_count > 0 else rates

    def histogram_quantile(
        self,
        metric_base: str,
        quantile: float = 0.95,
    ) -> float | None:
        """Approximate quantile from histogram buckets via linear interpolation.

        `metric_base` is the name without the `_bucket` suffix.
        """
        bucket_metric = f"{metric_base}_bucket"
        per_le: dict[float, float] = {}
        for (name, labels_tuple), ts in self.series.items():
            if name != bucket_metric:
                continue
            le_value = _le_from_labels(labels_tuple)
            if le_value is None:
                continue
            latest = ts.latest()
            if latest is None:
                continue
            per_le[le_value] = per_le.get(le_value, 0.0) + latest
        if not per_le:
            return None
        ordered = sorted(per_le.items())
        total = ordered[-1][1]
        if total <= 0:
            return None
        target = quantile * total
        prev_le = 0.0
        prev_count = 0.0
        for le_value, count in ordered:
            if count >= target:
                if le_value == float("inf"):
                    return prev_le if prev_count > 0 else None
                if count == prev_count:
                    return le_value
                fraction = (target - prev_count) / (count - prev_count)
                return prev_le + (le_value - prev_le) * fraction
            prev_le = le_value
            prev_count = count
        return None


def _freeze_labels(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _le_from_labels(labels_tuple: tuple[tuple[str, str], ...]) -> float | None:
    for key, value in labels_tuple:
        if key == "le":
            if value in ("+Inf", "Inf"):
                return float("inf")
            try:
                return float(value)
            except ValueError:
                return None
    return None


@dataclass(frozen=True)
class ModelMetricsSnapshot:
    """Headline metrics rendered in the TUI for one model."""

    throughput_tokens_per_s: float | None
    e2e_latency_p95_seconds: float | None
    requests_running: float | None
    requests_waiting: float | None
    gpu_cache_usage_percent: float | None
    last_error: str | None
    last_scrape_at: float | None

    @classmethod
    def empty(cls, last_error: str | None = None, last_scrape_at: float | None = None) -> ModelMetricsSnapshot:
        return cls(
            throughput_tokens_per_s=None,
            e2e_latency_p95_seconds=None,
            requests_running=None,
            requests_waiting=None,
            gpu_cache_usage_percent=None,
            last_error=last_error,
            last_scrape_at=last_scrape_at,
        )


# vLLM renamed `gpu_cache_usage_perc` to `kv_cache_usage_perc` around 0.7.
# Probe both names so vllmctl works across a wider range of versions.
_KV_CACHE_USAGE_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)


def _first_present_latest(history: MetricsHistory, *names: str) -> float | None:
    for name in names:
        value = history.latest(name)
        if value is not None:
            return value
    return None


def snapshot_from_history(history: MetricsHistory) -> ModelMetricsSnapshot:
    cache_pct = _first_present_latest(history, *_KV_CACHE_USAGE_METRICS)
    if cache_pct is not None and 0.0 <= cache_pct <= 1.0:
        cache_pct = cache_pct * 100.0
    return ModelMetricsSnapshot(
        throughput_tokens_per_s=history.rate("vllm:generation_tokens_total"),
        e2e_latency_p95_seconds=history.histogram_quantile("vllm:e2e_request_latency_seconds", 0.95),
        requests_running=history.latest("vllm:num_requests_running"),
        requests_waiting=history.latest("vllm:num_requests_waiting"),
        gpu_cache_usage_percent=cache_pct,
        last_error=history.last_error,
        last_scrape_at=history.last_scrape_at,
    )
