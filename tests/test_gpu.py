"""Tests for the nvidia-smi-based GPU stats reader."""

from __future__ import annotations

import math

import pytest

from vllmctl.gpu import (
    GpuSnapshot,
    gpus_for_model,
    parse_nvidia_smi_csv,
    query_gpus,
)

# --- parse_nvidia_smi_csv ---


def test_parse_single_gpu() -> None:
    text = "0, NVIDIA A100-SXM4-40GB, 87, 18432, 40960, 350.5, 72\n"
    snaps = parse_nvidia_smi_csv(text)
    assert len(snaps) == 1
    s = snaps[0]
    assert s == GpuSnapshot(
        index=0,
        name="NVIDIA A100-SXM4-40GB",
        utilization_percent=87.0,
        memory_used_mb=18432.0,
        memory_total_mb=40960.0,
        power_w=350.5,
        temperature_c=72.0,
    )


def test_parse_multi_gpu() -> None:
    text = (
        "0, NVIDIA A100, 50, 10000, 40960, 250, 60\n"
        "1, NVIDIA A100, 80, 30000, 40960, 320, 75\n"
        "2, NVIDIA A100, 0, 100, 40960, 50, 35\n"
    )
    snaps = parse_nvidia_smi_csv(text)
    assert [s.index for s in snaps] == [0, 1, 2]
    assert snaps[1].utilization_percent == 80.0
    assert snaps[2].memory_used_mb == 100.0


def test_parse_handles_missing_data_sentinels() -> None:
    """Some consumer GPUs report [N/A] for power.draw and similar fields."""
    text = "0, GeForce RTX 4090, 25, 5000, 24576, [N/A], 65\n"
    snaps = parse_nvidia_smi_csv(text)
    assert len(snaps) == 1
    assert snaps[0].power_w is None
    assert snaps[0].temperature_c == 65.0


def test_parse_handles_not_supported() -> None:
    text = "0, Some GPU, [Not Supported], 1000, 8000, [Not Supported], [Not Supported]\n"
    snaps = parse_nvidia_smi_csv(text)
    assert snaps[0].utilization_percent is None
    assert snaps[0].power_w is None
    assert snaps[0].temperature_c is None
    assert snaps[0].memory_total_mb == 8000.0


def test_parse_skips_invalid_index() -> None:
    text = (
        "garbage, X, 0, 0, 0, 0, 0\n"  # index can't be parsed
        "0, NVIDIA, 50, 1000, 8000, 100, 50\n"
    )
    snaps = parse_nvidia_smi_csv(text)
    assert len(snaps) == 1
    assert snaps[0].index == 0


def test_parse_skips_short_lines() -> None:
    text = "0, only two fields\n0, NVIDIA, 1, 1, 1, 1, 1\n"
    snaps = parse_nvidia_smi_csv(text)
    assert len(snaps) == 1


def test_parse_empty_input() -> None:
    assert parse_nvidia_smi_csv("") == []
    assert parse_nvidia_smi_csv("\n\n   \n") == []


def test_parse_handles_nan_safely() -> None:
    """Float parsing of weird strings shouldn't raise, None or NaN stays in bounds."""
    text = "0, X, weird, 1, 1, 1, 1\n"
    snaps = parse_nvidia_smi_csv(text)
    assert snaps[0].utilization_percent is None  # 'weird' isn't a sentinel but isn't a float
    assert not math.isnan(snaps[0].memory_used_mb or 0.0)


# --- query_gpus ---


def test_query_gpus_returns_empty_when_nvidia_smi_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If nvidia-smi isn't on PATH, query_gpus returns []. No crash, no warning."""
    import vllmctl.gpu as gpu_mod  # noqa: PLC0415

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _: None)
    assert query_gpus() == []


def test_query_gpus_filters_by_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `indices=` filter drops GPUs whose physical index isn't asked for."""
    import vllmctl.gpu as gpu_mod  # noqa: PLC0415

    sample_csv = (
        "0, NVIDIA A100, 50, 10000, 40960, 250, 60\n"
        "1, NVIDIA A100, 80, 30000, 40960, 320, 75\n"
        "2, NVIDIA A100, 10, 1000, 40960, 100, 40\n"
    )

    class _FakeResult:
        returncode = 0
        stdout = sample_csv

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod.subprocess, "run", lambda *a, **kw: _FakeResult())

    snaps = query_gpus(indices=[0, 2])
    assert {s.index for s in snaps} == {0, 2}


def test_query_gpus_returns_empty_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllmctl.gpu as gpu_mod  # noqa: PLC0415

    class _FakeResult:
        returncode = 9
        stdout = ""

    monkeypatch.setattr(gpu_mod.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_mod.subprocess, "run", lambda *a, **kw: _FakeResult())

    assert query_gpus() == []


# --- gpus_for_model ---


def test_gpus_for_model_parses_comma_list() -> None:
    assert gpus_for_model({"CUDA_VISIBLE_DEVICES": "0,1,2"}) == [0, 1, 2]


def test_gpus_for_model_handles_spaces() -> None:
    assert gpus_for_model({"CUDA_VISIBLE_DEVICES": "0, 1 , 2"}) == [0, 1, 2]


def test_gpus_for_model_handles_single_device() -> None:
    assert gpus_for_model({"CUDA_VISIBLE_DEVICES": "0"}) == [0]


def test_gpus_for_model_returns_empty_when_unset() -> None:
    assert gpus_for_model({}) == []
    assert gpus_for_model({"CUDA_VISIBLE_DEVICES": ""}) == []
    assert gpus_for_model({"CUDA_VISIBLE_DEVICES": "   "}) == []


def test_gpus_for_model_skips_garbage_tokens() -> None:
    """Mixed valid/invalid tokens, keep what we can parse."""
    assert gpus_for_model({"CUDA_VISIBLE_DEVICES": "0,abc,2"}) == [0, 2]
