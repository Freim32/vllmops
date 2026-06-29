"""nvidia-smi based GPU stats reader."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    name: str
    utilization_percent: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    power_w: float | None
    temperature_c: float | None


_QUERY_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
)


def query_gpus(
    indices: list[int] | None = None,
    *,
    timeout: float = 2.0,
) -> list[GpuSnapshot]:
    """Run nvidia-smi and parse the result.

    Returns an empty list when nvidia-smi is missing, exits non-zero, or
    times out. Filters by `indices` when provided.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []

    snapshots = parse_nvidia_smi_csv(result.stdout)

    if indices is not None:
        wanted = set(indices)
        snapshots = [s for s in snapshots if s.index in wanted]
    return snapshots


def parse_nvidia_smi_csv(text: str) -> list[GpuSnapshot]:
    """Parse the CSV output of `nvidia-smi --query-gpu=...,format=csv,noheader,nounits`.

    Tolerates `[N/A]`, `[Not Supported]`, and blank entries by mapping them to None.
    """
    snapshots: list[GpuSnapshot] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(_QUERY_FIELDS):
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        snapshots.append(
            GpuSnapshot(
                index=index,
                name=parts[1],
                utilization_percent=_parse_float(parts[2]),
                memory_used_mb=_parse_float(parts[3]),
                memory_total_mb=_parse_float(parts[4]),
                power_w=_parse_float(parts[5]),
                temperature_c=_parse_float(parts[6]),
            )
        )
    return snapshots


def gpus_for_model(model_env: dict[str, str]) -> list[int]:
    """Parse CUDA_VISIBLE_DEVICES from a model's env into physical indices."""
    raw = model_env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    indices: list[int] = []
    for raw_token in raw.split(","):
        token = raw_token.strip()
        if not token:
            continue
        try:
            indices.append(int(token))
        except ValueError:
            continue
    return indices


def _parse_float(text: str) -> float | None:
    if text in ("", "[N/A]", "N/A", "[Not Supported]", "[Unknown Error]"):
        return None
    try:
        return float(text)
    except ValueError:
        return None
