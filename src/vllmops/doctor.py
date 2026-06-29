"""Read-only environment diagnostic for the local vllmops setup."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vllmops import gpu as gpu_module
from vllmops import service
from vllmops.config import load_model_file
from vllmops.project import Project, find_project_root, load_project

Status = Literal["ok", "warn", "fail"]

_LOG_LINE_PREFIX = re.compile(r"^(WARNING|INFO|ERROR|DEBUG|CRITICAL)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    hint: str | None = None


def check_python_version() -> CheckResult:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        return CheckResult("Python version", "ok", version)
    return CheckResult(
        "Python version",
        "fail",
        f"{version} (need >=3.10)",
        hint="Install Python 3.10 or newer",
    )


def check_nvidia_smi() -> CheckResult:
    if shutil.which("nvidia-smi") is None:
        return CheckResult(
            "nvidia-smi",
            "warn",
            "not on PATH",
            hint="install NVIDIA drivers, or ignore if running CPU-only experiments",
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return CheckResult("nvidia-smi", "warn", str(exc))
    if result.returncode != 0:
        return CheckResult(
            "nvidia-smi",
            "warn",
            "command failed",
            hint=(result.stderr.strip().splitlines()[0] if result.stderr.strip() else None),
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return CheckResult("nvidia-smi", "warn", "no GPUs reported")
    return CheckResult("nvidia-smi", "ok", f"{len(lines)} GPU(s) detected")


def check_project_root() -> CheckResult:
    root = find_project_root(Path.cwd())
    if root is None:
        return CheckResult(
            "Project root",
            "fail",
            "no .vllmops/config.yaml found in current dir or any parent",
            hint="run `vllmops init` to create a project workspace",
        )
    return CheckResult("Project root", "ok", str(root))


def check_venv(project: Project) -> CheckResult:
    venv = project.root / ".venv"
    if not venv.is_dir():
        return CheckResult(
            ".venv directory",
            "fail",
            f"{venv} does not exist",
            hint="run `uv sync` in the project root",
        )
    return CheckResult(".venv directory", "ok", str(venv))


def check_vllm_executable(project: Project) -> CheckResult:
    exe = project.venv_executable
    if exe is None:
        return CheckResult(
            "vllm executable",
            "fail",
            "not found in .venv/bin or .venv/Scripts",
            hint="run `uv sync` to install vllm into the project venv",
        )
    return CheckResult("vllm executable", "ok", str(exe))


def check_vllm_version(project: Project) -> CheckResult:
    exe = project.venv_executable
    if exe is None:
        return CheckResult("vllm version", "fail", "vllm not installed")
    try:
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            "vllm version",
            "warn",
            "timed out after 15s",
            hint="run `vllm --version` manually to investigate",
        )
    except OSError as exc:
        return CheckResult("vllm version", "fail", str(exc))
    if result.returncode != 0:
        first_err = result.stderr.strip().splitlines()
        hint = first_err[0] if first_err else None
        return CheckResult(
            "vllm version",
            "warn",
            f"exit code {result.returncode}",
            hint=hint,
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    candidates = [line for line in lines if not _LOG_LINE_PREFIX.match(line)]
    version = candidates[0] if candidates else (lines[0] if lines else "(no output)")
    return CheckResult("vllm version", "ok", version)


def check_hf_token(project: Project) -> CheckResult:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return CheckResult("HF_TOKEN", "ok", "set in shell")
    dotenv = service.load_dotenv(project)
    if dotenv.get("HF_TOKEN") or dotenv.get("HUGGING_FACE_HUB_TOKEN"):
        return CheckResult("HF_TOKEN", "ok", "set in .env")
    return CheckResult(
        "HF_TOKEN",
        "warn",
        "not set",
        hint="needed only for gated models (Llama, Gemma, ...); set in .env or shell",
    )


def check_runtime_writable(project: Project) -> CheckResult:
    runtime = project.resolve("runtime")
    if not runtime.exists():
        return CheckResult(
            "runtime/ directory",
            "warn",
            "does not exist yet",
            hint="created automatically on first model start",
        )
    if not os.access(runtime, os.W_OK):
        return CheckResult(
            "runtime/ directory",
            "fail",
            f"{runtime} is not writable",
            hint=f"check ownership/permissions on {runtime}",
        )
    return CheckResult("runtime/ directory", "ok", str(runtime))


def check_catalog(project: Project) -> CheckResult:
    try:
        entries = service.list_catalog_entries(project)
    except Exception as exc:
        return CheckResult("Catalog", "fail", f"cannot read: {exc}")
    if not entries:
        return CheckResult(
            "Catalog",
            "warn",
            "no models declared yet",
            hint="`vllmops create-model` to add one",
        )
    broken = [e for e in entries if e.is_broken]
    if broken:
        names = ", ".join(e.name for e in broken[:3])
        more = f" +{len(broken) - 3} more" if len(broken) > 3 else ""
        return CheckResult(
            "Catalog",
            "warn",
            f"{len(entries)} model(s), {len(broken)} broken: {names}{more}",
            hint="`vllmops validate` for details",
        )
    return CheckResult("Catalog", "ok", f"{len(entries)} model(s) valid")


def check_port_conflicts(project: Project) -> CheckResult:
    try:
        entries = service.list_catalog_entries(project)
    except Exception:
        return CheckResult("Port conflicts", "warn", "could not read catalog")
    by_port: dict[int, list[str]] = {}
    for entry in entries:
        if entry.is_broken or entry.status is None or entry.status.metrics_port is None:
            continue
        by_port.setdefault(entry.status.metrics_port, []).append(entry.name)
    conflicts = {port: names for port, names in by_port.items() if len(names) > 1}
    if conflicts:
        items = "; ".join(f"port {port}: {', '.join(names)}" for port, names in conflicts.items())
        return CheckResult(
            "Port conflicts",
            "warn",
            items,
            hint="only an issue if those models run concurrently",
        )
    return CheckResult("Port conflicts", "ok", "no duplicates")


def check_gpu_conflicts(project: Project) -> CheckResult:
    try:
        entries = service.list_catalog_entries(project)
    except Exception:
        return CheckResult("GPU conflicts", "warn", "could not read catalog")
    gpus_per_model: dict[str, list[int]] = {}
    for entry in entries:
        if entry.is_broken:
            continue
        try:
            model = load_model_file(entry.yaml_path)
        except Exception:
            continue
        gpus_per_model[entry.name] = gpu_module.gpus_for_model(model.env)
    names = list(gpus_per_model)
    shared: list[tuple[str, str, int]] = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            common = set(gpus_per_model[n1]) & set(gpus_per_model[n2])
            for g in sorted(common):
                shared.append((n1, n2, g))
    if shared:
        items = "; ".join(f"{a} ↔ {b} on GPU {g}" for a, b, g in shared[:3])
        more = f" +{len(shared) - 3} more" if len(shared) > 3 else ""
        return CheckResult(
            "GPU conflicts",
            "warn",
            f"{len(shared)} overlap(s): {items}{more}",
            hint="only an issue if those models run concurrently",
        )
    return CheckResult("GPU conflicts", "ok", "no overlapping CUDA_VISIBLE_DEVICES")


def run_checks(project: Project | None = None) -> list[CheckResult]:
    """Run all environment checks. Project-scoped ones are skipped if no project."""
    results: list[CheckResult] = [check_python_version(), check_nvidia_smi()]

    project_check = check_project_root()
    results.append(project_check)
    if project_check.status == "fail":
        return results

    if project is None:
        try:
            project = load_project()
        except Exception as exc:
            results.append(
                CheckResult(
                    "Project config",
                    "fail",
                    f"could not load: {exc}",
                    hint="check .vllmops/config.yaml syntax",
                )
            )
            return results

    results.extend(
        [
            check_venv(project),
            check_vllm_executable(project),
            check_vllm_version(project),
            check_hf_token(project),
            check_runtime_writable(project),
            check_catalog(project),
            check_port_conflicts(project),
            check_gpu_conflicts(project),
        ]
    )
    return results
