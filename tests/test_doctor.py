"""Tests for vllmops.doctor environment diagnostic checks."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from tests.conftest import sleeper_payload, write_model_yaml
from vllmops import doctor
from vllmops.project import Project


def test_check_python_version_current_passes() -> None:
    """The CI uses Python >=3.10, so this should always be ok."""
    result = doctor.check_python_version()
    assert result.status == "ok"
    assert sys.version_info >= (3, 10)


def test_check_project_root_fails_outside_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = doctor.check_project_root()
    assert result.status == "fail"
    assert "no .vllmops/config.yaml" in result.detail


def test_check_project_root_ok_inside_project(monkeypatch: pytest.MonkeyPatch, project: Project) -> None:
    monkeypatch.chdir(project.root)
    result = doctor.check_project_root()
    assert result.status == "ok"


def test_check_venv_missing(project: Project) -> None:
    """A freshly initialized project has no .venv yet."""
    result = doctor.check_venv(project)
    assert result.status == "fail"
    assert "does not exist" in result.detail


def test_check_venv_present(project: Project) -> None:
    (project.root / ".venv").mkdir(parents=True, exist_ok=True)
    result = doctor.check_venv(project)
    assert result.status == "ok"


def test_check_vllm_executable_missing(project: Project) -> None:
    result = doctor.check_vllm_executable(project)
    assert result.status == "fail"


def test_check_vllm_executable_present(project: Project) -> None:
    """Plant a fake vllm at .venv/bin/vllm (or .venv/Scripts/vllm.exe on Windows)."""
    if sys.platform == "win32":
        fake = project.root / ".venv" / "Scripts" / "vllm.exe"
    else:
        fake = project.root / ".venv" / "bin" / "vllm"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("#!/bin/sh\nexit 0\n")
    result = doctor.check_vllm_executable(project)
    assert result.status == "ok"


def test_check_vllm_version_filters_log_prefixed_lines(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lines starting with a log-level prefix are skipped when picking the version."""
    if sys.platform == "win32":
        fake = project.root / ".venv" / "Scripts" / "vllm.exe"
    else:
        fake = project.root / ".venv" / "bin" / "vllm"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("dummy")

    class FakeCompleted:
        returncode = 0
        stdout = "WARNING 06-22 18:01:55 Using 'pin_memory=False'.\n0.7.3\n"
        stderr = ""

    def fake_run(*_args: object, **_kwargs: object) -> FakeCompleted:
        return FakeCompleted()

    monkeypatch.setattr("vllmops.doctor.subprocess.run", fake_run)
    result = doctor.check_vllm_version(project)
    assert result.status == "ok"
    assert result.detail == "0.7.3"


def test_check_hf_token_from_shell_env(monkeypatch: pytest.MonkeyPatch, project: Project) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxxxx")
    result = doctor.check_hf_token(project)
    assert result.status == "ok"
    assert "shell" in result.detail


def test_check_hf_token_from_dotenv(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    (project.root / ".env").write_text("HF_TOKEN=hf_yyy\n")
    result = doctor.check_hf_token(project)
    assert result.status == "ok"
    assert ".env" in result.detail


def test_check_hf_token_warns_when_missing(monkeypatch: pytest.MonkeyPatch, project: Project) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    result = doctor.check_hf_token(project)
    assert result.status == "warn"


def test_check_catalog_empty(project: Project) -> None:
    result = doctor.check_catalog(project)
    assert result.status == "warn"
    assert "no models" in result.detail


def test_check_catalog_all_valid(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    result = doctor.check_catalog(project)
    assert result.status == "ok"
    assert "2 model" in result.detail


def test_check_catalog_reports_broken(project: Project) -> None:
    write_model_yaml(project, "ok", sleeper_payload("ok", port=18001))
    (project.models_dir / "bad.yaml").write_text("name: bad\nbogus_field: 1\n")
    result = doctor.check_catalog(project)
    assert result.status == "warn"
    assert "broken" in result.detail


def test_check_port_conflicts_no_conflicts(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    result = doctor.check_port_conflicts(project)
    assert result.status == "ok"


def test_check_port_conflicts_warns_on_duplicates(project: Project) -> None:
    """Duplicate ports across configs are now allowed; doctor surfaces them as
    a warning since runtime guarantees only one model binds at a time."""
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18001))
    result = doctor.check_port_conflicts(project)
    assert result.status == "warn"
    assert "18001" in result.detail
    assert "a" in result.detail and "b" in result.detail


def test_check_runtime_writable_warns_when_missing(project: Project) -> None:
    """runtime/ is created on first model start; absence is a warning, not a fail."""
    shutil.rmtree(project.root / "runtime", ignore_errors=True)
    result = doctor.check_runtime_writable(project)
    assert result.status == "warn"


def test_check_runtime_writable_ok(project: Project) -> None:
    (project.root / "runtime").mkdir(parents=True, exist_ok=True)
    result = doctor.check_runtime_writable(project)
    assert result.status == "ok"


def test_check_gpu_conflicts_no_models(project: Project) -> None:
    result = doctor.check_gpu_conflicts(project)
    assert result.status == "ok"
    assert "no overlap" in result.detail


def test_check_gpu_conflicts_detects_overlap(project: Project) -> None:
    a = sleeper_payload("a", port=18001)
    a["env"] = {"CUDA_VISIBLE_DEVICES": "0,1"}
    b = sleeper_payload("b", port=18002)
    b["env"] = {"CUDA_VISIBLE_DEVICES": "1,2"}
    write_model_yaml(project, "a", a)
    write_model_yaml(project, "b", b)
    result = doctor.check_gpu_conflicts(project)
    assert result.status == "warn"
    assert "GPU 1" in result.detail


def test_run_checks_no_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Outside a project, only ambient checks run plus the project-root fail."""
    monkeypatch.chdir(tmp_path)
    results = doctor.run_checks()
    names = [r.name for r in results]
    assert "Python version" in names
    assert "nvidia-smi" in names
    assert "Project root" in names
    # No project-scoped checks
    assert ".venv directory" not in names


def test_run_checks_full_pipeline(monkeypatch: pytest.MonkeyPatch, project: Project) -> None:
    monkeypatch.chdir(project.root)
    results = doctor.run_checks(project)
    names = [r.name for r in results]
    # Ambient + project-scoped checks all present
    for expected in [
        "Python version",
        "nvidia-smi",
        "Project root",
        ".venv directory",
        "vllm executable",
        "HF_TOKEN",
        "Catalog",
        "Port conflicts",
        "GPU conflicts",
    ]:
        assert expected in names, f"missing check: {expected}"
