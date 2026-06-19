"""Tests for vllmctl.service.

Most tests here are platform-agnostic and run on Windows. The wait_for_ready
tests need a real PID, but they only signal-probe (kill 0), no fork required ,
so they also run on Windows. Real spawn/terminate is exercised in test_lifecycle.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.conftest import fast_exit_payload, posix_only, sleeper_payload, write_model_yaml
from vllmctl import service
from vllmctl.project import Project
from vllmctl.service import (
    ModelNotRunningError,
    ModelStartupFailedError,
    ModelStartupTimeoutError,
    UnknownModelError,
    VllmExecutableNotFoundError,
    build_runtime_env,
    check_vllm_available,
    health_url,
    load_dotenv,
    parse_dotenv,
    probe_health,
    resolve_vllm_executable,
    wait_for_ready,
)

# --- parse_dotenv ---


def test_parse_dotenv_basic() -> None:
    parsed = parse_dotenv("FOO=bar\nBAZ=qux")
    assert parsed == {"FOO": "bar", "BAZ": "qux"}


def test_parse_dotenv_strips_quotes() -> None:
    parsed = parse_dotenv('FOO="quoted value"\nBAR=\'single\'')
    assert parsed == {"FOO": "quoted value", "BAR": "single"}


def test_parse_dotenv_handles_export_prefix() -> None:
    parsed = parse_dotenv("export FOO=bar")
    assert parsed == {"FOO": "bar"}


def test_parse_dotenv_skips_comments_and_blanks() -> None:
    parsed = parse_dotenv("# comment\n\nFOO=bar\n   # also comment\n")
    assert parsed == {"FOO": "bar"}


def test_parse_dotenv_rejects_invalid_keys() -> None:
    parsed = parse_dotenv("1BAD=x\n=novalue\nGOOD=ok")
    assert parsed == {"GOOD": "ok"}


def test_parse_dotenv_keeps_empty_value() -> None:
    parsed = parse_dotenv("FOO=")
    assert parsed == {"FOO": ""}


def test_parse_dotenv_ignores_lines_without_equals() -> None:
    parsed = parse_dotenv("FOO=bar\njust_a_word\nBAZ=qux")
    assert parsed == {"FOO": "bar", "BAZ": "qux"}


# --- load_dotenv ---


def test_load_dotenv_returns_empty_when_missing(project: Project) -> None:
    assert load_dotenv(project) == {}


def test_load_dotenv_reads_root_env(project: Project) -> None:
    (project.root / ".env").write_text("HF_TOKEN=hf_xxx\nFOO=bar\n", encoding="utf-8")
    assert load_dotenv(project) == {"HF_TOKEN": "hf_xxx", "FOO": "bar"}


# --- build_runtime_env ---


def test_runtime_env_precedence_dotenv_shell_model(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project.root / ".env").write_text(
        "DOTENV_ONLY=from_dotenv\nSHARED=dotenv_wins\nMODEL_OVERRIDE=dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SHARED", "shell_wins")
    monkeypatch.setenv("MODEL_OVERRIDE", "shell")
    monkeypatch.setenv("SHELL_ONLY", "from_shell")
    model_env = {"MODEL_OVERRIDE": "model_wins", "MODEL_ONLY": "from_model"}

    env = build_runtime_env(project, model_env)

    assert env["DOTENV_ONLY"] == "from_dotenv"
    assert env["SHARED"] == "shell_wins"  # shell beats dotenv
    assert env["MODEL_OVERRIDE"] == "model_wins"  # model beats both
    assert env["SHELL_ONLY"] == "from_shell"
    assert env["MODEL_ONLY"] == "from_model"


def test_runtime_env_sets_pythonunbuffered_by_default(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    env = build_runtime_env(project, {})
    assert env["PYTHONUNBUFFERED"] == "1"


def test_runtime_env_pythonunbuffered_is_overridable(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline PYTHONUNBUFFERED=1 is just a default; user input wins."""
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    env = build_runtime_env(project, {})
    assert env["PYTHONUNBUFFERED"] == "0"
    # Model YAML beats shell too
    env2 = build_runtime_env(project, {"PYTHONUNBUFFERED": "x"})
    assert env2["PYTHONUNBUFFERED"] == "x"


def test_runtime_env_no_venv_leaves_path_alone(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a project venv, PATH and VIRTUAL_ENV are not touched."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    env = build_runtime_env(project, {})
    assert env["PATH"] == "/usr/bin:/bin"
    assert "VIRTUAL_ENV" not in env


def test_runtime_env_venv_prepends_path_and_sets_virtual_env(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When .venv exists, its bin/Scripts dir is prepended to PATH."""
    venv_bin = project.root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/usr/bin")
    env = build_runtime_env(project, {})
    assert env["PATH"].startswith(f"{venv_bin}{os.pathsep}")
    assert env["PATH"].endswith("/usr/bin")
    assert env["VIRTUAL_ENV"] == str(project.root / ".venv")


def test_runtime_env_venv_drops_pythonhome(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PYTHONHOME is removed when a project venv is active, it would break the venv."""
    venv_bin = project.root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setenv("PYTHONHOME", "/some/path")
    env = build_runtime_env(project, {})
    assert "PYTHONHOME" not in env


# --- health URL + probe ---


def test_health_url() -> None:
    assert health_url("127.0.0.1", 8001) == "http://127.0.0.1:8001/health"


def test_probe_health_success(mock_health_server: tuple[int, object]) -> None:
    port, _ = mock_health_server
    assert probe_health(f"http://127.0.0.1:{port}/health") is True


def test_probe_health_unreachable_port_returns_false() -> None:
    # Port 9 is the discard service; almost never has a listener
    assert probe_health("http://127.0.0.1:9/health", timeout=0.3) is False


def test_probe_health_bogus_url_returns_false() -> None:
    assert probe_health("http://invalid.invalid:1/health", timeout=0.3) is False


# --- runtime paths + status ---


def test_runtime_paths_under_project_runtime_dir(project: Project) -> None:
    paths = service.runtime_paths_for(project, "m1")
    assert paths.pid_path == project.root / "runtime" / "pids" / "m1.pid"
    assert paths.log_path == project.root / "runtime" / "logs" / "m1.log"


def test_status_no_pid_file_reports_stopped(project: Project) -> None:
    status = service.get_model_status(project, "missing")
    assert status.running is False
    assert status.pid is None
    assert status.stale_pid_file is False


@posix_only
def test_status_stale_pid_detected(project: Project) -> None:
    """is_alive() relies on POSIX `kill(pid, 0)` semantics; Windows can't reliably
    distinguish a non-existent PID from a permission error."""
    paths = service.runtime_paths_for(project, "ghost")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text("99999999")
    status = service.get_model_status(project, "ghost")
    assert status.running is False
    assert status.stale_pid_file is True


def test_status_running_when_pid_alive(project: Project) -> None:
    paths = service.runtime_paths_for(project, "self")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text(str(os.getpid()))
    status = service.get_model_status(project, "self")
    assert status.running is True
    assert status.pid == os.getpid()


def test_list_model_statuses_empty_catalog(project: Project) -> None:
    assert service.list_model_statuses(project) == []


def test_list_model_statuses_iterates_catalog(project: Project) -> None:
    free = 18001
    write_model_yaml(project, "a", sleeper_payload("a", port=free))
    write_model_yaml(project, "b", sleeper_payload("b", port=free + 1))
    statuses = service.list_model_statuses(project)
    assert {s.name for s in statuses} == {"a", "b"}


# --- list_catalog_entries (lenient) ---


def test_list_catalog_entries_returns_valid_models(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    entries = service.list_catalog_entries(project)
    assert {e.name for e in entries} == {"a", "b"}
    assert all(not e.is_broken for e in entries)
    assert all(e.status is not None for e in entries)


def test_list_catalog_entries_marks_invalid_yaml_as_broken(project: Project) -> None:
    """A YAML that fails pydantic validation becomes a broken entry instead of
    crashing the listing for the other models."""
    write_model_yaml(project, "ok", sleeper_payload("ok", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nthis_field_does_not_exist: 42\n", encoding="utf-8")
    entries = service.list_catalog_entries(project)
    by_name = {e.name: e for e in entries}
    assert "ok" in by_name and not by_name["ok"].is_broken
    assert "bad" in by_name and by_name["bad"].is_broken
    assert by_name["bad"].error  # populated with a meaningful message


def test_list_catalog_entries_marks_yaml_syntax_errors(project: Project) -> None:
    """Even a YAML that won't parse at all should produce a broken entry, not raise."""
    write_model_yaml(project, "ok", sleeper_payload("ok", port=18001))
    bad = project.models_dir / "syntax.yaml"
    bad.write_text("name: tiny\n  bad: indent: this\n", encoding="utf-8")
    entries = service.list_catalog_entries(project)
    syntax_entry = next(e for e in entries if e.name == "syntax")
    assert syntax_entry.is_broken
    assert syntax_entry.error


def test_list_catalog_entries_flags_duplicate_name(project: Project) -> None:
    """Two YAML files claiming the same model name → second becomes broken."""
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    duplicate = project.models_dir / "z_other.yaml"
    payload = sleeper_payload("a", port=18002)  # same name, different file
    import yaml as _yaml  # noqa: PLC0415

    duplicate.write_text(_yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    entries = service.list_catalog_entries(project)
    # The second one (sorted lexicographically) is the broken duplicate
    broken = [e for e in entries if e.is_broken]
    assert len(broken) == 1
    assert "duplicate model name" in (broken[0].error or "")


def test_list_catalog_entries_flags_duplicate_port(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18001))  # same port
    entries = service.list_catalog_entries(project)
    broken = [e for e in entries if e.is_broken]
    assert len(broken) == 1
    assert "duplicate metrics port" in (broken[0].error or "")


def test_list_catalog_entries_empty_when_dir_missing(project: Project) -> None:
    """No models dir at all (project freshly created without create-model) → []."""
    import shutil as _shutil  # noqa: PLC0415

    _shutil.rmtree(project.models_dir, ignore_errors=True)
    assert service.list_catalog_entries(project) == []


def test_list_catalog_entries_yaml_path_is_actual_file(project: Project) -> None:
    """The entry's yaml_path points to the file we found, regardless of model name."""
    write_model_yaml(project, "a_file", sleeper_payload("custom-name", port=18001))
    entries = service.list_catalog_entries(project)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.yaml_path.name == "a_file.yaml"
    # Name comes from the YAML body when valid
    assert entry.name == "custom-name"


def test_valid_model_keeps_metrics_port_even_when_sibling_yaml_is_broken(
    project: Project,
) -> None:
    """A broken YAML in the catalog used to wipe metrics_port for every other
    model because the strict load failed, make sure that regression stays gone."""
    write_model_yaml(project, "good", sleeper_payload("good", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nbogus_field: 1\n", encoding="utf-8")

    entries = service.list_catalog_entries(project)
    by_name = {e.name: e for e in entries}
    good = by_name["good"]
    assert not good.is_broken
    assert good.status is not None
    assert good.status.metrics_port == 18001


def test_lookup_metrics_port_survives_broken_siblings(project: Project) -> None:
    """The CLI/standalone path also needs to keep working when one YAML is bad."""
    write_model_yaml(project, "good", sleeper_payload("good", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nbogus_field: 1\n", encoding="utf-8")

    status = service.get_model_status(project, "good")
    assert status.metrics_port == 18001


def test_build_command_args_survives_broken_sibling(project: Project) -> None:
    """A broken YAML elsewhere in the catalog must not block start/restart of
    a valid model, `build_command_args` was using strict `load_catalog` which
    aborted everything."""
    write_model_yaml(project, "good", sleeper_payload("good", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nbogus_field: 1\n", encoding="utf-8")

    # Should NOT raise. Returns the good model's argv and env.
    args, env = service.build_command_args(project, "good")
    assert len(args) >= 3  # executable + subcommand + model
    assert isinstance(env, dict)


def test_build_command_string_survives_broken_sibling(project: Project) -> None:
    write_model_yaml(project, "good", sleeper_payload("good", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nbogus_field: 1\n", encoding="utf-8")

    # Should NOT raise. The rendered command exists and contains the executable.
    rendered = service.build_command_string(project, "good")
    assert rendered  # non-empty string
    assert "import time" in rendered  # the sleeper payload's "model" string


def test_build_command_args_unknown_model_still_raises_with_broken_sibling(
    project: Project,
) -> None:
    """A broken sibling must not turn the 'unknown model' error into 'oh wait,
    let's spawn nothing successfully'."""
    write_model_yaml(project, "good", sleeper_payload("good", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nbogus_field: 1\n", encoding="utf-8")

    with pytest.raises(UnknownModelError):
        service.build_command_args(project, "nonexistent")


def test_find_model_returns_none_when_missing(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    assert service.find_model(project, "missing-model") is None


def test_find_model_falls_back_when_filename_does_not_match_yaml_name(
    project: Project,
) -> None:
    """`vllmctl create-model` produces `<name>.yaml`, but a hand-edited workspace
    can have any filename, find_model should still locate the model by its
    YAML body."""
    write_model_yaml(project, "weirdfile", sleeper_payload("real-name", port=18001))
    model = service.find_model(project, "real-name")
    assert model is not None
    assert model.name == "real-name"


def test_find_model_skips_broken_files_in_fallback_scan(project: Project) -> None:
    write_model_yaml(project, "weirdfile", sleeper_payload("real-name", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: real-name\nbogus_field: 1\n", encoding="utf-8")
    # Even though `bad.yaml` claims `name: real-name`, it's invalid → skipped.
    # The fallback scan keeps going and finds the valid one.
    model = service.find_model(project, "real-name")
    assert model is not None
    assert model.name == "real-name"


def test_list_catalog_entries_picks_up_runtime_yaml_breakage(project: Project) -> None:
    """Editing a YAML to a broken state mid-session must flip the entry to broken
    on the very next call, no caching, no stale state."""
    write_model_yaml(project, "tiny", sleeper_payload("tiny", port=18001))
    first = service.list_catalog_entries(project)
    assert len(first) == 1
    assert not first[0].is_broken

    # Now corrupt the file in place, simulating an in-TUI edit gone wrong.
    yaml_path = first[0].yaml_path
    yaml_path.write_text("name: tiny\nthis_field_does_not_exist: 1\n", encoding="utf-8")

    second = service.list_catalog_entries(project)
    assert len(second) == 1
    assert second[0].is_broken
    assert second[0].error  # populated

    # And we can heal it the same way, without restarting anything.
    write_model_yaml(project, "tiny", sleeper_payload("tiny", port=18001))
    third = service.list_catalog_entries(project)
    assert not third[0].is_broken


def test_build_command_args_unknown_model_raises(project: Project) -> None:
    write_model_yaml(project, "real", sleeper_payload("real", port=18001))
    with pytest.raises(UnknownModelError):
        service.build_command_args(project, "nope")


def test_build_command_string_unknown_model_raises(project: Project) -> None:
    write_model_yaml(project, "real", sleeper_payload("real", port=18001))
    with pytest.raises(UnknownModelError):
        service.build_command_string(project, "nope")


def test_build_command_string_includes_env_prefix(project: Project) -> None:
    payload = sleeper_payload("m", port=18001)
    payload["env"] = {"FOO": "bar baz"}
    write_model_yaml(project, "m", payload)
    rendered = service.build_command_string(project, "m")
    assert rendered.startswith("FOO='bar baz' ") or rendered.startswith("FOO=bar\\ baz ")


# --- tail_log ---


def test_tail_log_missing_file(project: Project) -> None:
    assert service.tail_log(project, "nope") == []


def test_tail_log_returns_last_n(project: Project) -> None:
    paths = service.runtime_paths_for(project, "m")
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.log_path.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    tail = service.tail_log(project, "m", lines=5)
    assert tail == [f"line {i}" for i in range(95, 100)]


def test_tail_log_zero_or_negative_returns_empty(project: Project) -> None:
    paths = service.runtime_paths_for(project, "m")
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.log_path.write_text("a\nb\nc\n")
    assert service.tail_log(project, "m", lines=0) == []
    assert service.tail_log(project, "m", lines=-1) == []


def test_tail_log_more_lines_than_file_has(project: Project) -> None:
    paths = service.runtime_paths_for(project, "m")
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.log_path.write_text("only one\nand two\n")
    tail = service.tail_log(project, "m", lines=100)
    assert tail == ["only one", "and two"]


# --- wait_for_ready (PID-based, no fork required) ---


def test_wait_for_ready_no_pid_file_raises(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    with pytest.raises(ModelNotRunningError):
        wait_for_ready(project, "m", timeout=1.0)


@posix_only
def test_wait_for_ready_dead_pid_raises_startup_failed(project: Project) -> None:
    """Relies on POSIX kill(pid, 0) semantics to detect a non-existent PID."""
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    paths = service.runtime_paths_for(project, "m")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text("99999999")  # not running
    with pytest.raises(ModelStartupFailedError):
        wait_for_ready(project, "m", timeout=1.0, interval=0.05)


def test_wait_for_ready_returns_when_health_responds(
    project: Project, mock_health_server: tuple[int, object]
) -> None:
    port, _ = mock_health_server
    write_model_yaml(project, "m", sleeper_payload("m", port=port))
    paths = service.runtime_paths_for(project, "m")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text(str(os.getpid()))

    status = wait_for_ready(project, "m", timeout=5.0, interval=0.1)
    assert status.running is True
    assert status.metrics_port == port


def test_wait_for_ready_times_out_when_no_server(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=9))  # discard port, no listener
    paths = service.runtime_paths_for(project, "m")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text(str(os.getpid()))

    with pytest.raises(ModelStartupTimeoutError):
        wait_for_ready(project, "m", timeout=0.5, interval=0.1)


def test_wait_for_ready_no_port_raises_value_error(project: Project) -> None:
    payload = sleeper_payload("m", port=18001, with_metrics=False)
    write_model_yaml(project, "m", payload)
    paths = service.runtime_paths_for(project, "m")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text(str(os.getpid()))

    with pytest.raises(ValueError, match="no HTTP port"):
        wait_for_ready(project, "m", timeout=0.5)


# --- resolve_vllm_executable + check_vllm_available ---


def test_resolve_vllm_falls_back_when_no_venv(project: Project) -> None:
    assert resolve_vllm_executable(project, "vllm") == "vllm"


def test_resolve_vllm_uses_project_venv_when_present(project: Project) -> None:
    fake = project.root / ".venv" / "bin" / "vllm"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\nexit 0\n")
    assert resolve_vllm_executable(project, "vllm") == str(fake)


def test_resolve_vllm_respects_explicit_path(project: Project) -> None:
    """A user-specified executable (anything that's not the default 'vllm')
    must be returned verbatim, even if a project venv would otherwise win."""
    fake = project.root / ".venv" / "bin" / "vllm"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\nexit 0\n")
    assert resolve_vllm_executable(project, "/opt/custom/vllm") == "/opt/custom/vllm"
    assert resolve_vllm_executable(project, "my-vllm") == "my-vllm"


def test_check_vllm_available_raises_for_missing_default(project: Project) -> None:
    """Without a project venv and without `vllm` on PATH, we expect a friendly error."""
    if shutil.which("vllm") is not None:
        pytest.skip("test environment unexpectedly has `vllm` on PATH")
    with pytest.raises(VllmExecutableNotFoundError, match="vLLM not found"):
        check_vllm_available(project, "vllm")


def test_check_vllm_available_accepts_existing_absolute_path(
    project: Project, tmp_path: Path
) -> None:
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\nexit 0\n")
    check_vllm_available(project, str(fake))  # must not raise


def test_check_vllm_available_rejects_missing_absolute_path(
    project: Project, tmp_path: Path
) -> None:
    with pytest.raises(VllmExecutableNotFoundError, match="not found at"):
        check_vllm_available(project, str(tmp_path / "nope" / "vllm"))


# --- create_model ---


def test_create_model_writes_yaml(project: Project) -> None:
    result = service.create_model(
        project, name="newm", hf_model="hf/x", gpus="0", port=18001
    )
    assert result.destination.is_file()
    assert result.destination.name == "newm.yaml"


def test_create_model_refuses_existing_without_force(project: Project) -> None:
    service.create_model(project, name="dup", hf_model="hf/x", gpus="0", port=18001)
    with pytest.raises(service.ModelAlreadyExistsError):
        service.create_model(project, name="dup", hf_model="hf/x", gpus="0", port=18002)


def test_create_model_force_overwrites(project: Project) -> None:
    service.create_model(project, name="dup", hf_model="hf/x", gpus="0", port=18001)
    result = service.create_model(
        project, name="dup", hf_model="hf/y", gpus="0", port=18002, force=True
    )
    assert result.model.vllm.model == "hf/y"


def test_fast_exit_payload_is_valid_yaml(project: Project, tmp_path: Path) -> None:
    """Sanity check that the helper produces a config the catalog will load."""
    write_model_yaml(project, "fail", fast_exit_payload("fail", port=18001))
    catalog = service.load_catalog_for(project)
    assert catalog.get("fail") is not None
