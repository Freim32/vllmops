"""Tests for vllmctl.service.

Most tests here are platform-agnostic and run on Windows. The wait_for_ready
tests need a real PID, but they only signal-probe (kill 0), no fork required ,
so they also run on Windows. Real spawn/terminate is exercised in test_lifecycle.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tests.conftest import (
    MockCompletionsHandler,
    fast_exit_payload,
    posix_only,
    sleeper_payload,
    write_model_yaml,
)
from vllmctl import service
from vllmctl.config import ModelConfig, VllmConfig
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
    parsed = parse_dotenv("FOO=\"quoted value\"\nBAR='single'")
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


def test_runtime_env_precedence_dotenv_shell_model(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_runtime_env_sets_pythonunbuffered_by_default(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    env = build_runtime_env(project, {})
    assert env["PYTHONUNBUFFERED"] == "1"


def test_runtime_env_pythonunbuffered_is_overridable(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """The baseline PYTHONUNBUFFERED=1 is just a default; user input wins."""
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    env = build_runtime_env(project, {})
    assert env["PYTHONUNBUFFERED"] == "0"
    # Model YAML beats shell too
    env2 = build_runtime_env(project, {"PYTHONUNBUFFERED": "x"})
    assert env2["PYTHONUNBUFFERED"] == "x"


def test_runtime_env_no_venv_leaves_path_alone(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a project venv, PATH and VIRTUAL_ENV are not touched."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    env = build_runtime_env(project, {})
    assert env["PATH"] == "/usr/bin:/bin"
    assert "VIRTUAL_ENV" not in env


def test_runtime_env_venv_prepends_path_and_sets_virtual_env(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """When .venv exists, its bin/Scripts dir is prepended to PATH."""
    venv_bin = project.root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/usr/bin")
    env = build_runtime_env(project, {})
    assert env["PATH"].startswith(f"{venv_bin}{os.pathsep}")
    assert env["PATH"].endswith("/usr/bin")
    assert env["VIRTUAL_ENV"] == str(project.root / ".venv")


def test_runtime_env_venv_drops_pythonhome(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_list_catalog_entries_allows_duplicate_port(project: Project) -> None:
    """Duplicate ports are now allowed at config time; conflict is a runtime concern."""
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18001))
    entries = service.list_catalog_entries(project)
    assert {e.name for e in entries} == {"a", "b"}
    assert all(not e.is_broken for e in entries)


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


# --- action availability predicates ---


def _entry(
    name: str,
    *,
    broken: bool = False,
    running: bool = False,
    metrics_port: int | None = 8001,
    stale: bool = False,
) -> service.CatalogEntry:
    """Build a CatalogEntry directly without going through YAML loading."""
    if broken:
        return service.CatalogEntry(
            name=name,
            yaml_path=Path(f"{name}.yaml"),
            status=None,
            error="invalid",
        )
    status = service.ModelStatus(
        name=name,
        running=running,
        pid=42 if running else None,
        log_path=Path("log"),
        pid_path=Path("pid"),
        metrics_port=metrics_port,
        stale_pid_file=stale,
    )
    return service.CatalogEntry(name=name, yaml_path=Path(f"{name}.yaml"), status=status)


def test_can_start_rejects_broken() -> None:
    assert service.can_start(_entry("a", broken=True)) is False


def test_can_start_rejects_running() -> None:
    assert service.can_start(_entry("a", running=True)) is False


def test_can_start_accepts_stopped() -> None:
    assert service.can_start(_entry("a", running=False)) is True


def test_can_stop_accepts_running() -> None:
    project = None  # not used for non-broken path
    assert service.can_stop(project, _entry("a", running=True)) is True  # type: ignore[arg-type]


def test_can_stop_rejects_stopped() -> None:
    project = None
    assert service.can_stop(project, _entry("a", running=False)) is False  # type: ignore[arg-type]


def test_can_stop_broken_with_live_pid(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken entry whose PID file points to a live process is stoppable."""
    write_model_yaml(project, "ok", sleeper_payload("ok", port=18001))
    bad = project.models_dir / "broken.yaml"
    bad.write_text("name: broken\nbogus_field: 1\n", encoding="utf-8")

    paths = service.runtime_paths_for(project, "broken")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text("12345", encoding="utf-8")

    monkeypatch.setattr(service.lifecycle, "is_alive", lambda pid: pid == 12345)

    broken_entry = next(e for e in service.list_catalog_entries(project) if e.name == "broken")
    assert broken_entry.is_broken
    assert service.can_stop(project, broken_entry) is True


def test_can_stop_broken_without_pid_file(project: Project) -> None:
    """A broken entry with no PID file cannot be stopped."""
    bad = project.models_dir / "broken.yaml"
    bad.write_text("name: broken\nbogus_field: 1\n", encoding="utf-8")

    broken_entry = next(e for e in service.list_catalog_entries(project) if e.name == "broken")
    assert service.can_stop(project, broken_entry) is False


def test_can_stop_broken_with_dead_pid(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken entry with a stale PID file (process gone) cannot be stopped."""
    bad = project.models_dir / "broken.yaml"
    bad.write_text("name: broken\nbogus_field: 1\n", encoding="utf-8")

    paths = service.runtime_paths_for(project, "broken")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text("99999", encoding="utf-8")

    monkeypatch.setattr(service.lifecycle, "is_alive", lambda pid: False)

    broken_entry = next(e for e in service.list_catalog_entries(project) if e.name == "broken")
    assert service.can_stop(project, broken_entry) is False


def test_can_restart_rejects_broken() -> None:
    assert service.can_restart(_entry("a", broken=True)) is False


def test_can_restart_accepts_running() -> None:
    assert service.can_restart(_entry("a", running=True)) is True


def test_can_restart_accepts_stopped() -> None:
    assert service.can_restart(_entry("a", running=False)) is True


def test_can_smoke_test_rejects_broken() -> None:
    assert service.can_smoke_test(_entry("a", broken=True)) is False


def test_can_smoke_test_rejects_stopped() -> None:
    assert service.can_smoke_test(_entry("a", running=False)) is False


def test_can_smoke_test_rejects_no_metrics_port() -> None:
    assert service.can_smoke_test(_entry("a", running=True, metrics_port=None)) is False


def test_can_smoke_test_accepts_running_with_port() -> None:
    assert service.can_smoke_test(_entry("a", running=True, metrics_port=8001)) is True


def test_can_copy_logs_rejects_when_log_missing(project: Project) -> None:
    write_model_yaml(project, "x", sleeper_payload("x", port=18001))
    entries = service.list_catalog_entries(project)
    assert service.can_copy_logs(project, entries[0]) is False


def test_can_copy_logs_rejects_when_log_empty(project: Project) -> None:
    write_model_yaml(project, "x", sleeper_payload("x", port=18001))
    paths = service.runtime_paths_for(project, "x")
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.log_path.write_text("", encoding="utf-8")
    entries = service.list_catalog_entries(project)
    assert service.can_copy_logs(project, entries[0]) is False


def test_can_copy_logs_accepts_when_log_has_content(project: Project) -> None:
    write_model_yaml(project, "x", sleeper_payload("x", port=18001))
    paths = service.runtime_paths_for(project, "x")
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.log_path.write_text("hello\n", encoding="utf-8")
    entries = service.list_catalog_entries(project)
    assert service.can_copy_logs(project, entries[0]) is True


# --- next_available_port ---


def test_next_available_port_starts_at_port_start_when_empty(project: Project) -> None:
    assert service.next_available_port(project) == 8001


def test_next_available_port_skips_used(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=8001))
    write_model_yaml(project, "b", sleeper_payload("b", port=8002))
    assert service.next_available_port(project) == 8003


def test_next_available_port_tolerates_broken_sibling(project: Project) -> None:
    """A broken YAML must not block port lookup."""
    write_model_yaml(project, "a", sleeper_payload("a", port=8001))
    bad = project.models_dir / "broken.yaml"
    bad.write_text("name: broken\nbogus_field: 1\n", encoding="utf-8")
    assert service.next_available_port(project) == 8002


def test_next_available_port_tolerates_duplicate_port(project: Project) -> None:
    """Two siblings sharing a port (now allowed) shouldn't crash port lookup."""
    write_model_yaml(project, "a", sleeper_payload("a", port=8002))
    write_model_yaml(project, "b", sleeper_payload("b", port=8002))
    assert service.next_available_port(project) == 8001


# --- list_profiles ---


def _set_profiles(project: Project, profiles: dict[str, list[str]]) -> Project:
    """Mutate the project config to add profiles, reload, return the new Project."""
    import yaml as _yaml  # noqa: PLC0415

    from vllmctl.project import load_project as _load_project  # noqa: PLC0415

    cfg_path = project.config_path
    raw = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["profiles"] = profiles
    cfg_path.write_text(_yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return _load_project(project.root)


def test_list_profiles_no_profiles_puts_everything_in_general(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    views = service.list_profiles(project)
    assert [v.name for v in views] == ["general"]
    assert {e.name for e in views[0].entries} == {"a", "b"}


def test_list_profiles_shared_model_appears_in_each_profile(project: Project) -> None:
    """A model listed in two profiles appears under BOTH and not in general."""
    write_model_yaml(project, "shared", sleeper_payload("shared", port=18001))
    write_model_yaml(project, "only_dev", sleeper_payload("only_dev", port=18002))
    project = _set_profiles(project, {"dev": ["shared", "only_dev"], "prod": ["shared"]})
    views = service.list_profiles(project)
    by_name = {v.name: v for v in views}
    assert {e.name for e in by_name["dev"].entries} == {"shared", "only_dev"}
    assert {e.name for e in by_name["prod"].entries} == {"shared"}
    assert by_name["general"].entries == []


def test_list_profiles_unassigned_models_go_to_general(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    write_model_yaml(project, "c", sleeper_payload("c", port=18003))
    project = _set_profiles(project, {"dev": ["a", "b"]})
    views = service.list_profiles(project)
    by_name = {v.name: v for v in views}
    assert {e.name for e in by_name["dev"].entries} == {"a", "b"}
    assert {e.name for e in by_name["general"].entries} == {"c"}


def test_list_profiles_preserves_declaration_order(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    project = _set_profiles(project, {"prod": ["b"], "dev": ["a"]})
    views = service.list_profiles(project)
    # general is always appended last
    assert [v.name for v in views] == ["prod", "dev", "general"]


def test_list_profiles_missing_member_is_reported(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    project = _set_profiles(project, {"dev": ["a", "ghost"]})
    views = service.list_profiles(project)
    dev = next(v for v in views if v.name == "dev")
    assert [e.name for e in dev.entries] == ["a"]
    assert dev.missing == ["ghost"]


def test_list_profiles_empty_profile_returned_for_show(project: Project) -> None:
    """list_profiles returns declared-but-empty profiles untrimmed; the consumer filters."""
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    project = _set_profiles(project, {"placeholder": [], "dev": ["a"]})
    views = service.list_profiles(project)
    placeholder = next(v for v in views if v.name == "placeholder")
    assert placeholder.entries == []
    assert placeholder.total_count == 0


def test_list_profiles_general_can_be_empty(project: Project) -> None:
    """When all catalog models belong to a profile, general is still listed but empty."""
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    project = _set_profiles(project, {"dev": ["a"]})
    views = service.list_profiles(project)
    general = next(v for v in views if v.name == "general")
    assert general.entries == []


def test_list_profiles_counts(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    views = service.list_profiles(project)
    general = next(v for v in views if v.name == "general")
    assert general.total_count == 2
    assert general.running_count == 0  # no spawn


def test_list_profiles_broken_member_is_still_in_profile(project: Project) -> None:
    """A broken YAML referenced in a profile still gets routed to that profile."""
    write_model_yaml(project, "ok", sleeper_payload("ok", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nbogus_field: 1\n", encoding="utf-8")
    project = _set_profiles(project, {"dev": ["ok", "bad"]})
    views = service.list_profiles(project)
    dev = next(v for v in views if v.name == "dev")
    assert {e.name for e in dev.entries} == {"ok", "bad"}
    assert any(e.is_broken for e in dev.entries)


# --- bulk profile operations (logic, no real spawn) ---


def test_start_profile_unknown_raises(project: Project) -> None:
    with pytest.raises(service.UnknownProfileError):
        service.start_profile(project, "ghost")


def test_stop_profile_unknown_raises(project: Project) -> None:
    with pytest.raises(service.UnknownProfileError):
        service.stop_profile(project, "ghost")


def test_restart_profile_unknown_raises(project: Project) -> None:
    with pytest.raises(service.UnknownProfileError):
        service.restart_profile(project, "ghost")


def test_start_profile_marks_broken_yaml_as_skipped(project: Project) -> None:
    """A broken YAML in the profile is skipped, not attempted."""
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nbogus_field: 1\n", encoding="utf-8")
    project = _set_profiles(project, {"dev": ["bad"]})
    result = service.start_profile(project, "dev")
    assert result.succeeded == []
    assert result.skipped == [("bad", "invalid YAML")]
    assert result.failed == []


def test_stop_profile_skips_not_running(project: Project) -> None:
    """All members already stopped → all in skipped, none in succeeded."""
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    project = _set_profiles(project, {"dev": ["a", "b"]})
    result = service.stop_profile(project, "dev")
    assert result.succeeded == []
    assert {name for name, _ in result.skipped} == {"a", "b"}
    assert all(reason == "not running" for _, reason in result.skipped)
    assert result.failed == []


def test_restart_profile_skips_broken_yaml(project: Project) -> None:
    """restart_profile cannot start a broken YAML; it's skipped."""
    write_model_yaml(project, "ok", sleeper_payload("ok", port=18001))
    bad = project.models_dir / "bad.yaml"
    bad.write_text("name: bad\nbogus_field: 1\n", encoding="utf-8")
    project = _set_profiles(project, {"dev": ["ok", "bad"]})
    result = service.restart_profile(project, "dev")
    assert ("bad", "invalid YAML") in result.skipped
    # "ok" will be attempted; on Windows/non-POSIX it may fail, but the
    # broken classification is what we're asserting here.


def test_start_profile_empty_profile_returns_empty_result(project: Project) -> None:
    """A profile that's not declared in catalog (only in config) → empty entries → no-op."""
    project = _set_profiles(project, {"dev": ["ghost"]})  # ghost not in catalog
    result = service.start_profile(project, "dev")
    assert result.total == 0
    assert result.profile == "dev"
    assert result.action == "start"


def test_bulk_result_total_counts_correctly(project: Project) -> None:
    """BulkResult.total sums succeeded + skipped + failed."""
    project = _set_profiles(project, {"dev": []})
    result = service.start_profile(project, "dev")
    assert result.total == 0


def test_start_profile_general_works_without_declaration(project: Project) -> None:
    """The synthetic 'general' profile must accept bulk operations even
    though it's never declared in config."""
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    result = service.start_profile(project, service.GENERAL_PROFILE)
    assert result.profile == service.GENERAL_PROFILE
    # On non-POSIX, the start will fail; what matters is the routing reaches it.
    assert result.total == 1


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


def test_wait_for_ready_returns_when_health_responds(project: Project, mock_health_server: tuple[int, object]) -> None:
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


def test_check_vllm_available_accepts_existing_absolute_path(project: Project, tmp_path: Path) -> None:
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\nexit 0\n")
    check_vllm_available(project, str(fake))  # must not raise


def test_check_vllm_available_rejects_missing_absolute_path(project: Project, tmp_path: Path) -> None:
    with pytest.raises(VllmExecutableNotFoundError, match="not found at"):
        check_vllm_available(project, str(tmp_path / "nope" / "vllm"))


# --- create_model ---


def test_create_model_writes_yaml(project: Project) -> None:
    result = service.create_model(project, name="newm", hf_model="hf/x", gpus="0", port=18001)
    assert result.destination.is_file()
    assert result.destination.name == "newm.yaml"


def test_create_model_refuses_existing_without_force(project: Project) -> None:
    service.create_model(project, name="dup", hf_model="hf/x", gpus="0", port=18001)
    with pytest.raises(service.ModelAlreadyExistsError):
        service.create_model(project, name="dup", hf_model="hf/x", gpus="0", port=18002)


def test_create_model_force_overwrites(project: Project) -> None:
    service.create_model(project, name="dup", hf_model="hf/x", gpus="0", port=18001)
    result = service.create_model(project, name="dup", hf_model="hf/y", gpus="0", port=18002, force=True)
    assert result.model.vllm.model == "hf/y"


def test_fast_exit_payload_is_valid_yaml(project: Project, tmp_path: Path) -> None:
    """Sanity check that the helper produces a config the catalog will load."""
    write_model_yaml(project, "fail", fast_exit_payload("fail", port=18001))
    catalog = service.load_catalog_for(project)
    assert catalog.get("fail") is not None


# --- smoke_test_model ---


def _make_model_cfg(args: dict | None = None, extra: list[str] | None = None) -> ModelConfig:
    """Tiny ModelConfig builder for _resolve_served_name unit tests."""
    return ModelConfig(
        name="x",
        vllm=VllmConfig(
            model="hf/foo",
            args=args or {},
            extra_args=extra or [],
        ),
    )


def test_resolve_served_name_falls_back_to_vllm_model() -> None:
    cfg = _make_model_cfg()
    assert service._resolve_served_name(cfg) == "hf/foo"


def test_resolve_served_name_honors_args_dict() -> None:
    cfg = _make_model_cfg(args={"--served-model-name": "alias"})
    assert service._resolve_served_name(cfg) == "alias"


def test_resolve_served_name_takes_first_when_list() -> None:
    cfg = _make_model_cfg(args={"--served-model-name": ["primary", "alt"]})
    assert service._resolve_served_name(cfg) == "primary"


def test_resolve_served_name_honors_extra_args() -> None:
    cfg = _make_model_cfg(extra=["--served-model-name", "from-extra"])
    assert service._resolve_served_name(cfg) == "from-extra"


def test_resolve_served_name_extra_args_without_value_falls_back() -> None:
    cfg = _make_model_cfg(extra=["--served-model-name"])
    assert service._resolve_served_name(cfg) == "hf/foo"


@posix_only
def test_stop_model_raises_unknown_model_when_name_missing(project: Project) -> None:
    """A typo'd name surfaces as UnknownModelError, not the misleading 'not running'."""
    with pytest.raises(service.UnknownModelError):
        service.stop_model(project, "does-not-exist")


@posix_only
def test_stop_model_tolerates_broken_yaml(project: Project) -> None:
    """A model whose YAML is broken at stop time is still nameable for PID-based stop."""
    bad = project.models_dir / "qwen3.yaml"
    bad.write_text("name: qwen3\nbogus_field: 1\n", encoding="utf-8")
    # YAML is broken but the name `qwen3` resolves via filename stem; no PID exists,
    # so we get ModelNotRunningError rather than UnknownModelError.
    with pytest.raises(service.ModelNotRunningError):
        service.stop_model(project, "qwen3")


@posix_only
def test_start_model_raises_port_conflict_when_other_running(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """If model B claims the same port as already-running model A, start fails."""
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18001))

    # Mark `a` as running by faking its catalog entry status.
    from vllmctl.service import CatalogEntry, ModelStatus  # noqa: PLC0415

    a_paths = service.runtime_paths_for(project, "a")
    real_list = service.list_catalog_entries

    def faked_list(proj: Project, config_dir: Path | None = None) -> list[CatalogEntry]:
        entries = real_list(proj, config_dir)
        patched: list[CatalogEntry] = []
        for entry in entries:
            if entry.name == "a" and entry.status is not None:
                patched.append(
                    CatalogEntry(
                        name=entry.name,
                        yaml_path=entry.yaml_path,
                        status=ModelStatus(
                            name="a",
                            running=True,
                            pid=99999,
                            log_path=a_paths.log_path,
                            pid_path=a_paths.pid_path,
                            metrics_port=18001,
                            stale_pid_file=False,
                        ),
                    )
                )
            else:
                patched.append(entry)
        return patched

    monkeypatch.setattr(service, "list_catalog_entries", faked_list)
    with pytest.raises(service.PortConflictError, match="18001"):
        service.start_model(project, "b")


def test_smoke_test_raises_for_unknown_model(project: Project) -> None:
    with pytest.raises(service.UnknownModelError):
        service.smoke_test_model(project, "does-not-exist")


def test_smoke_test_raises_when_model_broken(project: Project) -> None:
    bad = project.models_dir / "broken.yaml"
    bad.write_text("name: broken\nbogus_field: 1\n", encoding="utf-8")
    with pytest.raises(service.SmokeTestError, match="invalid YAML"):
        service.smoke_test_model(project, "broken")


def test_smoke_test_raises_when_model_not_running(project: Project) -> None:
    write_model_yaml(project, "idle", sleeper_payload("idle", port=18001))
    with pytest.raises(service.SmokeTestError, match="not running"):
        service.smoke_test_model(project, "idle")


@posix_only
def test_smoke_test_returns_response_on_success(
    project: Project,
    mock_completions: tuple[int, type[MockCompletionsHandler]],
) -> None:
    """End-to-end: spawn a sleeper, point the mock /v1/completions at its port,
    smoke test parses the response."""
    port, handler_class = mock_completions
    handler_class.response_status = 200
    handler_class.response_body = json.dumps({"choices": [{"text": " Hello there!"}]}).encode("utf-8")

    write_model_yaml(project, "mock", sleeper_payload("mock", port=port))
    service.start_model(project, "mock")
    try:
        result = service.smoke_test_model(project, "mock", host="127.0.0.1")
    finally:
        try:
            service.stop_model(project, "mock", timeout=5)
        except Exception:
            pass

    assert result.model == "mock"
    assert "Hello there" in result.response_text
    assert result.latency_seconds >= 0


@posix_only
def test_smoke_test_reports_http_error(
    project: Project,
    mock_completions: tuple[int, type[MockCompletionsHandler]],
) -> None:
    port, handler_class = mock_completions
    handler_class.response_status = 400
    handler_class.response_body = json.dumps({"error": {"message": "model 'wrong-name' not found"}}).encode("utf-8")

    write_model_yaml(project, "mock", sleeper_payload("mock", port=port))
    service.start_model(project, "mock")
    try:
        with pytest.raises(service.SmokeTestError, match="HTTP 400"):
            service.smoke_test_model(project, "mock", host="127.0.0.1")
    finally:
        try:
            service.stop_model(project, "mock", timeout=5)
        except Exception:
            pass


@posix_only
def test_smoke_test_reports_malformed_response(
    project: Project,
    mock_completions: tuple[int, type[MockCompletionsHandler]],
) -> None:
    port, handler_class = mock_completions
    handler_class.response_status = 200
    handler_class.response_body = b"not json at all"

    write_model_yaml(project, "mock", sleeper_payload("mock", port=port))
    service.start_model(project, "mock")
    try:
        with pytest.raises(service.SmokeTestError, match="unexpected response"):
            service.smoke_test_model(project, "mock", host="127.0.0.1")
    finally:
        try:
            service.stop_model(project, "mock", timeout=5)
        except Exception:
            pass
