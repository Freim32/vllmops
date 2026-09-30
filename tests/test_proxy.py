"""Tests for the LiteLLM proxy config generation and lifecycle."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.conftest import posix_only, sleeper_payload, write_model_yaml
from vllmops import lifecycle, proxy, service
from vllmops.project import Project, ProjectConfigError, load_project
from vllmops.proxy import ProxyOptions


def _patch_config(project: Project, updates: dict[str, Any]) -> Project:
    """Merge updates into .vllmops/config.yaml, reload, return the new Project."""
    raw = yaml.safe_load(project.config_path.read_text(encoding="utf-8")) or {}
    raw.update(updates)
    project.config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_project(project.root)


def _mark_running(project: Project, name: str) -> None:
    """Make a model look running by pointing its pid file at the test process."""
    paths = service.runtime_paths_for(project, name)
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text(str(os.getpid()), encoding="utf-8")


def _mark_stopped(project: Project, name: str) -> None:
    service.runtime_paths_for(project, name).pid_path.unlink(missing_ok=True)


def _write_overlay(project: Project, payload: object) -> None:
    project.proxy_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    project.proxy_overlay_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _options(project: Project, **overrides: Any) -> ProxyOptions:
    """The project's proxy options, with fields swapped for the case under test.

    Production code only ever builds these from the config section, but the
    functions that consume them are pure, so a test can hand them any shape
    without writing a config file for each combination.
    """
    return replace(proxy.config_options(project), **overrides)


def _entry_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in config["model_list"]:
        if entry["model_name"] == name:
            found: dict[str, Any] = entry
            return found
    raise AssertionError(f"{name} not in model_list")


# --- model selection ---


def test_only_running_models_by_default(project: Project) -> None:
    write_model_yaml(project, "up", sleeper_payload("up", port=18001))
    write_model_yaml(project, "down", sleeper_payload("down", port=18002))
    _mark_running(project, "up")

    result = proxy.build_proxy_config(project, _options(project))

    assert [m.name for m in result.models] == ["up"]
    assert ("down", "not running") in result.skipped


def test_include_stopped_covers_the_whole_catalog(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert [m.name for m in result.models] == ["a", "b"]
    assert result.skipped == []


def test_profile_narrows_the_selection(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    project = _patch_config(project, {"profiles": {"dev": ["a"]}})

    result = proxy.build_proxy_config(project, _options(project, profile="dev", include_stopped=True))

    assert [m.name for m in result.models] == ["a"]


def test_unknown_profile_raises(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    with pytest.raises(service.UnknownProfileError):
        proxy.build_proxy_config(project, _options(project, profile="nope", include_stopped=True))


def test_broken_yaml_is_skipped_with_its_error(project: Project) -> None:
    write_model_yaml(project, "ok", sleeper_payload("ok", port=18001))
    project.models_dir.mkdir(parents=True, exist_ok=True)
    (project.models_dir / "bad.yaml").write_text("name: bad\nvllm: [oops]\n", encoding="utf-8")

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert [m.name for m in result.models] == ["ok"]
    skipped_names = dict(result.skipped)
    assert "bad" in skipped_names
    assert skipped_names["bad"]


def test_model_without_port_is_skipped(project: Project) -> None:
    write_model_yaml(project, "noport", sleeper_payload("noport", port=0, with_metrics=False))

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.models == []
    assert result.skipped == [("noport", "no HTTP port configured")]


# --- generated config shape ---


def test_model_name_is_the_vllmops_name_and_upstream_is_the_served_name(project: Project) -> None:
    payload = sleeper_payload("qwen", port=18001)
    payload["vllm"]["args"] = {"--served-model-name": "Qwen/Qwen3-8B"}
    write_model_yaml(project, "qwen", payload)

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))
    entry = _entry_by_name(result.config, "qwen")

    assert entry["litellm_params"]["model"] == "hosted_vllm/Qwen/Qwen3-8B"


def test_served_name_falls_back_to_the_vllm_model(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))
    assert result.models[0].served_model == "import time; time.sleep(60)"


def test_served_name_from_a_list_takes_the_first(project: Project) -> None:
    payload = sleeper_payload("m", port=18001)
    payload["vllm"]["args"] = {"--served-model-name": ["primary", "alias"]}
    write_model_yaml(project, "m", payload)

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.models[0].served_model == "primary"


def test_served_name_from_extra_args(project: Project) -> None:
    payload = sleeper_payload("m", port=18001)
    payload["vllm"]["extra_args"] = ["--served-model-name", "from-extra"]
    write_model_yaml(project, "m", payload)

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.models[0].served_model == "from-extra"


def test_api_base_carries_the_v1_postfix(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.models[0].api_base == "http://127.0.0.1:18001/v1"
    assert _entry_by_name(result.config, "m")["litellm_params"]["api_base"].endswith("/v1")


def test_upstream_host_override(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))

    result = proxy.build_proxy_config(project, _options(project, upstream_host="10.0.0.5", include_stopped=True))

    assert result.models[0].api_base == "http://10.0.0.5:18001/v1"


def test_api_key_from_args_propagates(project: Project) -> None:
    payload = sleeper_payload("m", port=18001)
    payload["vllm"]["args"] = {"--api-key": "sk-local"}
    write_model_yaml(project, "m", payload)

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert _entry_by_name(result.config, "m")["litellm_params"]["api_key"] == "sk-local"


def test_api_key_from_extra_args_propagates(project: Project) -> None:
    payload = sleeper_payload("m", port=18001)
    payload["vllm"]["extra_args"] = ["--api-key", "sk-extra"]
    write_model_yaml(project, "m", payload)

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert _entry_by_name(result.config, "m")["litellm_params"]["api_key"] == "sk-extra"


def test_no_api_key_key_when_the_model_has_none(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert "api_key" not in _entry_by_name(result.config, "m")["litellm_params"]


def test_api_key_from_a_list_valued_arg_takes_the_first(project: Project) -> None:
    payload = sleeper_payload("m", port=18001)
    payload["vllm"]["args"] = {"--api-key": ["sk-first", "sk-second"]}
    write_model_yaml(project, "m", payload)

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert _entry_by_name(result.config, "m")["litellm_params"]["api_key"] == "sk-first"


def test_a_boolean_api_key_arg_carries_no_value(project: Project) -> None:
    """`--api-key: true` is a flag, not a key. It must not become the string "True"."""
    payload = sleeper_payload("m", port=18001)
    payload["vllm"]["args"] = {"--api-key": True}
    write_model_yaml(project, "m", payload)

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert "api_key" not in _entry_by_name(result.config, "m")["litellm_params"]


def test_a_model_that_stops_parsing_after_the_catalog_read_is_skipped(project: Project) -> None:
    """The catalog read and the config build parse the YAML separately, so a file
    edited in between must land in the skip list instead of raising."""
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    entries = service.list_catalog_entries(project)
    entries[0].yaml_path.write_text("name: m\nvllm: [not, a, mapping]\n", encoding="utf-8")

    models, skipped = proxy.eligible_models(entries, _options(project, include_stopped=True))

    assert models == []
    assert [name for name, _ in skipped] == ["m"]


# --- master key ---


def test_master_key_absent_when_not_configured(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(proxy.MASTER_KEY_ENV, raising=False)
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert "general_settings" not in result.config


def test_master_key_from_shell_is_emitted_as_a_reference(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(proxy.MASTER_KEY_ENV, "sk-super-secret")
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))
    rendered = proxy.render_proxy_config(result.config)

    assert result.config["general_settings"]["master_key"] == f"os.environ/{proxy.MASTER_KEY_ENV}"
    assert "sk-super-secret" not in rendered


def test_master_key_from_dotenv_is_detected(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(proxy.MASTER_KEY_ENV, raising=False)
    (project.root / ".env").write_text(f"{proxy.MASTER_KEY_ENV}=sk-from-dotenv\n", encoding="utf-8")
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.config["general_settings"]["master_key"] == f"os.environ/{proxy.MASTER_KEY_ENV}"


def test_overlay_master_key_wins(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(proxy.MASTER_KEY_ENV, "sk-shell")
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _write_overlay(project, {"general_settings": {"master_key": "os.environ/MY_OWN_KEY"}})

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.config["general_settings"]["master_key"] == "os.environ/MY_OWN_KEY"


# --- overlay merge ---


def test_overlay_settings_block_is_merged(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _write_overlay(project, {"litellm_settings": {"drop_params": True, "num_retries": 2}})

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.config["litellm_settings"] == {"drop_params": True, "num_retries": 2}
    assert result.unknown_overlay_keys == []


def test_overlay_keeps_generated_keys_of_the_same_block(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(proxy.MASTER_KEY_ENV, "sk-shell")
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _write_overlay(project, {"general_settings": {"disable_spend_logs": True}})

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.config["general_settings"] == {
        "master_key": f"os.environ/{proxy.MASTER_KEY_ENV}",
        "disable_spend_logs": True,
    }


def test_overlay_model_list_is_appended(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _write_overlay(
        project,
        {"model_list": [{"model_name": "remote", "litellm_params": {"model": "openai/gpt-4o"}}]},
    )

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert [entry["model_name"] for entry in result.config["model_list"]] == ["m", "remote"]


def test_overlay_unknown_keys_are_reported_but_passed_through(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _write_overlay(project, {"litellm_setting": {"drop_params": True}})

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    assert result.unknown_overlay_keys == ["litellm_setting"]
    assert result.config["litellm_setting"] == {"drop_params": True}


def test_overlay_must_be_a_mapping(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _write_overlay(project, ["not", "a", "mapping"])

    with pytest.raises(ValueError, match="YAML mapping"):
        proxy.build_proxy_config(project, _options(project, include_stopped=True))


def test_missing_overlay_is_not_an_error(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    assert not project.proxy_overlay_path.exists()
    assert proxy.load_proxy_overlay(project) == {}


# --- rendering and paths ---


def test_render_is_valid_yaml_with_model_list_first(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))

    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))
    rendered = proxy.render_proxy_config(result.config)

    assert rendered.startswith("model_list:")
    assert yaml.safe_load(rendered) == result.config


def test_write_proxy_config_lands_in_runtime(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    result = proxy.build_proxy_config(project, _options(project, include_stopped=True))

    path = proxy.write_proxy_config(project, result.config)

    assert path == project.root / "runtime" / "litellm.yaml"
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == result.config


def test_runtime_paths_cannot_collide_with_a_model(project: Project) -> None:
    paths = proxy.runtime_paths(project)
    assert paths.pid_path.name == "_proxy.pid"
    assert paths.log_path.name == "_proxy.log"


def test_bind_all_interfaces_is_probed_on_loopback() -> None:
    assert proxy.base_url("0.0.0.0", 4000) == "http://127.0.0.1:4000"
    assert proxy.readiness_url("0.0.0.0", 4000) == "http://127.0.0.1:4000/health/readiness"
    assert proxy.base_url("10.0.0.5", 4000) == "http://10.0.0.5:4000"


# --- options and command ---


def test_config_options_default_to_a_loopback_gateway(project: Project) -> None:
    assert proxy.config_options(project) == ProxyOptions(host="127.0.0.1", port=4000, upstream_host="127.0.0.1")


def test_config_options_carry_every_proxy_key(project: Project) -> None:
    project = _patch_config(
        project,
        {
            "proxy": {
                "host": "0.0.0.0",
                "port": 4321,
                "upstream_host": "10.0.0.5",
                "profile": "dev",
                "expose": "all",
                "num_workers": 3,
                "detailed_debug": True,
            }
        },
    )

    assert proxy.config_options(project, config_dir=project.models_dir) == ProxyOptions(
        host="0.0.0.0",
        port=4321,
        upstream_host="10.0.0.5",
        profile="dev",
        include_stopped=True,
        config_dir=project.models_dir,
        num_workers=3,
        detailed_debug=True,
    )


def test_config_rejects_an_unknown_expose_value(project: Project) -> None:
    with pytest.raises(ProjectConfigError, match="proxy.expose"):
        _patch_config(project, {"proxy": {"expose": "sometimes"}})


def test_config_rejects_a_worker_count_below_one(project: Project) -> None:
    with pytest.raises(ProjectConfigError, match="proxy.num_workers"):
        _patch_config(project, {"proxy": {"num_workers": 0}})


def test_config_rejects_a_leftover_follow_models_key(project: Project) -> None:
    with pytest.raises(ProjectConfigError, match="proxy.follow_models"):
        _patch_config(project, {"proxy": {"follow_models": True}})


def test_command_shape(project: Project, tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    argv = proxy.build_proxy_command(project, config_path, proxy.config_options(project))

    assert argv[1:] == [
        "--config",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        "4000",
        "--num_workers",
        "1",
    ]


def test_detailed_debug_is_opt_in(project: Project, tmp_path: Path) -> None:
    options = _options(project, detailed_debug=True, num_workers=4)
    argv = proxy.build_proxy_command(project, tmp_path / "c.yaml", options)

    assert "--detailed_debug" in argv
    assert argv[argv.index("--num_workers") + 1] == "4"


# --- executable resolution ---


def test_custom_executable_is_returned_verbatim(project: Project) -> None:
    project = _patch_config(project, {"proxy": {"executable": "/opt/litellm/bin/litellm"}})
    assert proxy.resolve_litellm_executable(project) == "/opt/litellm/bin/litellm"


def test_project_venv_executable_wins(project: Project) -> None:
    for relative in (Path(".venv/bin/litellm"), Path(".venv/Scripts/litellm.exe")):
        candidate = project.root / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("#!/bin/sh\n", encoding="utf-8")
        assert proxy.resolve_litellm_executable(project) == str(candidate)
        candidate.unlink()


def test_check_litellm_available_hints_at_the_install_command(project: Project) -> None:
    missing = project.root / "nowhere" / "litellm"
    with pytest.raises(proxy.LitellmExecutableNotFoundError, match="not found"):
        proxy.check_litellm_available(project, str(missing))


def test_check_litellm_available_accepts_an_existing_file(project: Project, tmp_path: Path) -> None:
    stub = tmp_path / "litellm"
    stub.write_text("#!/bin/sh\n", encoding="utf-8")
    proxy.check_litellm_available(project, str(stub))


def test_check_litellm_available_accepts_a_bare_name_on_path(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """A name with no separator is looked up in PATH, not under the project."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/local/bin/litellm")
    proxy.check_litellm_available(project, "litellm")


# --- port conflict ---


def test_proxy_port_taken_by_a_running_model_conflicts(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=4000))
    _mark_running(project, "m")

    with pytest.raises(service.PortConflictError, match="4000"):
        proxy._check_port_free(project, _options(project))


def test_proxy_port_free_when_the_model_is_stopped(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=4000))
    proxy._check_port_free(project, _options(project))


# --- status ---


def _write_current_config(project: Project, options: ProxyOptions) -> None:
    proxy.write_proxy_config(project, proxy.build_proxy_config(project, options).config)


def test_status_reports_stopped_before_the_first_start(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _mark_running(project, "m")

    status = proxy.proxy_status(project, _options(project))

    assert status.running is False
    assert status.pid is None
    assert status.configured_models == []
    assert status.eligible_models == ["m"]


def test_status_reports_no_drift_before_the_config_is_generated(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _mark_running(project, "m")

    status = proxy.proxy_status(project, _options(project))

    assert not proxy.generated_config_path(project).exists()
    assert status.drifted is False
    assert status.drift_reason is None


def test_status_reports_no_drift_when_the_config_matches(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _mark_running(project, "m")
    options = _options(project)
    _write_current_config(project, options)

    status = proxy.proxy_status(project, options)

    assert status.configured_models == ["m"]
    assert status.drifted is False


def test_status_detects_drift_after_a_model_starts(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    _mark_running(project, "a")
    options = _options(project)
    _write_current_config(project, options)

    _mark_running(project, "b")
    status = proxy.proxy_status(project, options)

    assert status.configured_models == ["a"]
    assert status.eligible_models == ["a", "b"]
    assert status.drifted is True
    assert status.drift_reason == "added b"


def test_status_detects_drift_when_a_model_changed_port(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    _mark_running(project, "a")
    options = _options(project)
    _write_current_config(project, options)

    write_model_yaml(project, "a", sleeper_payload("a", port=18009))
    status = proxy.proxy_status(project, options)

    assert status.configured_models == status.eligible_models == ["a"]
    assert status.drifted is True


def test_status_tolerates_a_corrupt_config_on_disk(project: Project) -> None:
    proxy.generated_config_path(project).parent.mkdir(parents=True, exist_ok=True)
    proxy.generated_config_path(project).write_text("model_list: [oops\n", encoding="utf-8")

    status = proxy.proxy_status(project, _options(project))

    assert status.configured_models == []


def test_status_tolerates_a_config_without_a_model_list(project: Project) -> None:
    """Valid YAML that lost its model_list reads as zero configured models."""
    proxy.generated_config_path(project).parent.mkdir(parents=True, exist_ok=True)
    proxy.generated_config_path(project).write_text("general_settings: {}\n", encoding="utf-8")

    status = proxy.proxy_status(project, _options(project))

    assert status.configured_models == []


def test_status_flags_a_stale_pid_file(project: Project) -> None:
    paths = proxy.runtime_paths(project)
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text("999999999", encoding="utf-8")

    status = proxy.proxy_status(project, _options(project))

    assert status.running is False
    assert status.stale_pid_file is True


# --- lifecycle (POSIX only, like the vLLM process tests) ---


def _litellm_stub(tmp_path: Path) -> Path:
    """A stand-in for the litellm binary that just stays alive."""
    stub = tmp_path / "litellm-stub"
    stub.write_text("#!/bin/sh\nexec sleep 60\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _project_with_stub(project: Project, tmp_path: Path) -> Project:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _mark_running(project, "m")
    return _patch_config(project, {"proxy": {"executable": str(_litellm_stub(tmp_path))}})


def _kill_proxy(project: Project) -> None:
    pid = lifecycle.read_pid(proxy.runtime_paths(project).pid_path)
    if pid is not None and lifecycle.is_alive(pid):
        lifecycle.terminate(pid, timeout=5.0)


@posix_only
def test_start_writes_a_pid_file_and_the_config(project: Project, tmp_path: Path) -> None:
    project = _project_with_stub(project, tmp_path)
    try:
        result = proxy.start_proxy(project, _options(project))

        assert lifecycle.is_alive(result.pid)
        assert lifecycle.read_pid(proxy.runtime_paths(project).pid_path) == result.pid
        assert result.config_path.is_file()
        assert [m.name for m in result.models] == ["m"]
    finally:
        _kill_proxy(project)


@posix_only
def test_double_start_refuses(project: Project, tmp_path: Path) -> None:
    project = _project_with_stub(project, tmp_path)
    try:
        proxy.start_proxy(project, _options(project))
        with pytest.raises(proxy.ProxyAlreadyRunningError):
            proxy.start_proxy(project, _options(project))
    finally:
        _kill_proxy(project)


@posix_only
def test_stop_removes_the_pid_file(project: Project, tmp_path: Path) -> None:
    project = _project_with_stub(project, tmp_path)
    result = proxy.start_proxy(project, _options(project))

    proxy.stop_proxy(project, timeout=5.0)

    assert not proxy.runtime_paths(project).pid_path.exists()
    assert not lifecycle.is_alive(result.pid)


@posix_only
def test_stop_when_not_running_raises(project: Project) -> None:
    with pytest.raises(proxy.ProxyNotRunningError):
        proxy.stop_proxy(project, timeout=5.0)


@posix_only
def test_restart_yields_a_new_pid(project: Project, tmp_path: Path) -> None:
    project = _project_with_stub(project, tmp_path)
    try:
        first = proxy.start_proxy(project, _options(project))
        second = proxy.restart_proxy(project, _options(project), timeout=5.0)

        assert second.pid != first.pid
        assert not lifecycle.is_alive(first.pid)
        assert lifecycle.is_alive(second.pid)
    finally:
        _kill_proxy(project)


@posix_only
def test_restart_works_from_a_stopped_state(project: Project, tmp_path: Path) -> None:
    project = _project_with_stub(project, tmp_path)
    try:
        result = proxy.restart_proxy(project, _options(project), timeout=5.0)
        assert lifecycle.is_alive(result.pid)
    finally:
        _kill_proxy(project)


@posix_only
def test_a_restart_keeps_the_gateway_log_of_the_previous_run(project: Project, tmp_path: Path) -> None:
    project = _project_with_stub(project, tmp_path)
    try:
        first = proxy.start_proxy(project, _options(project))
        proxy.restart_proxy(project, _options(project), timeout=5.0)

        log = first.log_path.read_text(encoding="utf-8")
        assert log.count("=== vllmops: process started ") == 2
        assert not first.log_path.with_suffix(".log.prev").exists()
    finally:
        _kill_proxy(project)


@posix_only
def test_a_restart_without_models_keeps_the_running_gateway(project: Project, tmp_path: Path) -> None:
    project = _project_with_stub(project, tmp_path)
    try:
        first = proxy.start_proxy(project, _options(project))
        config_before = first.config_path.read_text(encoding="utf-8")
        _mark_stopped(project, "m")

        with pytest.raises(proxy.NoProxyModelsError):
            proxy.restart_proxy(project, _options(project), timeout=5.0)

        assert lifecycle.is_alive(first.pid)
        assert lifecycle.read_pid(proxy.runtime_paths(project).pid_path) == first.pid
        assert first.config_path.read_text(encoding="utf-8") == config_before
    finally:
        _kill_proxy(project)


@posix_only
def test_a_restart_onto_a_taken_port_keeps_the_running_gateway(project: Project, tmp_path: Path) -> None:
    project = _project_with_stub(project, tmp_path)
    try:
        first = proxy.start_proxy(project, _options(project))

        with pytest.raises(service.PortConflictError):
            proxy.restart_proxy(project, _options(project, port=18001), timeout=5.0)

        assert lifecycle.is_alive(first.pid)
    finally:
        _kill_proxy(project)


@posix_only
def test_a_restart_does_not_respawn_when_the_old_gateway_survives(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_stub(project, tmp_path)
    try:
        first = proxy.start_proxy(project, _options(project))
        spawned: list[list[str]] = []

        def fake_spawn(args: list[str], *_: object) -> int:
            spawned.append(args)
            return 0

        monkeypatch.setattr(lifecycle, "terminate", lambda pid, timeout: False)
        monkeypatch.setattr(lifecycle, "spawn_detached", fake_spawn)

        with pytest.raises(proxy.ProxyStopFailedError, match=str(first.pid)):
            proxy.restart_proxy(project, _options(project), timeout=5.0)

        assert spawned == []
        assert lifecycle.read_pid(proxy.runtime_paths(project).pid_path) == first.pid
    finally:
        monkeypatch.undo()
        _kill_proxy(project)


@posix_only
def test_start_without_eligible_models_raises(project: Project, tmp_path: Path) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    project = _patch_config(project, {"proxy": {"executable": str(_litellm_stub(tmp_path))}})

    with pytest.raises(proxy.NoProxyModelsError) as excinfo:
        proxy.start_proxy(project, _options(project))

    assert excinfo.value.skipped == [("m", "not running")]


@posix_only
def test_start_without_litellm_installed_fails_with_a_hint(project: Project, tmp_path: Path) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _mark_running(project, "m")
    project = _patch_config(project, {"proxy": {"executable": str(tmp_path / "absent-litellm")}})

    with pytest.raises(proxy.LitellmExecutableNotFoundError):
        proxy.start_proxy(project, _options(project))


# --- caller-supplied catalog ---


def test_eligible_models_matches_the_catalog_reading_wrapper(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    _mark_running(project, "a")
    options = _options(project)

    entries = service.list_catalog_entries(project)

    assert proxy.eligible_models(entries, options) == proxy._select_models(project, options)


def test_supplied_entries_are_used_as_is(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _mark_running(project, "m")

    result = proxy.build_proxy_config(project, _options(project), entries=[])

    assert result.models == []


def test_supplied_entries_are_ignored_when_a_profile_narrows(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _mark_running(project, "m")
    project = _patch_config(project, {"profiles": {"dev": ["m"]}})

    result = proxy.build_proxy_config(project, _options(project, profile="dev"), entries=[])

    assert [model.name for model in result.models] == ["m"]


def test_supplied_entries_are_ignored_for_a_custom_models_dir(project: Project) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))
    _mark_running(project, "m")

    result = proxy.build_proxy_config(project, _options(project, config_dir=project.models_dir), entries=[])

    assert [model.name for model in result.models] == ["m"]


# --- drift detection ---


def test_drift_sees_no_change_right_after_a_generation(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    _mark_running(project, "a")
    options = _options(project)
    _write_current_config(project, options)

    reason, _ = proxy._drift(project, options)

    assert reason is None


def test_drift_reports_a_model_that_appeared(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    _mark_running(project, "a")
    options = _options(project)
    _write_current_config(project, options)

    _mark_running(project, "b")
    reason, _ = proxy._drift(project, options)

    assert reason == "added b"


def test_drift_reports_a_model_that_disappeared(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    _mark_running(project, "a")
    options = _options(project)
    _write_current_config(project, options)

    _mark_stopped(project, "a")
    reason, _ = proxy._drift(project, options)

    assert reason == "removed a"


def test_drift_notices_an_overlay_edit(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    _mark_running(project, "a")
    options = _options(project)
    _write_current_config(project, options)

    _write_overlay(project, {"litellm_settings": {"drop_params": True}})
    reason, _ = proxy._drift(project, options)

    assert reason == "config changed"


def test_drift_reports_a_config_that_was_never_generated(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    _mark_running(project, "a")

    reason, _ = proxy._drift(project, _options(project))

    assert reason == "generated config missing"


# --- refresh ---


def test_refresh_does_not_start_a_stopped_gateway(project: Project) -> None:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    _mark_running(project, "a")

    assert proxy.refresh_proxy(project) is None
    assert not proxy.generated_config_path(project).exists()


def test_refresh_ignores_a_stale_pid_file(project: Project) -> None:
    paths = proxy.runtime_paths(project)
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text("999999999", encoding="utf-8")

    assert proxy.refresh_proxy(project) is None


def _project_with_two_models(project: Project, tmp_path: Path) -> Project:
    write_model_yaml(project, "a", sleeper_payload("a", port=18001))
    write_model_yaml(project, "b", sleeper_payload("b", port=18002))
    _mark_running(project, "a")
    return _patch_config(project, {"proxy": {"executable": str(_litellm_stub(tmp_path))}})


@posix_only
def test_refresh_leaves_an_up_to_date_gateway_running(project: Project, tmp_path: Path) -> None:
    project = _project_with_two_models(project, tmp_path)
    try:
        started = proxy.start_proxy(project, _options(project))

        refresh = proxy.refresh_proxy(project)

        assert refresh is not None
        assert refresh.action == "unchanged"
        assert lifecycle.is_alive(started.pid)
    finally:
        _kill_proxy(project)


def _gateway_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The litellm stub never listens, so readiness is faked for refreshes meant to succeed."""
    monkeypatch.setattr(service, "probe_health", lambda url: True)


@posix_only
def test_refresh_respawns_with_the_new_model(project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project_with_two_models(project, tmp_path)
    _gateway_answers(monkeypatch)
    try:
        started = proxy.start_proxy(project, _options(project))
        _mark_running(project, "b")

        refresh = proxy.refresh_proxy(project)

        assert refresh is not None
        assert refresh.action == "restarted"
        assert refresh.reason == "added b"
        assert refresh.ready_error is None
        assert refresh.pid != started.pid
        assert [model.name for model in refresh.models] == ["a", "b"]
        assert not lifecycle.is_alive(started.pid)
        assert proxy._configured_model_names(proxy.generated_config_path(project)) == ["a", "b"]
    finally:
        _kill_proxy(project)


@posix_only
def test_refresh_reports_a_respawned_gateway_that_does_not_answer(project: Project, tmp_path: Path) -> None:
    project = _project_with_two_models(project, tmp_path)
    try:
        proxy.start_proxy(project, _options(project))
        _mark_running(project, "b")

        refresh = proxy.refresh_proxy(project, ready_timeout=0.0)

        assert refresh is not None
        assert refresh.action == "restarted"
        assert refresh.ready_error is not None
        assert proxy.READINESS_PATH in refresh.ready_error
        assert lifecycle.is_alive(refresh.pid)
    finally:
        _kill_proxy(project)


@posix_only
def test_refresh_keeps_the_configured_port_across_a_respawn(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_two_models(project, tmp_path)
    _gateway_answers(monkeypatch)
    project = _patch_config(project, {"proxy": {**project.config.proxy.model_dump(), "port": 5002}})
    try:
        proxy.start_proxy(project, proxy.config_options(project))
        _mark_running(project, "b")

        refresh = proxy.refresh_proxy(project)

        assert refresh is not None and refresh.action == "restarted"
        status = proxy.proxy_status(project, proxy.config_options(project))
        assert status.running is True
        assert status.url == "http://127.0.0.1:5002"
    finally:
        _kill_proxy(project)


@posix_only
def test_refresh_does_nothing_when_no_model_is_left(project: Project, tmp_path: Path) -> None:
    project = _project_with_two_models(project, tmp_path)
    try:
        started = proxy.start_proxy(project, _options(project))
        _mark_stopped(project, "a")

        refresh = proxy.refresh_proxy(project)

        assert refresh is not None
        assert refresh.action == "no-models"
        assert refresh.pid == started.pid
        assert lifecycle.is_alive(started.pid)
        assert proxy._configured_model_names(proxy.generated_config_path(project)) == ["a"]
    finally:
        _kill_proxy(project)
