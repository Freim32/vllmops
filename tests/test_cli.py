"""Tests for the typer CLI: what reaches stdout and stderr."""

from __future__ import annotations

import os

import pytest
import yaml
from typer.testing import CliRunner

from tests.conftest import posix_only, sleeper_payload, write_model_yaml
from vllmops import proxy, service
from vllmops.cli import app
from vllmops.project import Project


@pytest.fixture
def runner(project: Project, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """A runner whose commands find `project` from the working directory."""
    monkeypatch.chdir(project.root)
    return CliRunner()


def _mark_running(project: Project, name: str) -> None:
    paths = service.runtime_paths_for(project, name)
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text(str(os.getpid()), encoding="utf-8")


def test_proxy_config_prints_only_the_yaml_on_stdout(project: Project, runner: CliRunner) -> None:
    write_model_yaml(project, "up", sleeper_payload("up", port=18001))
    write_model_yaml(project, "down", sleeper_payload("down", port=18002))
    _mark_running(project, "up")

    result = runner.invoke(app, ["proxy", "config"])

    assert result.exit_code == 0
    expected = proxy.build_proxy_config(project, proxy.config_options(project)).config
    assert yaml.safe_load(result.stdout) == expected
    assert "skip" in result.stderr
    assert "down" in result.stderr


def test_proxy_config_warns_about_overlay_keys_on_stderr(project: Project, runner: CliRunner) -> None:
    write_model_yaml(project, "up", sleeper_payload("up", port=18001))
    _mark_running(project, "up")
    project.proxy_overlay_path.write_text("typo_settings: {}\n", encoding="utf-8")

    result = runner.invoke(app, ["proxy", "config"])

    assert yaml.safe_load(result.stdout)["typo_settings"] == {}
    assert "typo_settings" in result.stderr
    assert "unrecognized" not in result.stdout


def test_logs_without_a_log_file_keeps_stdout_empty(project: Project, runner: CliRunner) -> None:
    write_model_yaml(project, "m", sleeper_payload("m", port=18001))

    result = runner.invoke(app, ["logs", "m", "-n", "10"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "no log yet" in result.stderr


def test_proxy_logs_without_a_log_file_keeps_stdout_empty(runner: CliRunner) -> None:
    result = runner.invoke(app, ["proxy", "logs", "-n", "10"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "no log yet" in result.stderr


# --- invalid .vllmops/config.yaml ---


def test_an_invalid_project_config_is_one_line_not_a_traceback(project: Project, runner: CliRunner) -> None:
    project.config_path.write_text("proxy:\n  expose: tutti\n", encoding="utf-8")

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    one_line = " ".join(result.stdout.split())  # Rich wraps at the runner's 80 columns
    assert "Invalid .vllmops/config.yaml: `proxy.expose`: Input should be 'running' or 'all'" in one_line
    assert "Traceback" not in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_doctor_reports_an_invalid_project_config_in_one_line(project: Project, runner: CliRunner) -> None:
    project.config_path.write_text("proxy:\n  exopse: all\n", encoding="utf-8")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "unknown field `proxy.exopse`" in " ".join(result.stdout.split())


# --- broken model files ---


def _write_broken(project: Project, name: str) -> None:
    project.models_dir.mkdir(parents=True, exist_ok=True)
    (project.models_dir / f"{name}.yaml").write_text(f"name: {name}\nbogus_field: 1\n", encoding="utf-8")


def test_status_lists_a_broken_file_next_to_the_valid_models(project: Project, runner: CliRunner) -> None:
    write_model_yaml(project, "good", sleeper_payload("good", port=18001))
    _mark_running(project, "good")
    _write_broken(project, "rotto")

    result = runner.invoke(app, ["status"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    rows = {line.split("│")[1].strip(): line for line in result.stdout.splitlines() if line.count("│") > 2}
    assert "running" in rows["good"]
    assert "invalid" in rows["rotto"]
    assert "missing required field" in rows["rotto"]
    assert "vllmops validate" in result.stdout


def test_command_on_a_broken_file_says_why_instead_of_unknown(project: Project, runner: CliRunner) -> None:
    _write_broken(project, "rotto")

    result = runner.invoke(app, ["command", "rotto"])

    assert result.exit_code == 1
    one_line = " ".join(result.stdout.split())
    assert "Invalid model config:" in one_line
    assert "missing required field `vllm`" in one_line
    assert "Unknown model" not in one_line


@posix_only
def test_start_on_a_broken_file_says_why_instead_of_unknown(project: Project, runner: CliRunner) -> None:
    _write_broken(project, "rotto")

    result = runner.invoke(app, ["start", "rotto"])

    assert result.exit_code == 1
    assert "Invalid model config:" in " ".join(result.stdout.split())


def test_command_on_a_missing_name_is_still_unknown(project: Project, runner: CliRunner) -> None:
    _write_broken(project, "rotto")

    result = runner.invoke(app, ["command", "nope"])

    assert result.exit_code == 1
    assert "Unknown model:" in result.stdout


# --- brackets survive Rich markup ---


def test_a_service_error_with_brackets_is_printed_verbatim(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object) -> None:
        raise RuntimeError("bad value [x] in config")

    monkeypatch.setattr(proxy, "start_proxy", fail)

    result = runner.invoke(app, ["proxy", "start"])

    assert result.exit_code == 1
    assert "bad value [x] in config" in result.stdout


def test_the_litellm_install_hint_keeps_its_extra(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object) -> None:
        raise proxy.LitellmExecutableNotFoundError("run: uv tool install 'litellm[proxy]'")

    monkeypatch.setattr(proxy, "start_proxy", fail)

    result = runner.invoke(app, ["proxy", "start"])

    assert "'litellm[proxy]'" in result.stdout


def test_doctor_prints_details_and_hints_verbatim(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllmops import doctor  # noqa: PLC0415

    checks = [doctor.CheckResult("litellm executable", "warn", "not in [.venv]", hint="add 'litellm[proxy]'")]
    monkeypatch.setattr(doctor, "run_checks", lambda: checks)

    result = runner.invoke(app, ["doctor"])

    assert "not in [.venv]" in result.stdout
    assert "add 'litellm[proxy]'" in result.stdout


def test_command_prints_bracketed_arguments_verbatim(project: Project, runner: CliRunner) -> None:
    payload = sleeper_payload("m", port=18001)
    # Rich only eats word-like tags, so a bare number in brackets would not show the bug.
    payload["vllm"]["args"] = {"--override-generation-config": '{"stop": ["[end]"]}'}
    write_model_yaml(project, "m", payload)

    result = runner.invoke(app, ["command", "m"])

    assert result.exit_code == 0
    assert '"[end]"' in result.stdout


def test_the_startup_log_tail_keeps_vllm_prefixes(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    from vllmops.cli import _print_log_tail  # noqa: PLC0415

    log_path = service.runtime_paths_for(project, "m").log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("INFO [launcher.py:70] Route: /v1/models\nERROR [/core] boom\n", encoding="utf-8")

    _print_log_tail(project, "m")

    out = capsys.readouterr().out
    assert "[launcher.py:70] Route: /v1/models" in out
    assert "[/core] boom" in out
