"""Tests for the typer CLI: what reaches stdout and stderr."""

from __future__ import annotations

import os

import pytest
import yaml
from typer.testing import CliRunner

from tests.conftest import sleeper_payload, write_model_yaml
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
