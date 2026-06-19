"""Tests for vllmctl.lifecycle and the spawn-driven branches of service.

POSIX-only: every test in this module is skipped on Windows because
spawn_detached uses os.killpg, signal.SIGKILL, and start_new_session.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from tests.conftest import (
    fast_exit_payload,
    posix_only,
    sleeper_payload,
    write_model_yaml,
)
from vllmctl import lifecycle, service
from vllmctl.project import Project
from vllmctl.service import (
    ModelAlreadyRunningError,
    ModelNotRunningError,
    ModelStartupFailedError,
)

pytestmark = posix_only


# --- pure helpers ---


def test_is_alive_self() -> None:
    assert lifecycle.is_alive(os.getpid()) is True


def test_is_alive_non_existent_pid() -> None:
    # 99999999 is well above the max PID on Linux/macOS by default
    assert lifecycle.is_alive(99999999) is False


def test_is_alive_invalid_pids() -> None:
    assert lifecycle.is_alive(0) is False
    assert lifecycle.is_alive(-1) is False


def test_read_pid_missing(tmp_path: Path) -> None:
    assert lifecycle.read_pid(tmp_path / "no.pid") is None


def test_read_pid_garbage(tmp_path: Path) -> None:
    p = tmp_path / "garbage.pid"
    p.write_text("not a number\n")
    assert lifecycle.read_pid(p) is None


def test_read_pid_valid(tmp_path: Path) -> None:
    p = tmp_path / "valid.pid"
    p.write_text("12345\n")
    assert lifecycle.read_pid(p) == 12345


def test_terminate_already_dead_returns_true() -> None:
    assert lifecycle.terminate(99999999, timeout=0.5) is True


# --- log rotation ---


def test_rotate_log_file_no_op_when_missing(tmp_path: Path) -> None:
    log_path = tmp_path / "missing.log"
    assert lifecycle.rotate_log_file(log_path) is None
    assert not log_path.exists()
    assert not (tmp_path / "missing.log.prev").exists()


def test_rotate_log_file_moves_existing_to_prev(tmp_path: Path) -> None:
    log_path = tmp_path / "model.log"
    log_path.write_text("first run output\n", encoding="utf-8")
    backup = lifecycle.rotate_log_file(log_path)
    assert backup == log_path.with_suffix(".log.prev")
    assert backup is not None
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == "first run output\n"
    assert not log_path.exists()


def test_rotate_log_file_overwrites_existing_prev(tmp_path: Path) -> None:
    log_path = tmp_path / "model.log"
    backup_path = tmp_path / "model.log.prev"
    log_path.write_text("second run", encoding="utf-8")
    backup_path.write_text("ancient run", encoding="utf-8")
    lifecycle.rotate_log_file(log_path)
    # The previous .prev gets replaced by what was log_path
    assert backup_path.read_text(encoding="utf-8") == "second run"


# --- spawn / terminate roundtrip ---


def _wait_until_alive(pid: int, deadline: float = 2.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if lifecycle.is_alive(pid):
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} never became alive within {deadline}s")


def _wait_until_dead(pid: int, deadline: float = 5.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if not lifecycle.is_alive(pid):
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} did not die within {deadline}s")


def test_spawn_detached_writes_pid_file(tmp_path: Path) -> None:
    log_path = tmp_path / "out.log"
    pid_path = tmp_path / "out.pid"
    pid = lifecycle.spawn_detached(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env=dict(os.environ),
        log_path=log_path,
        pid_path=pid_path,
    )
    try:
        _wait_until_alive(pid)
        assert lifecycle.read_pid(pid_path) == pid
    finally:
        lifecycle.terminate(pid, timeout=2.0)


def test_spawn_detached_captures_stdout(tmp_path: Path) -> None:
    log_path = tmp_path / "out.log"
    pid_path = tmp_path / "out.pid"
    pid = lifecycle.spawn_detached(
        [sys.executable, "-u", "-c", "print('hello world')"],
        env=dict(os.environ),
        log_path=log_path,
        pid_path=pid_path,
    )
    _wait_until_dead(pid, deadline=3.0)
    assert "hello world" in log_path.read_text(encoding="utf-8")


def test_terminate_sends_sigterm_then_dies(tmp_path: Path) -> None:
    log_path = tmp_path / "out.log"
    pid_path = tmp_path / "out.pid"
    pid = lifecycle.spawn_detached(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env=dict(os.environ),
        log_path=log_path,
        pid_path=pid_path,
    )
    _wait_until_alive(pid)
    assert lifecycle.terminate(pid, timeout=5.0) is True
    assert lifecycle.is_alive(pid) is False


def test_terminate_escalates_to_sigkill_when_sigterm_ignored(tmp_path: Path) -> None:
    """Spawn a process that ignores SIGTERM; terminate() should escalate to SIGKILL."""
    log_path = tmp_path / "out.log"
    pid_path = tmp_path / "out.pid"
    pid = lifecycle.spawn_detached(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ],
        env=dict(os.environ),
        log_path=log_path,
        pid_path=pid_path,
    )
    _wait_until_alive(pid)
    # Short timeout forces escalation to SIGKILL fast
    assert lifecycle.terminate(pid, timeout=0.5) is True
    assert lifecycle.is_alive(pid) is False


def test_spawn_inherits_env(tmp_path: Path) -> None:
    log_path = tmp_path / "out.log"
    pid_path = tmp_path / "out.pid"
    pid = lifecycle.spawn_detached(
        [sys.executable, "-u", "-c", "import os; print(os.environ.get('VLLMCTL_TEST_MARKER'))"],
        env={**os.environ, "VLLMCTL_TEST_MARKER": "marker-xyz"},
        log_path=log_path,
        pid_path=pid_path,
    )
    _wait_until_dead(pid, deadline=3.0)
    assert "marker-xyz" in log_path.read_text(encoding="utf-8")


# --- service-level start/stop/restart ---


def test_service_start_then_stop(project: Project) -> None:
    write_model_yaml(project, "sleeper", sleeper_payload("sleeper", port=18001))
    started = service.start_model(project, "sleeper")
    try:
        assert started.running
        assert started.pid is not None
        _wait_until_alive(started.pid)
        snapshot = service.get_model_status(project, "sleeper")
        assert snapshot.running
    finally:
        if service.get_model_status(project, "sleeper").running:
            service.stop_model(project, "sleeper", timeout=2.0)

    after = service.get_model_status(project, "sleeper")
    assert after.running is False


def test_service_start_already_running_raises(project: Project) -> None:
    write_model_yaml(project, "sleeper", sleeper_payload("sleeper", port=18001))
    service.start_model(project, "sleeper")
    try:
        with pytest.raises(ModelAlreadyRunningError):
            service.start_model(project, "sleeper")
    finally:
        service.stop_model(project, "sleeper", timeout=2.0)


def test_service_stop_when_not_running_raises(project: Project) -> None:
    write_model_yaml(project, "sleeper", sleeper_payload("sleeper", port=18001))
    with pytest.raises(ModelNotRunningError):
        service.stop_model(project, "sleeper", timeout=1.0)


def test_service_stop_cleans_pid_file(project: Project) -> None:
    write_model_yaml(project, "sleeper", sleeper_payload("sleeper", port=18001))
    service.start_model(project, "sleeper")
    paths = service.runtime_paths_for(project, "sleeper")
    assert paths.pid_path.is_file()
    service.stop_model(project, "sleeper", timeout=2.0)
    assert not paths.pid_path.exists()


def test_service_restart_replaces_pid(project: Project) -> None:
    write_model_yaml(project, "sleeper", sleeper_payload("sleeper", port=18001))
    first = service.start_model(project, "sleeper")
    try:
        second = service.restart_model(project, "sleeper", timeout=2.0)
        assert second.running
        assert second.pid is not None
        assert second.pid != first.pid
    finally:
        if service.get_model_status(project, "sleeper").running:
            service.stop_model(project, "sleeper", timeout=2.0)


def test_service_start_clears_stale_pid_file(project: Project) -> None:
    write_model_yaml(project, "sleeper", sleeper_payload("sleeper", port=18001))
    paths = service.runtime_paths_for(project, "sleeper")
    paths.pid_path.parent.mkdir(parents=True, exist_ok=True)
    paths.pid_path.write_text("99999999")  # stale
    started = service.start_model(project, "sleeper")
    try:
        assert started.pid != 99999999
    finally:
        service.stop_model(project, "sleeper", timeout=2.0)


# --- wait_for_ready: process-died detection with a real spawn ---


def test_wait_for_ready_detects_real_process_death(project: Project) -> None:
    """Spawn a model that exits immediately; wait_for_ready must surface it."""
    write_model_yaml(project, "fail", fast_exit_payload("fail", port=18099))
    started = service.start_model(project, "fail")
    assert started.pid is not None
    _wait_until_dead(started.pid, deadline=3.0)

    with pytest.raises(ModelStartupFailedError):
        service.wait_for_ready(project, "fail", timeout=2.0, interval=0.1)
