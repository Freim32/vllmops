"""POSIX process primitives for managing bare-metal vLLM processes."""

import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def ensure_supported_platform() -> None:
    if sys.platform == "win32":
        raise RuntimeError("vllmctl lifecycle commands require POSIX (Linux/macOS); Windows is not supported.")


def is_alive(pid: int) -> bool:
    """Return True if a process with the given PID exists.

    Reaps zombie children of the current process before probing, otherwise a
    long-lived parent (such as pytest) would keep zombies indefinitely and
    `kill(pid, 0)` would report them as alive forever.
    """
    if pid <= 0:
        return False

    if sys.platform != "win32":
        try:
            reaped_pid, _ = os.waitpid(pid, os.WNOHANG)  # type: ignore[attr-defined]
            if reaped_pid == pid:
                return False
        except ChildProcessError:
            pass
        except OSError:
            pass

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.is_file():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def spawn_detached(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    pid_path: Path,
) -> int:
    """Spawn a detached background process and write its PID to pid_path.

    The env dict is used verbatim as the child's environment. The child
    becomes its own session leader so the whole group can later be signaled
    with `os.killpg(pid, ...)`. Each spawn rotates the existing log file to
    `<log_path>.prev` so the new run starts with a clean log.
    """
    ensure_supported_platform()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    rotate_log_file(log_path)

    log_handle = open(log_path, "wb", buffering=0)
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()

    pid_path.write_text(str(process.pid), encoding="utf-8")
    return process.pid


def rotate_log_file(log_path: Path) -> Path | None:
    """Rotate `log_path` to `<log_path>.prev` so the next run starts fresh.

    Returns the backup path on success, None if there was nothing to rotate
    or the rename failed.
    """
    if not log_path.is_file():
        return None
    backup = log_path.with_suffix(log_path.suffix + ".prev")
    try:
        log_path.replace(backup)
    except OSError:
        return None
    return backup


def _signal_group_or_pid(pid: int, sig: int) -> bool:
    """Send a signal to the process group, falling back to the pid alone."""
    try:
        os.killpg(pid, sig)  # type: ignore[attr-defined]
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            return False


def terminate(pid: int, timeout: float = 30.0) -> bool:
    """Send SIGTERM, wait up to timeout, escalate to SIGKILL.

    Returns True when the process is no longer alive at the end of the call.
    """
    ensure_supported_platform()
    if not is_alive(pid):
        return True

    _signal_group_or_pid(pid, signal.SIGTERM)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.2)

    _signal_group_or_pid(pid, signal.SIGKILL)  # type: ignore[attr-defined]

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.1)
    return not is_alive(pid)
