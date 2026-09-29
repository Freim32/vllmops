"""POSIX process primitives for managing bare-metal vLLM processes."""

import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_WIN32_QUERY_LIMITED_INFORMATION = 0x1000
_WIN32_ERROR_ACCESS_DENIED = 5
_WIN32_STILL_ACTIVE = 259


def ensure_supported_platform() -> None:
    if sys.platform == "win32":
        raise RuntimeError("vllmops lifecycle commands require POSIX (Linux/macOS); Windows is not supported.")


def _win32_pid_exists(pid: int) -> bool:
    """Ask the kernel for a handle on the process.

    Windows numbers CTRL_C_EVENT as signal 0, so `os.kill(pid, 0)` there does not
    probe anything: it sends a console interrupt to that process group, and a
    caller asking about its own PID gets a KeyboardInterrupt of its own.
    """
    if sys.platform != "win32":  # keeps the calls below out of the POSIX type view
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(_WIN32_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # A process we are not allowed to query is still a process.
        return ctypes.get_last_error() == _WIN32_ERROR_ACCESS_DENIED
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == _WIN32_STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def is_alive(pid: int) -> bool:
    """Return True if a process with the given PID exists.

    Reaps zombie children of the current process before probing, otherwise a
    long-lived parent (such as pytest) would keep zombies indefinitely and
    `kill(pid, 0)` would report them as alive forever.
    """
    if pid <= 0:
        return False

    if sys.platform == "win32":
        return _win32_pid_exists(pid)

    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
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
        # POSIX-only in typeshed: the ignore is needed on Windows, unused on Linux.
        os.killpg(pid, sig)  # type: ignore[attr-defined, unused-ignore]
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

    _signal_group_or_pid(pid, signal.SIGKILL)  # type: ignore[attr-defined, unused-ignore]

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.1)
    return not is_alive(pid)
