"""Pure backend layer shared by the CLI and the TUI.

Functions here never write to a console, prompt the user, or call sys.exit.
They take resolved inputs, return data, and raise exceptions on failure.
"""

import os
import shlex
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vllmctl import lifecycle
from vllmctl.config import (
    Catalog,
    ModelConfig,
    create_default_model_config,
    dump_model_file,
    load_catalog,
    load_catalog_or_empty,
)
from vllmctl.project import GENERAL_PROFILE, Project, init_project, load_project


class ModelAlreadyExistsError(FileExistsError):
    """Raised when a model YAML exists and force is not set."""


class UnknownModelError(LookupError):
    """Raised when a requested model name is not in the catalog."""


class ModelAlreadyRunningError(RuntimeError):
    """Raised when start is requested but the model is already running."""


class ModelNotRunningError(RuntimeError):
    """Raised when stop or similar is requested but the model is not running."""


class ModelStartupFailedError(RuntimeError):
    """Raised when a model process exits while we were waiting for it to become ready."""


class ModelStartupTimeoutError(TimeoutError):
    """Raised when wait_for_ready exceeds its timeout without /health responding."""


class UnknownProfileError(LookupError):
    """Raised when a requested profile name is not declared in config."""


class VllmExecutableNotFoundError(RuntimeError):
    """Raised when the vllm binary cannot be found in the project venv or on PATH."""


@dataclass(frozen=True)
class CreateModelResult:
    destination: Path
    model: ModelConfig


@dataclass(frozen=True)
class RuntimePaths:
    pid_path: Path
    log_path: Path


@dataclass(frozen=True)
class ModelStatus:
    name: str
    running: bool
    pid: int | None
    log_path: Path
    pid_path: Path
    metrics_port: int | None
    stale_pid_file: bool


@dataclass(frozen=True)
class CatalogEntry:
    """One YAML in the models dir, valid or broken.

    Exactly one of `status` (parsed OK) or `error` (parse failure) is set.
    """

    name: str
    yaml_path: Path
    status: ModelStatus | None = None
    error: str | None = None

    @property
    def is_broken(self) -> bool:
        return self.error is not None


DOTENV_FILENAME = ".env"


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse a minimal .env file: KEY=value, optional quotes, # comments, export prefix."""
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key[0].isdigit() or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = value
    return result


def load_dotenv(project: Project) -> dict[str, str]:
    """Read the project's .env file. Returns an empty dict if missing or unreadable."""
    path = project.root / DOTENV_FILENAME
    if not path.is_file():
        return {}
    try:
        return parse_dotenv(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


# PYTHONUNBUFFERED forces Python to flush stdout/stderr line by line. Without
# it, vLLM's output is block-buffered when redirected to a log file and the
# TUI (or `vllmctl logs --follow`) sits frozen for many seconds at a time.
_BASELINE_ENV = {"PYTHONUNBUFFERED": "1"}


def project_venv_bin(project: Project) -> Path | None:
    """Return the project's `.venv/bin` (POSIX) or `.venv/Scripts` (Windows) if it exists."""
    for candidate in (project.root / ".venv" / "bin", project.root / ".venv" / "Scripts"):
        if candidate.is_dir():
            return candidate
    return None


def _activate_project_venv(project: Project, env: dict[str, str]) -> dict[str, str]:
    """Apply the env mutations of `source .venv/bin/activate`.

    Prepends `.venv/bin` to PATH, sets VIRTUAL_ENV, drops any inherited
    PYTHONHOME (which would break venv resolution).
    """
    venv_bin = project_venv_bin(project)
    if venv_bin is None:
        return env
    env = dict(env)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', os.defpath)}"
    env["VIRTUAL_ENV"] = str(venv_bin.parent)
    env.pop("PYTHONHOME", None)
    return env


def build_runtime_env(project: Project, model_env: dict[str, str]) -> dict[str, str]:
    """Merge env sources for a spawned vLLM process.

    Precedence (later wins): baseline, .env, shell, model YAML env. On top of
    the merge, the project's venv is activated when present.

    Secrets like HF_TOKEN are forbidden in model YAML by config validation,
    so they only flow in from .env or the shell.
    """
    merged = {**_BASELINE_ENV, **load_dotenv(project), **os.environ, **model_env}
    return _activate_project_venv(project, merged)


def get_project(start: Path | None = None) -> Project:
    return load_project(start)


def initialize_workspace(
    target: Path,
    force: bool = False,
    name: str | None = None,
) -> list[Path]:
    return init_project(target, force=force, name=name)


def resolve_models_dir(project: Project, override: Path | None) -> Path:
    return project.models_dir if override is None else override


def load_catalog_for(project: Project, config_dir: Path | None = None) -> Catalog:
    return load_catalog(resolve_models_dir(project, config_dir))


def load_catalog_or_empty_for(project: Project, config_dir: Path | None = None) -> Catalog:
    return load_catalog_or_empty(resolve_models_dir(project, config_dir))


def next_available_port(project: Project, config_dir: Path | None = None) -> int:
    catalog = load_catalog_or_empty_for(project, config_dir)
    return catalog.next_available_port(project.config.defaults.port_start)


def create_model(
    project: Project,
    *,
    name: str,
    hf_model: str,
    gpus: str,
    port: int,
    config_dir: Path | None = None,
    force: bool = False,
) -> CreateModelResult:
    """Create a new model YAML file. Raises ModelAlreadyExistsError if it exists and force is False."""
    effective_dir = resolve_models_dir(project, config_dir)
    destination = effective_dir / f"{name}.yaml"

    if destination.exists() and not force:
        raise ModelAlreadyExistsError(str(destination))

    model = create_default_model_config(
        name=name,
        hf_model=hf_model,
        gpus=gpus,
        port=port,
        host=project.config.defaults.host,
        hf_home=project.config.paths.hf_home,
        vllm_executable=project.config.defaults.vllm_executable,
        vllm_subcommand=project.config.defaults.vllm_subcommand,
        log_level=project.config.defaults.log_level,
    )
    dump_model_file(destination, model)
    return CreateModelResult(destination=destination, model=model)


def build_command_string(
    project: Project,
    model_name: str,
    config_dir: Path | None = None,
) -> str:
    """Return the bare-metal shell command for a model, including env prefix."""
    args, env = build_command_args(project, model_name, config_dir)
    command = shlex.join(args)
    if env:
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        command = f"{env_prefix} {command}"
    return command


def resolve_vllm_executable(project: Project, configured: str) -> str:
    """Pick the vllm binary, preferring the project's .venv when default."""
    if configured != "vllm":
        return configured
    venv_executable = project.venv_executable
    if venv_executable is not None:
        return str(venv_executable)
    return configured


def check_vllm_available(project: Project, executable: str) -> None:
    """Raise VllmExecutableNotFoundError with an actionable hint if vllm is missing."""
    candidate = Path(executable)
    if candidate.is_absolute() or any(sep in executable for sep in ("/", "\\")):
        if not candidate.is_file():
            raise VllmExecutableNotFoundError(f"vLLM executable not found at: {candidate}")
        return
    if shutil.which(executable) is not None:
        return
    venv_hint = project.root / ".venv" / "bin" / "vllm"
    raise VllmExecutableNotFoundError(
        f"vLLM not found.\n"
        f"  Looked for: {venv_hint} (no project venv)\n"
        f"  And in PATH for: {executable}\n\n"
        f"To install vLLM in this project:\n"
        f"  cd {project.root}\n"
        f"  uv sync\n"
        f"  # or: python -m venv .venv && .venv/bin/pip install -e .\n"
        f"\n"
        f"Or set vllm.executable to a custom path in the model YAML."
    )


def build_command_args(
    project: Project,
    model_name: str,
    config_dir: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, env) ready for spawning a vLLM process.

    Uses lenient single-model lookup so a broken sibling YAML doesn't
    prevent starting / restarting this model.
    """
    model = find_model(project, model_name, config_dir)
    if model is None:
        raise UnknownModelError(model_name)
    args = list(model.command_args())
    args[0] = resolve_vllm_executable(project, model.vllm.executable)
    return args, dict(model.env)


def runtime_paths_for(project: Project, model_name: str) -> RuntimePaths:
    pids_dir = project.resolve(project.config.paths.pids_dir)
    logs_dir = project.resolve(project.config.paths.logs_dir)
    return RuntimePaths(
        pid_path=pids_dir / f"{model_name}.pid",
        log_path=logs_dir / f"{model_name}.log",
    )


def find_model(
    project: Project,
    model_name: str,
    config_dir: Path | None = None,
) -> ModelConfig | None:
    """Lenient single-model lookup that tolerates broken sibling YAMLs.

    Tries the conventional `<model_name>.yaml` first, then falls back to
    scanning the directory for any file whose body has the matching name.
    """
    from vllmctl.config import load_model_file  # noqa: PLC0415

    effective_dir = resolve_models_dir(project, config_dir)
    if not effective_dir.exists():
        return None

    for ext in (".yaml", ".yml"):
        candidate = effective_dir / f"{model_name}{ext}"
        if candidate.is_file():
            try:
                model = load_model_file(candidate)
            except Exception:
                break
            if model.name == model_name:
                return model
            break

    for path in sorted(effective_dir.glob("*.yaml")) + sorted(effective_dir.glob("*.yml")):
        try:
            model = load_model_file(path)
        except Exception:
            continue
        if model.name == model_name:
            return model
    return None


def _lookup_metrics_port(
    project: Project,
    model_name: str,
    config_dir: Path | None,
) -> int | None:
    """Return the metrics port for a single model, tolerating broken siblings."""
    model = find_model(project, model_name, config_dir)
    return model.metrics_port if model is not None else None


def get_model_status(
    project: Project,
    model_name: str,
    config_dir: Path | None = None,
) -> ModelStatus:
    paths = runtime_paths_for(project, model_name)
    pid = lifecycle.read_pid(paths.pid_path)
    alive = pid is not None and lifecycle.is_alive(pid)
    stale = pid is not None and not alive

    return ModelStatus(
        name=model_name,
        running=alive,
        pid=pid if alive else None,
        log_path=paths.log_path,
        pid_path=paths.pid_path,
        metrics_port=_lookup_metrics_port(project, model_name, config_dir),
        stale_pid_file=stale,
    )


def list_model_statuses(
    project: Project,
    config_dir: Path | None = None,
) -> list[ModelStatus]:
    catalog = load_catalog_or_empty_for(project, config_dir)
    return [get_model_status(project, model.name, config_dir) for model in catalog.models]


def list_catalog_entries(
    project: Project,
    config_dir: Path | None = None,
) -> list[CatalogEntry]:
    """Return one entry per YAML in the models dir, including broken ones.

    Unlike `list_model_statuses`, a single invalid file does not abort the
    whole listing; it becomes a `CatalogEntry(error=...)` row. Duplicate
    names or ports also produce broken rows.
    """
    from vllmctl.config import load_model_file  # noqa: PLC0415

    effective_dir = resolve_models_dir(project, config_dir)
    if not effective_dir.exists():
        return []

    files = sorted(effective_dir.glob("*.yaml")) + sorted(effective_dir.glob("*.yml"))
    entries: list[CatalogEntry] = []
    seen_names: dict[str, Path] = {}
    seen_ports: dict[int, Path] = {}

    for path in files:
        try:
            model = load_model_file(path)
        except Exception as exc:
            entries.append(
                CatalogEntry(name=path.stem, yaml_path=path, error=_short_yaml_error(exc))
            )
            continue

        if model.name in seen_names:
            entries.append(
                CatalogEntry(
                    name=model.name,
                    yaml_path=path,
                    error=f"duplicate model name (also defined in {seen_names[model.name].name})",
                )
            )
            continue
        if model.metrics_port is not None and model.metrics_port in seen_ports:
            entries.append(
                CatalogEntry(
                    name=model.name,
                    yaml_path=path,
                    error=(
                        f"duplicate metrics port {model.metrics_port}"
                        f" (also used by {seen_ports[model.metrics_port].name})"
                    ),
                )
            )
            continue

        seen_names[model.name] = path
        if model.metrics_port is not None:
            seen_ports[model.metrics_port] = path

        status = _build_model_status(project, model.name, model.metrics_port)
        entries.append(CatalogEntry(name=model.name, yaml_path=path, status=status))

    return entries


@dataclass(frozen=True)
class ProfileView:
    """A profile resolved against the current catalog."""

    name: str
    entries: list[CatalogEntry]
    missing: list[str]

    @property
    def is_general(self) -> bool:
        return self.name == GENERAL_PROFILE

    @property
    def running_count(self) -> int:
        return sum(1 for e in self.entries if e.status is not None and e.status.running)

    @property
    def total_count(self) -> int:
        return len(self.entries)


def list_profiles(
    project: Project,
    config_dir: Path | None = None,
) -> list[ProfileView]:
    """Return all declared profiles in order, plus the synthetic 'general' catch-all.

    Profiles are returned untrimmed: an empty profile (declared but with no
    catalog matches) is still in the list so `profile show` can report it.
    Use `[v for v in list_profiles(...) if v.entries]` to filter for rendering.
    """
    entries = list_catalog_entries(project, config_dir)
    by_name = {entry.name: entry for entry in entries}

    assigned: set[str] = set()
    views: list[ProfileView] = []

    for profile_name, model_names in project.config.profiles.items():
        profile_entries: list[CatalogEntry] = []
        missing: list[str] = []
        for model_name in model_names:
            entry = by_name.get(model_name)
            if entry is None:
                missing.append(model_name)
            else:
                profile_entries.append(entry)
                assigned.add(model_name)
        views.append(ProfileView(name=profile_name, entries=profile_entries, missing=missing))

    general_entries = [entry for entry in entries if entry.name not in assigned]
    views.append(ProfileView(name=GENERAL_PROFILE, entries=general_entries, missing=[]))

    return views


@dataclass(frozen=True)
class BulkResult:
    """Outcome of a profile-wide lifecycle operation.

    Idempotent contract: re-running the same action on the same profile is safe,
    and the "succeeded" set only counts models the operation actually transitioned.
    Models already in the target state land in "skipped", failures in "failed".
    """

    profile: str
    action: str
    succeeded: list[str]
    skipped: list[tuple[str, str]]
    failed: list[tuple[str, str]]

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.skipped) + len(self.failed)


def _find_profile(views: list[ProfileView], name: str) -> ProfileView | None:
    for view in views:
        if view.name == name:
            return view
    return None


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else exc.__class__.__name__


def _run_in_parallel(
    items: list[CatalogEntry],
    work: Callable[[CatalogEntry], object],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Run `work(entry)` on each item concurrently, collect succeeded vs failed."""
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    if not items:
        return succeeded, failed
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    max_workers = min(len(items), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(work, entry): entry for entry in items}
        for future, entry in futures.items():
            try:
                future.result()
                succeeded.append(entry.name)
            except Exception as exc:
                failed.append((entry.name, _short_error(exc)))
    return succeeded, failed


def start_profile(
    project: Project,
    profile_name: str,
    config_dir: Path | None = None,
) -> BulkResult:
    """Start every model in the profile in parallel. Idempotent.

    Skips already-running and broken-YAML members. Failures on one model never
    abort the others. Raises UnknownProfileError if the profile is not declared.
    """
    views = list_profiles(project, config_dir)
    view = _find_profile(views, profile_name)
    if view is None:
        raise UnknownProfileError(profile_name)

    skipped: list[tuple[str, str]] = []
    to_start: list[CatalogEntry] = []
    for entry in view.entries:
        if entry.is_broken:
            skipped.append((entry.name, "invalid YAML"))
            continue
        if entry.status is not None and entry.status.running:
            skipped.append((entry.name, "already running"))
            continue
        to_start.append(entry)

    succeeded, failed = _run_in_parallel(
        to_start,
        lambda entry: start_model(project, entry.name, config_dir=config_dir),
    )
    return BulkResult(
        profile=profile_name,
        action="start",
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
    )


def stop_profile(
    project: Project,
    profile_name: str,
    config_dir: Path | None = None,
    timeout: float = 30.0,
) -> BulkResult:
    """Stop every running model in the profile in parallel. Idempotent.

    Broken YAMLs are still attempted (stop is PID-based and doesn't need the
    YAML). Already-stopped models land in skipped.
    """
    views = list_profiles(project, config_dir)
    view = _find_profile(views, profile_name)
    if view is None:
        raise UnknownProfileError(profile_name)

    skipped: list[tuple[str, str]] = []
    to_stop: list[CatalogEntry] = []
    for entry in view.entries:
        if entry.is_broken:
            to_stop.append(entry)
            continue
        if entry.status is None or not entry.status.running:
            skipped.append((entry.name, "not running"))
            continue
        to_stop.append(entry)

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    if to_stop:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        max_workers = min(len(to_stop), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(stop_model, project, entry.name, timeout): entry
                for entry in to_stop
            }
            for future, entry in futures.items():
                try:
                    future.result()
                    succeeded.append(entry.name)
                except ModelNotRunningError:
                    skipped.append((entry.name, "not running"))
                except Exception as exc:
                    failed.append((entry.name, _short_error(exc)))

    return BulkResult(
        profile=profile_name,
        action="stop",
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
    )


def restart_profile(
    project: Project,
    profile_name: str,
    config_dir: Path | None = None,
    timeout: float = 30.0,
) -> BulkResult:
    """Restart every model in the profile in parallel. Idempotent.

    Stopped models get started. Running models get stop+start. Broken YAMLs
    are skipped (restart needs the YAML to start back up).
    """
    views = list_profiles(project, config_dir)
    view = _find_profile(views, profile_name)
    if view is None:
        raise UnknownProfileError(profile_name)

    skipped: list[tuple[str, str]] = []
    to_restart: list[CatalogEntry] = []
    for entry in view.entries:
        if entry.is_broken:
            skipped.append((entry.name, "invalid YAML"))
            continue
        to_restart.append(entry)

    succeeded, failed = _run_in_parallel(
        to_restart,
        lambda entry: restart_model(
            project, entry.name, timeout=timeout, config_dir=config_dir
        ),
    )
    return BulkResult(
        profile=profile_name,
        action="restart",
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
    )


def _build_model_status(
    project: Project,
    model_name: str,
    metrics_port: int | None,
) -> ModelStatus:
    """Assemble a ModelStatus from already-known data, no catalog re-read."""
    paths = runtime_paths_for(project, model_name)
    pid = lifecycle.read_pid(paths.pid_path)
    alive = pid is not None and lifecycle.is_alive(pid)
    stale = pid is not None and not alive
    return ModelStatus(
        name=model_name,
        running=alive,
        pid=pid if alive else None,
        log_path=paths.log_path,
        pid_path=paths.pid_path,
        metrics_port=metrics_port,
        stale_pid_file=stale,
    )


def _short_yaml_error(exc: BaseException) -> str:
    """Trim a pydantic / yaml error to a single TUI-friendly line."""
    text = str(exc)
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            return stripped
    return text or exc.__class__.__name__


def start_model(
    project: Project,
    model_name: str,
    config_dir: Path | None = None,
) -> ModelStatus:
    """Spawn the vLLM process for a model in the background.

    vLLM downloads weights from HuggingFace on first start using HF_HOME for
    caching. HF_TOKEN must be in the shell or .env for gated models.
    """
    lifecycle.ensure_supported_platform()

    status = get_model_status(project, model_name, config_dir)
    if status.running:
        raise ModelAlreadyRunningError(model_name)

    args, model_env = build_command_args(project, model_name, config_dir)
    check_vllm_available(project, args[0])
    runtime_env = build_runtime_env(project, model_env)
    paths = runtime_paths_for(project, model_name)

    if paths.pid_path.exists():
        paths.pid_path.unlink()

    pid = lifecycle.spawn_detached(args, runtime_env, paths.log_path, paths.pid_path)
    return ModelStatus(
        name=model_name,
        running=True,
        pid=pid,
        log_path=paths.log_path,
        pid_path=paths.pid_path,
        metrics_port=status.metrics_port,
        stale_pid_file=False,
    )


def stop_model(
    project: Project,
    model_name: str,
    timeout: float = 30.0,
    config_dir: Path | None = None,
) -> ModelStatus:
    lifecycle.ensure_supported_platform()
    paths = runtime_paths_for(project, model_name)
    pid = lifecycle.read_pid(paths.pid_path)

    if pid is None or not lifecycle.is_alive(pid):
        if paths.pid_path.exists():
            paths.pid_path.unlink()
        raise ModelNotRunningError(model_name)

    lifecycle.terminate(pid, timeout=timeout)
    if paths.pid_path.exists():
        paths.pid_path.unlink()

    return get_model_status(project, model_name, config_dir)


def restart_model(
    project: Project,
    model_name: str,
    timeout: float = 30.0,
    config_dir: Path | None = None,
) -> ModelStatus:
    lifecycle.ensure_supported_platform()
    paths = runtime_paths_for(project, model_name)
    pid = lifecycle.read_pid(paths.pid_path)

    if pid is not None and lifecycle.is_alive(pid):
        lifecycle.terminate(pid, timeout=timeout)
    if paths.pid_path.exists():
        paths.pid_path.unlink()

    return start_model(project, model_name, config_dir)


def health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def metrics_url(host: str, port: int, path: str = "/metrics") -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{host}:{port}{path}"


def metrics_url_for_model(
    project: Project,
    model_name: str,
    *,
    host: str = "127.0.0.1",
    config_dir: Path | None = None,
) -> str | None:
    """Build the /metrics URL for a configured model. Returns None if no port available."""
    try:
        catalog = load_catalog_or_empty_for(project, config_dir)
    except Exception:
        return None
    model = catalog.get(model_name)
    if model is None or model.metrics_port is None:
        return None
    return metrics_url(host, model.metrics_port, model.metrics_path)


def probe_health(url: str, timeout: float = 1.5) -> bool:
    """Return True when the URL responds with 2xx, False on any other outcome."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 (local URL)
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def wait_for_ready(
    project: Project,
    model_name: str,
    *,
    timeout: float = 300.0,
    interval: float = 2.0,
    config_dir: Path | None = None,
    host: str = "127.0.0.1",
    on_progress: Callable[[float], None] | None = None,
) -> ModelStatus:
    """Block until /health responds, the process dies, or timeout expires.

    Raises ModelNotRunningError if the model has no PID file at all,
    ModelStartupFailedError if the process exits during the wait,
    ModelStartupTimeoutError if /health never responds in time.
    """
    paths = runtime_paths_for(project, model_name)
    pid = lifecycle.read_pid(paths.pid_path)
    if pid is None:
        raise ModelNotRunningError(model_name)

    metrics_port = _lookup_metrics_port(project, model_name, config_dir)
    if metrics_port is None:
        raise ValueError(f"{model_name}: cannot probe /health, no HTTP port configured")

    url = health_url(host, metrics_port)
    deadline = time.monotonic() + timeout
    started = time.monotonic()

    while True:
        if not lifecycle.is_alive(pid):
            raise ModelStartupFailedError(
                f"{model_name} process (pid {pid}) exited before /health responded"
            )

        if probe_health(url):
            return get_model_status(project, model_name, config_dir)

        if time.monotonic() >= deadline:
            raise ModelStartupTimeoutError(
                f"{model_name} did not respond on /health within {timeout:.0f}s"
            )

        if on_progress is not None:
            on_progress(time.monotonic() - started)
        time.sleep(interval)


def tail_log(project: Project, model_name: str, lines: int = 30) -> list[str]:
    """Return the last `lines` lines of a model's log file (empty list if missing)."""
    if lines <= 0:
        return []
    paths = runtime_paths_for(project, model_name)
    if not paths.log_path.is_file():
        return []
    block = 4096
    try:
        with paths.log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= lines:
                read = min(block, size)
                size -= read
                handle.seek(size)
                data = handle.read(read) + data
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]
