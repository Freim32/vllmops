"""LiteLLM gateway in front of the project's vLLM servers.

Generates a LiteLLM proxy config from the model catalog and manages the proxy
process, so every model is reachable through one OpenAI-compatible endpoint
where `model` selects the upstream.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from vllmops import lifecycle, service
from vllmops.config import ModelConfig, load_model_file
from vllmops.config_errors import explain_yaml_error
from vllmops.project import Project

# A model name must start with an alphanumeric (NAME_PATTERN_STR), so the
# leading underscore keeps these runtime files out of every model's namespace.
PROXY_RUNTIME_NAME = "_proxy"
GENERATED_CONFIG_NAME = "litellm.yaml"

MASTER_KEY_ENV = "LITELLM_MASTER_KEY"
READINESS_PATH = "/health/readiness"
# An automatic refresh waits this long for the respawned gateway before it
# reports the restart as not ready.
REFRESH_READY_TIMEOUT = 60.0

# vLLM answers OpenAI routes under /v1, and LiteLLM appends the route itself,
# so api_base carries the /v1 postfix and nothing beyond it.
UPSTREAM_API_SUFFIX = "/v1"

# The hosted_vllm prefix routes to a self-hosted vLLM server. Preferred over
# `openai/`, which requires a real API key on every request.
UPSTREAM_PROVIDER = "hosted_vllm"

KNOWN_OVERLAY_KEYS = frozenset(
    {
        "model_list",
        "litellm_settings",
        "general_settings",
        "router_settings",
        "environment_variables",
    }
)


class ProxyAlreadyRunningError(RuntimeError):
    """Raised when start is requested but the proxy is already running."""


class ProxyNotRunningError(RuntimeError):
    """Raised when stop or similar is requested but the proxy is not running."""


class ProxyStartupFailedError(RuntimeError):
    """Raised when the proxy process exits while we were waiting for readiness."""


class ProxyStartupTimeoutError(TimeoutError):
    """Raised when the proxy never answers on the readiness endpoint in time."""


class ProxyStopFailedError(RuntimeError):
    """Raised when a restart cannot take the old gateway down, so no new one is spawned."""


class LitellmExecutableNotFoundError(RuntimeError):
    """Raised when the litellm binary cannot be found in the project venv or on PATH."""


class NoProxyModelsError(RuntimeError):
    """Raised when no catalog model qualifies for the generated config.

    Carries the per-model skip reasons so the caller can explain the emptiness
    instead of just reporting it.
    """

    def __init__(self, skipped: list[tuple[str, str]]) -> None:
        super().__init__("no models available to proxy")
        self.skipped = skipped


@dataclass(frozen=True)
class ProxyOptions:
    """Fully resolved proxy inputs. Build with `config_options`."""

    host: str
    port: int
    upstream_host: str
    profile: str | None = None
    include_stopped: bool = False
    config_dir: Path | None = None
    num_workers: int = 1
    detailed_debug: bool = False


@dataclass(frozen=True)
class ProxyModelEntry:
    """One model as exposed by the gateway."""

    name: str
    served_model: str
    api_base: str
    api_key: str | None = None


@dataclass(frozen=True)
class ProxyConfigResult:
    config: dict[str, Any]
    models: list[ProxyModelEntry]
    skipped: list[tuple[str, str]]
    unknown_overlay_keys: list[str]


@dataclass(frozen=True)
class ProxyStartResult:
    pid: int
    url: str
    config_path: Path
    log_path: Path
    models: list[ProxyModelEntry]
    skipped: list[tuple[str, str]]
    unknown_overlay_keys: list[str]


@dataclass(frozen=True)
class ProxyStatus:
    running: bool
    pid: int | None
    port: int
    url: str
    log_path: Path
    config_path: Path
    configured_models: list[str]
    eligible_models: list[str]
    stale_pid_file: bool
    drift_reason: str | None = None

    @property
    def drifted(self) -> bool:
        """True when the config on disk no longer matches what a fresh one would hold."""
        return self.drift_reason is not None


@dataclass(frozen=True)
class ProxyRefresh:
    """Outcome of an automatic refresh triggered by a model state change."""

    action: Literal["unchanged", "restarted", "no-models"]
    reason: str
    pid: int
    # Only "restarted" carries the routed models: the other two actions write nothing.
    models: list[ProxyModelEntry] = field(default_factory=list)
    # Set when a respawned gateway did not answer on readiness: the restart itself
    # happened, so this is reported rather than raised.
    ready_error: str | None = None


def config_options(project: Project, *, config_dir: Path | None = None) -> ProxyOptions:
    """The only way to build proxy options: the `proxy` config section verbatim.

    `config_dir` is not part of the gateway's shape, it says which catalog to
    read, so it comes from the invoking command like everywhere else in the CLI.
    """
    cfg = project.config.proxy
    return ProxyOptions(
        host=cfg.host,
        port=cfg.port,
        upstream_host=cfg.upstream_host,
        profile=cfg.profile,
        include_stopped=cfg.expose == "all",
        config_dir=config_dir,
        num_workers=cfg.num_workers,
        detailed_debug=cfg.detailed_debug,
    )


def runtime_paths(project: Project) -> service.RuntimePaths:
    return service.runtime_paths_for(project, PROXY_RUNTIME_NAME)


def generated_config_path(project: Project) -> Path:
    return project.resolve("runtime") / GENERATED_CONFIG_NAME


def _probe_host(host: str) -> str:
    """Turn a bind address into something reachable.

    0.0.0.0 and :: mean "every interface" to a listener but are not destinations.
    """
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def base_url(host: str, port: int) -> str:
    return f"http://{_probe_host(host)}:{port}"


def readiness_url(host: str, port: int) -> str:
    return f"{base_url(host, port)}{READINESS_PATH}"


def resolve_litellm_executable(project: Project) -> str:
    """Pick the litellm binary, preferring the project venv.

    A custom `proxy.executable` is returned verbatim. Otherwise: project venv,
    then the interpreter's own bin dir (covers a litellm installed next to vllmops,
    as `uv tool install vllmops --with 'litellm[proxy]'` does), then PATH.
    """
    configured = project.config.proxy.executable
    if configured != "litellm":
        return configured

    interpreter_bin = Path(sys.executable).parent
    candidates = [
        project.root / ".venv" / "bin" / "litellm",
        project.root / ".venv" / "Scripts" / "litellm.exe",
        interpreter_bin / "litellm",
        interpreter_bin / "litellm.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return configured


def check_litellm_available(project: Project, executable: str) -> None:
    """Raise LitellmExecutableNotFoundError with an actionable hint if litellm is missing."""
    candidate = Path(executable)
    if candidate.is_absolute() or any(sep in executable for sep in ("/", "\\")):
        if not candidate.is_file():
            raise LitellmExecutableNotFoundError(f"litellm executable not found at: {candidate}")
        return
    if shutil.which(executable) is not None:
        return
    raise LitellmExecutableNotFoundError(
        f"litellm not found.\n"
        f"  Looked for: {project.root / '.venv' / 'bin' / 'litellm'}\n"
        f"  And in PATH for: {executable}\n\n"
        f"To install the LiteLLM proxy in this project:\n"
        f"  cd {project.root}\n"
        f"  uv add 'litellm[proxy]'\n"
        f"\n"
        f"Or set proxy.executable to a custom path in .vllmops/config.yaml."
    )


def _arg_value(model_cfg: ModelConfig, flag: str) -> str | None:
    """Read a vLLM flag's value from `args` (dict) or `extra_args` (flat list)."""
    args = model_cfg.vllm.args
    if flag in args:
        value = args[flag]
        if isinstance(value, list):
            return str(value[0]) if value else None
        if isinstance(value, bool):
            return None
        return str(value)
    extra = model_cfg.vllm.extra_args
    for index, item in enumerate(extra):
        if item == flag and index + 1 < len(extra):
            return extra[index + 1]
    return None


def _entries_for(project: Project, options: ProxyOptions) -> list[service.CatalogEntry]:
    if options.profile is None:
        return service.list_catalog_entries(project, options.config_dir)
    views = service.list_profiles(project, options.config_dir)
    for view in views:
        if view.name == options.profile:
            return view.entries
    raise service.UnknownProfileError(options.profile)


def eligible_models(
    entries: list[service.CatalogEntry],
    options: ProxyOptions,
) -> tuple[list[ProxyModelEntry], list[tuple[str, str]]]:
    """Split already-read catalog entries into routable models and skip reasons."""
    models: list[ProxyModelEntry] = []
    skipped: list[tuple[str, str]] = []

    for entry in entries:
        status = entry.status
        if entry.is_broken or status is None:
            skipped.append((entry.name, entry.error or "invalid YAML"))
            continue
        if not options.include_stopped and not status.running:
            skipped.append((entry.name, "not running"))
            continue
        if status.metrics_port is None:
            skipped.append((entry.name, "no HTTP port configured"))
            continue
        try:
            model_cfg = load_model_file(entry.yaml_path)
        except Exception as exc:
            skipped.append((entry.name, explain_yaml_error(exc).summary))
            continue

        models.append(
            ProxyModelEntry(
                name=entry.name,
                served_model=service.resolve_served_name(model_cfg),
                api_base=f"http://{options.upstream_host}:{status.metrics_port}{UPSTREAM_API_SUFFIX}",
                api_key=_arg_value(model_cfg, "--api-key"),
            )
        )

    return models, skipped


def _select_models(
    project: Project,
    options: ProxyOptions,
    entries: list[service.CatalogEntry] | None = None,
) -> tuple[list[ProxyModelEntry], list[tuple[str, str]]]:
    """`eligible_models` over the catalog, read here unless the caller already has it.

    Supplied entries are the default catalog in full, so a profile or a custom
    models directory reads it again instead of trusting a list that was built
    from somewhere else.
    """
    if entries is not None and options.profile is None and options.config_dir is None:
        return eligible_models(entries, options)
    return eligible_models(_entries_for(project, options), options)


def _model_list_entry(model: ProxyModelEntry) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": f"{UPSTREAM_PROVIDER}/{model.served_model}",
        "api_base": model.api_base,
    }
    if model.api_key is not None:
        params["api_key"] = model.api_key
    return {"model_name": model.name, "litellm_params": params}


def _master_key_is_set(project: Project) -> bool:
    if os.environ.get(MASTER_KEY_ENV):
        return True
    return bool(service.load_dotenv(project).get(MASTER_KEY_ENV))


def load_proxy_overlay(project: Project) -> dict[str, Any]:
    """Read the optional hand-written `.vllmops/litellm.yaml`."""
    path = project.proxy_overlay_path
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return raw


def merge_overlay(config: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Apply the overlay on top of the generated config.

    `model_list` entries are appended, so an overlay can add an endpoint vllmops
    knows nothing about. Mapping blocks merge key by key, everything else is
    replaced outright.
    """
    merged = dict(config)
    for key, value in overlay.items():
        if key == "model_list":
            if isinstance(value, list):
                generated = merged.get("model_list", [])
                merged["model_list"] = [*generated, *value]
            continue
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = {**current, **value}
        else:
            merged[key] = value
    return merged


def build_proxy_config(
    project: Project,
    options: ProxyOptions,
    *,
    entries: list[service.CatalogEntry] | None = None,
) -> ProxyConfigResult:
    """Build the LiteLLM config for the current catalog. Pure: nothing is written."""
    models, skipped = _select_models(project, options, entries)

    config: dict[str, Any] = {"model_list": [_model_list_entry(model) for model in models]}
    if _master_key_is_set(project):
        # The reference form, never the value: the generated file stays free of secrets.
        config["general_settings"] = {"master_key": f"os.environ/{MASTER_KEY_ENV}"}

    overlay = load_proxy_overlay(project)
    unknown = sorted(set(overlay) - KNOWN_OVERLAY_KEYS)

    return ProxyConfigResult(
        config=merge_overlay(config, overlay),
        models=models,
        skipped=skipped,
        unknown_overlay_keys=unknown,
    )


def render_proxy_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def write_proxy_config(project: Project, config: dict[str, Any]) -> Path:
    path = generated_config_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_proxy_config(config), encoding="utf-8")
    return path


def build_proxy_command(project: Project, config_path: Path, options: ProxyOptions) -> list[str]:
    """Build the litellm argv.

    num_workers defaults to 1 rather than LiteLLM's CPU count: vLLM is the
    bottleneck, and extra workers only multiply memory and split LiteLLM's
    in-process rate-limit state.
    """
    args = [
        resolve_litellm_executable(project),
        "--config",
        str(config_path),
        "--host",
        options.host,
        "--port",
        str(options.port),
        "--num_workers",
        str(options.num_workers),
    ]
    if options.detailed_debug:
        args.append("--detailed_debug")
    return args


def _check_port_free(project: Project, options: ProxyOptions) -> None:
    for entry in service.list_catalog_entries(project, options.config_dir):
        status = entry.status
        if status is None or not status.running:
            continue
        if status.metrics_port == options.port:
            raise service.PortConflictError(f"port {options.port} is already in use by running model {entry.name!r}")


@dataclass(frozen=True)
class _Plan:
    options: ProxyOptions
    result: ProxyConfigResult


def _plan(project: Project, options: ProxyOptions) -> _Plan:
    """Every check a launch needs, with nothing written and nothing stopped.

    A restart runs this while the old gateway still serves, so a config that
    cannot start leaves it in place.
    """
    check_litellm_available(project, resolve_litellm_executable(project))
    _check_port_free(project, options)

    result = build_proxy_config(project, options)
    if not result.models:
        raise NoProxyModelsError(result.skipped)
    return _Plan(options=options, result=result)


def _launch(project: Project, plan: _Plan) -> ProxyStartResult:
    paths = runtime_paths(project)
    config_path = write_proxy_config(project, plan.result.config)
    args = build_proxy_command(project, config_path, plan.options)

    paths.pid_path.unlink(missing_ok=True)
    pid = lifecycle.spawn_detached(args, service.build_runtime_env(project, {}), paths.log_path, paths.pid_path)

    return ProxyStartResult(
        pid=pid,
        url=base_url(plan.options.host, plan.options.port),
        config_path=config_path,
        log_path=paths.log_path,
        models=plan.result.models,
        skipped=plan.result.skipped,
        unknown_overlay_keys=plan.result.unknown_overlay_keys,
    )


def start_proxy(project: Project, options: ProxyOptions) -> ProxyStartResult:
    """Generate the config and spawn the LiteLLM proxy in the background."""
    lifecycle.ensure_supported_platform()
    pid = lifecycle.read_pid(runtime_paths(project).pid_path)
    if pid is not None and lifecycle.is_alive(pid):
        raise ProxyAlreadyRunningError(f"litellm proxy (pid {pid})")
    return _launch(project, _plan(project, options))


def stop_proxy(project: Project, timeout: float = 30.0) -> None:
    lifecycle.ensure_supported_platform()
    paths = runtime_paths(project)
    pid = lifecycle.read_pid(paths.pid_path)

    if pid is None or not lifecycle.is_alive(pid):
        paths.pid_path.unlink(missing_ok=True)
        raise ProxyNotRunningError("litellm proxy")

    lifecycle.terminate(pid, timeout=timeout)
    paths.pid_path.unlink(missing_ok=True)


def restart_proxy(project: Project, options: ProxyOptions, timeout: float = 30.0) -> ProxyStartResult:
    """Start the proxy with a freshly generated config, replacing a running one.

    The new config is checked before the old process is touched, so a restart
    that cannot succeed leaves the running gateway and its config file alone.
    """
    lifecycle.ensure_supported_platform()
    plan = _plan(project, options)
    paths = runtime_paths(project)
    pid = lifecycle.read_pid(paths.pid_path)

    if pid is not None and lifecycle.is_alive(pid) and not lifecycle.terminate(pid, timeout=timeout):
        raise ProxyStopFailedError(f"old litellm proxy (pid {pid}) did not exit; not respawning on port {options.port}")
    paths.pid_path.unlink(missing_ok=True)

    return _launch(project, plan)


def _model_names_in_text(text: str) -> list[str]:
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return []
    entries = raw.get("model_list") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return []
    return [
        item["model_name"] for item in entries if isinstance(item, dict) and isinstance(item.get("model_name"), str)
    ]


def _configured_model_names(config_path: Path) -> list[str]:
    """Read the model aliases out of a generated config, tolerating a missing or broken file."""
    try:
        return _model_names_in_text(config_path.read_text(encoding="utf-8"))
    except OSError:
        return []


def _describe_change(before: list[str], after: list[str]) -> str:
    added = [name for name in after if name not in before]
    removed = [name for name in before if name not in after]
    parts = []
    if added:
        parts.append("added " + ", ".join(added))
    if removed:
        parts.append("removed " + ", ".join(removed))
    return "; ".join(parts) if parts else "config changed"


def _drift(
    project: Project,
    options: ProxyOptions,
    entries: list[service.CatalogEntry] | None = None,
) -> tuple[str | None, ProxyConfigResult]:
    """Compare a freshly generated config against the one on disk. Nothing is written.

    Returns the reason it moved, or None when they match. Comparing the rendered
    YAML rather than the model names catches a model that changed port and an
    edited overlay too.
    """
    result = build_proxy_config(project, options, entries=entries)
    rendered = render_proxy_config(result.config)
    try:
        current = generated_config_path(project).read_text(encoding="utf-8")
    except OSError:
        return "generated config missing", result

    if current == rendered:
        return None, result
    return _describe_change(_model_names_in_text(current), _model_names_in_text(rendered)), result


def proxy_status(
    project: Project,
    options: ProxyOptions,
    *,
    entries: list[service.CatalogEntry] | None = None,
) -> ProxyStatus:
    paths = runtime_paths(project)
    pid = lifecycle.read_pid(paths.pid_path)
    alive = pid is not None and lifecycle.is_alive(pid)
    config_path = generated_config_path(project)
    reason, result = _drift(project, options, entries)

    return ProxyStatus(
        running=alive,
        pid=pid if alive else None,
        port=options.port,
        url=base_url(options.host, options.port),
        log_path=paths.log_path,
        config_path=config_path,
        configured_models=_configured_model_names(config_path),
        eligible_models=[model.name for model in result.models],
        stale_pid_file=pid is not None and not alive,
        # A config that was never generated cannot be out of date, and the status
        # line already says so.
        drift_reason=None if not config_path.is_file() else reason,
    )


def refresh_proxy(
    project: Project,
    *,
    config_dir: Path | None = None,
    entries: list[service.CatalogEntry] | None = None,
    ready_timeout: float = REFRESH_READY_TIMEOUT,
) -> ProxyRefresh | None:
    """Bring a running gateway back in line with the catalog.

    Returns None when no gateway is running: one that is down is never started,
    and one that died on its own is never resurrected. A respawn returns only
    once the new gateway answers on readiness, or with `ready_error` set.
    """
    pid = lifecycle.read_pid(runtime_paths(project).pid_path)
    if pid is None or not lifecycle.is_alive(pid):
        return None

    options = config_options(project, config_dir=config_dir)
    reason, result = _drift(project, options, entries)

    if not result.models:
        # Respawning here would mean a gateway with an empty model_list, so the
        # last config stays in place: a stale entry answers with an error, which
        # beats a gateway that is not there at all.
        return ProxyRefresh("no-models", "no model qualifies", pid=pid)
    if reason is None:
        return ProxyRefresh("unchanged", "config up to date", pid=pid)

    restarted = restart_proxy(project, options)
    try:
        wait_for_proxy_ready(project, options, timeout=ready_timeout)
    except (ProxyStartupFailedError, ProxyStartupTimeoutError) as exc:
        return ProxyRefresh("restarted", reason, pid=restarted.pid, models=restarted.models, ready_error=str(exc))
    return ProxyRefresh("restarted", reason, pid=restarted.pid, models=restarted.models)


def wait_for_proxy_ready(
    project: Project,
    options: ProxyOptions,
    *,
    timeout: float = 120.0,
    interval: float = 1.0,
    on_progress: Callable[[float], None] | None = None,
) -> int:
    """Block until the proxy answers on readiness, the process dies, or timeout expires.

    Probes /health/readiness, which needs no auth and does not call the upstream
    models, unlike /health.
    """
    paths = runtime_paths(project)
    pid = lifecycle.read_pid(paths.pid_path)
    if pid is None:
        raise ProxyNotRunningError("litellm proxy")

    url = readiness_url(options.host, options.port)
    deadline = time.monotonic() + timeout
    started = time.monotonic()

    while True:
        if not lifecycle.is_alive(pid):
            raise ProxyStartupFailedError(f"litellm proxy (pid {pid}) exited before {READINESS_PATH} responded")

        if service.probe_health(url):
            return pid

        if time.monotonic() >= deadline:
            raise ProxyStartupTimeoutError(f"litellm proxy did not respond on {READINESS_PATH} within {timeout:.0f}s")

        if on_progress is not None:
            on_progress(time.monotonic() - started)
        time.sleep(interval)
