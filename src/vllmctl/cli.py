import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from vllmctl import __version__, service
from vllmctl.project import Project
from vllmctl.service import (
    ModelAlreadyExistsError,
    ModelAlreadyRunningError,
    ModelNotRunningError,
    ModelStartupFailedError,
    ModelStartupTimeoutError,
    ModelStatus,
    UnknownModelError,
    VllmExecutableNotFoundError,
)

app = typer.Typer(help="Manage bare-metal vLLM models with a TUI for live metrics.")
profile_app = typer.Typer(help="Inspect model profiles defined in .vllmctl/config.yaml.")
app.add_typer(profile_app, name="profile")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"vllmctl {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Manage bare-metal vLLM models with a TUI for live metrics."""
    del version  # handled in the callback


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Directory to initialize as a vllmctl project."),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Project name (defaults to a sanitized version of the folder name).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing project files (config.yaml, pyproject.toml, .env.example, ...).",
    ),
) -> None:
    """Initialize a vllmctl project workspace.

    Generates the directory layout, .vllmctl/config.yaml, a pyproject.toml with
    `vllm` as a dependency, .python-version, .env.example, and .gitignore.
    Run `uv sync` (or equivalent) inside the project to create the .venv with
    vLLM installed.
    """
    try:
        written = service.initialize_workspace(path, force=force, name=name)
    except FileExistsError as exc:
        console.print(f"[bold red]Already initialized:[/bold red] {exc}")
        console.print("Use --force to rewrite project files.")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[bold red]Invalid name:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Init failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    project = service.get_project(path)
    console.print(
        f"[bold green]Initialized vllmctl project[/bold green] {project.name} at {project.root}"
    )
    for file in written:
        console.print(f"[green]wrote[/green] {file}")
    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"  [cyan]cd {project.root}[/cyan]")
    console.print("  [cyan]uv sync[/cyan]                   # creates .venv and installs vLLM")
    console.print("  [cyan]vllmctl create-model ...[/cyan]")
    console.print("  [cyan]vllmctl start <name>[/cyan]")


@app.command("create-model")
def create_model(
    name: str | None = typer.Option(None, "--name", "-n", help="Mnemonic model name used by vllmctl."),
    hf_model: str | None = typer.Option(None, "--model", "-m", help="HuggingFace model id or local path."),
    gpus: str | None = typer.Option(None, "--gpus", "-g", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1."),
    port: int | None = typer.Option(None, "--port", "-p", help="vLLM HTTP port."),
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Where to write the YAML."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing model YAML."),
) -> None:
    """Create a minimal editable YAML config for one model."""
    project = service.get_project()

    try:
        default_port = service.next_available_port(project, config_dir)
    except Exception as exc:
        console.print(f"[bold red]Cannot read existing configs:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    resolved_name: str = name if name is not None else typer.prompt("Mnemonic name")
    resolved_hf_model: str = (
        hf_model if hf_model is not None else typer.prompt("HuggingFace model or local path")
    )
    resolved_gpus: str = (
        gpus if gpus is not None else typer.prompt("GPUs to use (CUDA_VISIBLE_DEVICES)", default="0")
    )
    resolved_port: int = (
        port if port is not None else typer.prompt("vLLM port", default=default_port, type=int)
    )

    effective_dir = service.resolve_models_dir(project, config_dir)
    destination = effective_dir / f"{resolved_name}.yaml"
    if destination.exists() and not force:
        if not typer.confirm(f"{destination} already exists. Overwrite?", default=False):
            console.print("[yellow]aborted[/yellow]")
            raise typer.Exit(code=1)
        force = True

    try:
        result = service.create_model(
            project,
            name=resolved_name,
            hf_model=resolved_hf_model,
            gpus=resolved_gpus,
            port=resolved_port,
            config_dir=config_dir,
            force=force,
        )
    except ModelAlreadyExistsError as exc:
        console.print(f"[bold red]Already exists:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Cannot create config:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]wrote[/green] {result.destination}")
    console.print("Edit this YAML by hand for advanced vLLM args, then run:")
    console.print("  vllmctl validate")
    console.print(f"  vllmctl start {resolved_name}")


@app.command()
def validate(
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
) -> None:
    """Validate model YAML files."""
    project = service.get_project()

    try:
        catalog = service.load_catalog_for(project, config_dir)
    except Exception as exc:
        console.print(f"[bold red]Invalid configuration:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="vLLM model catalog")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Port", justify="right")
    table.add_column("Env")
    table.add_column("Args", justify="right")

    for model in catalog.models:
        table.add_row(
            model.name,
            model.vllm.model,
            str(model.metrics_port or "-"),
            str(len(model.env)),
            str(len(model.vllm.args) + len(model.vllm.flags) + len(model.vllm.extra_args)),
        )

    console.print(table)
    console.print(f"[bold green]OK[/bold green] {len(catalog.models)} model(s) valid")


@app.command()
def command(
    model_name: str = typer.Argument(..., help="Configured model name."),
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
) -> None:
    """Print the bare-metal vLLM command for a configured model."""
    project = service.get_project()

    try:
        rendered = service.build_command_string(project, model_name, config_dir=config_dir)
    except UnknownModelError as exc:
        console.print(f"[bold red]Unknown model:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Invalid configuration:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(rendered)


def _print_status(status: ModelStatus) -> None:
    state = "[bold green]running[/bold green]" if status.running else "[red]stopped[/red]"
    pid_display = str(status.pid) if status.pid is not None else "-"
    metrics_display = (
        f"http://localhost:{status.metrics_port}/metrics" if status.metrics_port else "-"
    )
    stale = " [yellow](stale pid file)[/yellow]" if status.stale_pid_file else ""
    console.print(f"{status.name}: {state} pid={pid_display} metrics={metrics_display}{stale}")


def _print_bulk_result(result: service.BulkResult) -> None:
    summary_parts: list[str] = []
    if result.succeeded:
        summary_parts.append(f"[green]{len(result.succeeded)} {result.action}ed[/green]")
    if result.skipped:
        summary_parts.append(f"[yellow]{len(result.skipped)} skipped[/yellow]")
    if result.failed:
        summary_parts.append(f"[bold red]{len(result.failed)} failed[/bold red]")
    if not summary_parts:
        summary_parts.append("[dim]nothing to do[/dim]")
    console.print(f"[bold]{result.profile}[/bold]: " + ", ".join(summary_parts))

    for name, reason in result.skipped:
        console.print(f"  [yellow]skip[/yellow] {name} [dim]({reason})[/dim]")
    for name, error in result.failed:
        console.print(f"  [red]fail[/red] {name} [dim]{error}[/dim]")


def _require_one_target(
    model_name: str | None, profile: str | None, command: str
) -> None:
    """Enforce exactly one of model_name or --profile."""
    if model_name is None and profile is None:
        console.print(
            f"[bold red]Usage:[/bold red] vllmctl {command} <model_name> | --profile <name>"
        )
        raise typer.Exit(code=2)
    if model_name is not None and profile is not None:
        console.print(
            "[bold red]Provide either a model name or --profile, not both[/bold red]"
        )
        raise typer.Exit(code=2)


def _wait_for_profile_in_parallel(
    project: Project,
    names: list[str],
    timeout: float,
    config_dir: Path | None,
    health_host: str,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Probe /health on each name concurrently. Returns (ready, failed)."""
    if not names:
        return [], []
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    ready: list[str] = []
    failed: list[tuple[str, str]] = []
    max_workers = min(len(names), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                service.wait_for_ready,
                project,
                name,
                timeout=timeout,
                config_dir=config_dir,
                host=health_host,
            ): name
            for name in names
        }
        for future, name in futures.items():
            try:
                future.result()
                ready.append(name)
            except Exception as exc:
                failed.append((name, _format_exc(exc)))
    return ready, failed


def _format_exc(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else exc.__class__.__name__


def _print_log_tail(project: Project, model_name: str, lines: int = 30) -> None:
    tail = service.tail_log(project, model_name, lines=lines)
    if not tail:
        return
    console.print(f"[dim]--- last {len(tail)} lines of log ---[/dim]")
    for line in tail:
        console.print(line)
    console.print("[dim]--- end of log ---[/dim]")


@app.command()
def start(
    model_name: str | None = typer.Argument(None, help="Configured model name."),
    profile: str | None = typer.Option(
        None, "--profile", "-P", help="Start every model in this profile in parallel."
    ),
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Block until /health responds (default). Use --no-wait to fire and forget.",
    ),
    wait_timeout: float = typer.Option(300.0, "--wait-timeout", help="Seconds to wait for /health."),
    health_host: str = typer.Option("127.0.0.1", "--health-host", help="Host for /health probe."),
) -> None:
    """Start a vLLM server in the background.

    Blocks until vLLM responds on /health (default). vLLM downloads the model
    from HuggingFace on first start; HF_TOKEN is read from .env or the shell.
    """
    _require_one_target(model_name, profile, "start")
    project = service.get_project()

    if profile is not None:
        try:
            result = service.start_profile(project, profile, config_dir=config_dir)
        except service.UnknownProfileError as exc:
            console.print(f"[bold red]Unknown profile:[/bold red] {profile}")
            raise typer.Exit(code=1) from exc

        _print_bulk_result(result)

        if wait and result.succeeded:
            console.print(
                f"[dim]waiting on /health for {len(result.succeeded)}"
                f" model(s) in parallel...[/dim]"
            )
            ready_names, wait_failed = _wait_for_profile_in_parallel(
                project, result.succeeded, wait_timeout, config_dir, health_host
            )
            for name in ready_names:
                console.print(f"  [green]ready[/green] {name}")
            for name, err in wait_failed:
                console.print(f"  [red]not ready[/red] {name} [dim]{err}[/dim]")
            if wait_failed:
                raise typer.Exit(code=1)

        raise typer.Exit(code=1 if result.failed else 0)

    assert model_name is not None  # narrowed by _require_one_target
    try:
        status = service.start_model(project, model_name, config_dir=config_dir)
    except ModelAlreadyRunningError as exc:
        console.print(f"[yellow]{exc} is already running[/yellow]")
        raise typer.Exit(code=1) from exc
    except UnknownModelError as exc:
        console.print(f"[bold red]Unknown model:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except VllmExecutableNotFoundError as exc:
        console.print(f"[bold red]Cannot start:[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[bold red]Cannot start:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Cannot start:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    dotenv_count = len(service.load_dotenv(project))
    console.print(f"[green]spawned[/green] {model_name} pid={status.pid}")
    console.print(f"  logs: {status.log_path}")
    if status.metrics_port:
        console.print(f"  http: http://{health_host}:{status.metrics_port}")
    if dotenv_count:
        console.print(f"  loaded {dotenv_count} var(s) from .env (shell takes precedence)")

    if not wait:
        console.print(f"  follow startup with: vllmctl logs {model_name} --follow")
        return

    if status.metrics_port is None:
        console.print("[yellow]no HTTP port configured, skipping /health wait[/yellow]")
        return

    try:
        with console.status(
            f"[cyan]waiting for {model_name} on /health (timeout {wait_timeout:.0f}s)...[/cyan]"
        ) as spinner:
            ready = service.wait_for_ready(
                project,
                model_name,
                timeout=wait_timeout,
                config_dir=config_dir,
                host=health_host,
                on_progress=lambda elapsed: spinner.update(
                    f"[cyan]waiting on /health... {int(elapsed)}s elapsed[/cyan]"
                ),
            )
    except ModelStartupFailedError as exc:
        console.print(f"[bold red]Startup failed:[/bold red] {exc}")
        _print_log_tail(project, model_name)
        raise typer.Exit(code=1) from exc
    except ModelStartupTimeoutError as exc:
        console.print(f"[bold red]Timeout:[/bold red] {exc}")
        console.print(f"  process is still running; tail with: vllmctl logs {model_name} --follow")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Wait failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]ready[/bold green] {model_name} pid={ready.pid}")


@app.command()
def stop(
    model_name: str | None = typer.Argument(None, help="Configured model name."),
    profile: str | None = typer.Option(
        None, "--profile", "-P", help="Stop every running model in this profile."
    ),
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
    timeout: float = typer.Option(30.0, "--timeout", "-t", help="Seconds before SIGKILL."),
) -> None:
    """Stop a running vLLM server (SIGTERM, then SIGKILL after timeout)."""
    _require_one_target(model_name, profile, "stop")
    project = service.get_project()

    if profile is not None:
        try:
            result = service.stop_profile(
                project, profile, config_dir=config_dir, timeout=timeout
            )
        except service.UnknownProfileError as exc:
            console.print(f"[bold red]Unknown profile:[/bold red] {profile}")
            raise typer.Exit(code=1) from exc
        _print_bulk_result(result)
        raise typer.Exit(code=1 if result.failed else 0)

    assert model_name is not None
    try:
        service.stop_model(project, model_name, timeout=timeout)
    except ModelNotRunningError as exc:
        console.print(f"[yellow]{exc} is not running[/yellow]")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[bold red]Cannot stop:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]stopped[/green] {model_name}")


@app.command()
def restart(
    model_name: str | None = typer.Argument(None, help="Configured model name."),
    profile: str | None = typer.Option(
        None, "--profile", "-P", help="Restart every model in this profile in parallel."
    ),
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
    timeout: float = typer.Option(30.0, "--timeout", "-t", help="Seconds before SIGKILL."),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Block until /health responds."),
    wait_timeout: float = typer.Option(300.0, "--wait-timeout", help="Seconds to wait for /health."),
    health_host: str = typer.Option("127.0.0.1", "--health-host", help="Host for /health probe."),
) -> None:
    """Restart a vLLM server (stop if running, then start)."""
    _require_one_target(model_name, profile, "restart")
    project = service.get_project()

    if profile is not None:
        try:
            result = service.restart_profile(
                project, profile, config_dir=config_dir, timeout=timeout
            )
        except service.UnknownProfileError as exc:
            console.print(f"[bold red]Unknown profile:[/bold red] {profile}")
            raise typer.Exit(code=1) from exc

        _print_bulk_result(result)

        if wait and result.succeeded:
            console.print(
                f"[dim]waiting on /health for {len(result.succeeded)}"
                f" model(s) in parallel...[/dim]"
            )
            ready_names, wait_failed = _wait_for_profile_in_parallel(
                project, result.succeeded, wait_timeout, config_dir, health_host
            )
            for name in ready_names:
                console.print(f"  [green]ready[/green] {name}")
            for name, err in wait_failed:
                console.print(f"  [red]not ready[/red] {name} [dim]{err}[/dim]")
            if wait_failed:
                raise typer.Exit(code=1)

        raise typer.Exit(code=1 if result.failed else 0)

    assert model_name is not None
    try:
        status = service.restart_model(project, model_name, timeout=timeout, config_dir=config_dir)
    except UnknownModelError as exc:
        console.print(f"[bold red]Unknown model:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except VllmExecutableNotFoundError as exc:
        console.print(f"[bold red]Cannot restart:[/bold red]\n{exc}")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[bold red]Cannot restart:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]Cannot restart:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]respawned[/green] {model_name} pid={status.pid}")

    if not wait or status.metrics_port is None:
        return

    try:
        with console.status(
            f"[cyan]waiting for {model_name} on /health (timeout {wait_timeout:.0f}s)...[/cyan]"
        ) as spinner:
            ready = service.wait_for_ready(
                project,
                model_name,
                timeout=wait_timeout,
                config_dir=config_dir,
                host=health_host,
                on_progress=lambda elapsed: spinner.update(
                    f"[cyan]waiting on /health... {int(elapsed)}s elapsed[/cyan]"
                ),
            )
    except ModelStartupFailedError as exc:
        console.print(f"[bold red]Startup failed:[/bold red] {exc}")
        _print_log_tail(project, model_name)
        raise typer.Exit(code=1) from exc
    except ModelStartupTimeoutError as exc:
        console.print(f"[bold red]Timeout:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]ready[/bold green] {model_name} pid={ready.pid}")


@app.command()
def health(
    model_name: str = typer.Argument(..., help="Configured model name."),
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host for /health probe."),
    timeout: float = typer.Option(2.0, "--timeout", "-t", help="HTTP timeout."),
) -> None:
    """One-shot probe of a model's /health endpoint."""
    project = service.get_project()
    model_status = service.get_model_status(project, model_name, config_dir)
    if model_status.metrics_port is None:
        console.print("[bold red]No HTTP port configured for this model[/bold red]")
        raise typer.Exit(code=1)

    url = service.health_url(host, model_status.metrics_port)
    if service.probe_health(url, timeout=timeout):
        console.print(f"[bold green]healthy[/bold green] {url}")
        return
    console.print(f"[bold red]unhealthy[/bold red] {url}")
    raise typer.Exit(code=1)


@app.command()
def status(
    model_name: str | None = typer.Argument(None, help="Configured model name (omit for all)."),
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
) -> None:
    """Show running state for one model or for the whole catalog."""
    project = service.get_project()

    if model_name is not None:
        _print_status(service.get_model_status(project, model_name, config_dir))
        return

    try:
        statuses = service.list_model_statuses(project, config_dir)
    except Exception as exc:
        console.print(f"[bold red]Cannot read configs:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not statuses:
        console.print("[yellow]no models configured[/yellow]")
        return

    table = Table(title="vLLM model status")
    table.add_column("Name")
    table.add_column("State")
    table.add_column("PID", justify="right")
    table.add_column("Port", justify="right")
    table.add_column("Notes")
    for s in statuses:
        table.add_row(
            s.name,
            "running" if s.running else "stopped",
            str(s.pid) if s.pid is not None else "-",
            str(s.metrics_port) if s.metrics_port else "-",
            "stale pid file" if s.stale_pid_file else "",
        )
    console.print(table)


def _read_last_lines(path: Path, n: int) -> list[str]:
    if n <= 0:
        return []
    block = 4096
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        data = b""
        while size > 0 and data.count(b"\n") <= n:
            read = min(block, size)
            size -= read
            handle.seek(size)
            data = handle.read(read) + data
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()[-n:]


def _follow_log(path: Path) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        try:
            while True:
                line = handle.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                print(line, end="")
        except KeyboardInterrupt:
            return


@app.command()
def logs(
    model_name: str = typer.Argument(..., help="Configured model name."),
    tail: int = typer.Option(0, "--tail", "-n", help="Print last N lines (0 = path only)."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow the log (Ctrl-C to stop)."),
) -> None:
    """Print or follow a model's log file."""
    project = service.get_project()
    paths = service.runtime_paths_for(project, model_name)

    if not paths.log_path.exists():
        console.print(f"[yellow]no log yet:[/yellow] {paths.log_path}")
        return

    if not tail and not follow:
        console.print(str(paths.log_path))
        return

    if tail:
        for line in _read_last_lines(paths.log_path, tail):
            print(line)

    if follow:
        _follow_log(paths.log_path)


@profile_app.command("list")
def profile_list(
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
) -> None:
    """List all profiles with running/total counts."""
    project = service.get_project()
    views = service.list_profiles(project, config_dir)
    renderable = [v for v in views if v.entries]
    if not renderable:
        console.print("[yellow]no models in any profile[/yellow]")
        return

    table = Table(title="Profiles")
    table.add_column("Profile")
    table.add_column("Models", justify="right")
    table.add_column("Running", justify="right")
    table.add_column("Missing", justify="right")
    table.add_column("Members")
    for view in renderable:
        members = ", ".join(entry.name for entry in view.entries)
        style = "dim" if view.is_general else ""
        if view.is_general:
            missing = "-"
        elif view.missing:
            missing = f"[yellow]{len(view.missing)}[/yellow]"
        else:
            missing = "0"
        table.add_row(
            f"[{style}]{view.name}[/{style}]" if style else view.name,
            str(view.total_count),
            str(view.running_count),
            missing,
            members,
        )
    console.print(table)
    if any(v.missing for v in renderable):
        console.print(
            "[dim]profiles with missing members: run"
            " `vllmctl profile show <name>` for details[/dim]"
        )


@profile_app.command("show")
def profile_show(
    profile_name: str = typer.Argument(..., help="Profile name (use 'general' for unassigned models)."),
    config_dir: Path | None = typer.Option(None, "--config-dir", "-c", help="Models directory."),
) -> None:
    """Show a profile's members, their state, and any declared-but-missing models."""
    project = service.get_project()
    views = service.list_profiles(project, config_dir)
    view = next((v for v in views if v.name == profile_name), None)
    if view is None:
        console.print(f"[bold red]Unknown profile:[/bold red] {profile_name}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Profile:[/bold] {view.name}")
    if view.entries:
        table = Table(show_header=True)
        table.add_column("Name")
        table.add_column("State")
        table.add_column("Port", justify="right")
        table.add_column("PID", justify="right")
        for entry in view.entries:
            status = entry.status
            if entry.is_broken:
                state = "[red]invalid[/red]"
                port = "-"
                pid = "-"
            elif status is not None and status.running:
                state = "[green]running[/green]"
                port = str(status.metrics_port or "-")
                pid = str(status.pid or "-")
            else:
                state = "stopped"
                port = str(status.metrics_port or "-") if status else "-"
                pid = "-"
            table.add_row(entry.name, state, port, pid)
        console.print(table)
    else:
        console.print("[dim]no models in this profile[/dim]")

    if view.missing:
        console.print(
            f"\n[yellow]declared but not in catalog:[/yellow] {', '.join(view.missing)}"
        )


@app.command()
def tui(
    health_host: str = typer.Option(
        "127.0.0.1",
        "--health-host",
        help="Host used to scrape vLLM /metrics and probe /health.",
    ),
    theme: str = typer.Option(
        "monokai",
        "--theme",
        help="Textual theme. Cycle in-app with Ctrl+T or use the command palette (Ctrl+P).",
    ),
) -> None:
    """Launch the Textual TUI for live model lifecycle and metrics.

    Metrics are scraped directly from each model's /metrics endpoint (Prometheus
    text format) and kept in an in-memory ring buffer for the duration of the
    session. No external Prometheus or Docker stack required.
    """
    # Lazy import: keeps textual out of the import path of non-TUI commands.
    from vllmctl.tui import VllmctlApp  # noqa: PLC0415
    from vllmctl.tui.app import TuiOptions  # noqa: PLC0415

    project = service.get_project()
    options = TuiOptions(project=project, health_host=health_host, theme=theme)
    VllmctlApp(options).run()


if __name__ == "__main__":
    app()
