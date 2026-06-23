"""Main Textual application for vllmctl."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from vllmctl import gpu as gpu_module
from vllmctl import metrics, service
from vllmctl.gpu import GpuSnapshot
from vllmctl.metrics import MetricsHistory, snapshot_from_history
from vllmctl.project import Project
from vllmctl.service import (
    ModelAlreadyRunningError,
    ModelNotRunningError,
    UnknownModelError,
)
from vllmctl.tui.widgets import ErrorsPanel, GpuPanel, LogViewer, MetricsPanel, ModelsTree

STATUS_REFRESH_SECONDS = 2.0
LOG_POLL_SECONDS = 0.5
METRICS_SCRAPE_SECONDS = 5.0
HISTORY_CAPACITY = 360  # ~30 minutes at 5s scrape interval


@dataclass(frozen=True)
class TuiOptions:
    project: Project
    health_host: str = "127.0.0.1"
    theme: str = "monokai"


class VllmctlApp(App):
    """Three-pane TUI: models list, log tail, live metrics scraped from vLLM."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        layout: horizontal;
        height: 1fr;
        padding: 0 1;
    }
    #sidebar {
        width: 42;
        height: 1fr;
    }
    #right {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }
    #right > LogViewer {
        height: 2fr;
    }
    #right > #metrics-row {
        layout: horizontal;
        height: 1fr;
    }
    #right > #metrics-row > MetricsPanel {
        width: 1fr;
    }
    #right > #metrics-row > GpuPanel {
        width: 1fr;
    }
    #right > #metrics-row > ErrorsPanel {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("R", "refresh_now", "Refresh"),
        Binding("s", "start_model", "Start"),
        Binding("S", "stop_model", "Stop"),
        Binding("r", "restart_model", "Restart"),
        Binding("e", "edit_model", "Edit YAML"),
        Binding("t", "smoke_test", "Smoke test"),
        Binding("c", "copy_logs", "Copy logs"),
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
    ]

    LOG_CLIPBOARD_LIMIT_BYTES = 1_000_000  # 1 MB cap so we don't blow up OSC 52

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        del parameters
        if action == "start_model":
            return self._any_target_entry(service.can_start)
        if action == "stop_model":
            return self._any_target_entry(
                lambda e: service.can_stop(self._options.project, e)
            )
        if action == "restart_model":
            return self._any_target_entry(service.can_restart)
        if action == "smoke_test":
            # Smoke test runs on a single model, never on a whole profile.
            entry = self._selected_model_entry()
            return entry is not None and service.can_smoke_test(entry)
        if action == "edit_model":
            return self._models.selected_model_name is not None
        if action == "copy_logs":
            entry = self._selected_model_entry()
            return entry is not None and service.can_copy_logs(
                self._options.project, entry
            )
        return True

    def _any_target_entry(
        self, predicate: Callable[[service.CatalogEntry], bool]
    ) -> bool:
        """True if at least one entry under the current selection passes the predicate."""
        entries = self._target_entries()
        return any(predicate(entry) for entry in entries)

    def _selected_model_entry(self) -> service.CatalogEntry | None:
        name = self._models.selected_model_name
        return self._entry_for(name) if name is not None else None

    def _target_entries(self) -> list[service.CatalogEntry]:
        """Resolve the current selection to entries (1 if model, N if profile)."""
        entry = self._selected_model_entry()
        if entry is not None:
            return [entry]
        profile = self._models.selected_profile_name
        if profile is not None:
            for view in self._profile_views:
                if view.name == profile:
                    return list(view.entries)
        return []

    def __init__(self, options: TuiOptions) -> None:
        super().__init__()
        self._options = options
        self._models = ModelsTree()
        self._logs = LogViewer()
        self._metrics = MetricsPanel()
        self._gpu_panel = GpuPanel()
        self._errors = ErrorsPanel()
        self._busy = False
        self._last_attached_log: str | None = None
        self._histories: dict[str, MetricsHistory] = {}
        self._entries: list[service.CatalogEntry] = []
        self._profile_views: list[service.ProfileView] = []
        self._gpu_snapshots: list[GpuSnapshot] = []
        self._gpu_indices: dict[str, list[int]] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield self._models
            with Vertical(id="right"):
                yield self._logs
                with Horizontal(id="metrics-row"):
                    yield self._metrics
                    yield self._gpu_panel
                    yield self._errors
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "vllmctl"
        self.sub_title = self._options.project.name
        self.theme = self._options.theme
        self._errors.display = False
        self._refresh_statuses()
        self.set_interval(STATUS_REFRESH_SECONDS, self._refresh_statuses)
        self.set_interval(LOG_POLL_SECONDS, self._logs.poll)
        self.set_interval(METRICS_SCRAPE_SECONDS, self._scrape_all_async)
        self.run_worker(self._scrape_all(), exclusive=False)

    def _update_header_subtitle(self) -> None:
        """Header now shows project + counts of running/stopped/invalid models."""
        running = stopped = invalid = 0
        for entry in self._entries:
            if entry.is_broken:
                invalid += 1
            elif entry.status is not None and entry.status.running:
                running += 1
            else:
                stopped += 1
        parts = [self._options.project.name]
        if running:
            parts.append(f"{running} running")
        if stopped:
            parts.append(f"{stopped} stopped")
        if invalid:
            parts.append(f"{invalid} invalid")
        if running == 0 and stopped == 0 and invalid == 0:
            parts.append("no models")
        self.sub_title = " · ".join(parts)

    # --- helpers ---

    def _history_for(self, name: str) -> MetricsHistory:
        history = self._histories.get(name)
        if history is None:
            history = MetricsHistory(capacity=HISTORY_CAPACITY)
            self._histories[name] = history
        return history

    # --- refresh loops ---

    def _refresh_statuses(self) -> None:
        try:
            views = service.list_profiles(self._options.project)
        except Exception as exc:
            self.notify(
                f"failed to read catalog: {exc}",
                severity="error",
                timeout=5,
                markup=False,
            )
            return
        self._profile_views = views
        self._entries = [entry for view in views for entry in view.entries]
        self._refresh_gpu_indices()
        self._models.render_profiles(views)
        self._update_header_subtitle()
        self._sync_log_attachment()
        self._render_metrics_panel()
        self.refresh_bindings()

    def _refresh_gpu_indices(self) -> None:
        """Re-derive per-model CUDA_VISIBLE_DEVICES from the current catalog."""
        from vllmctl.config import load_model_file  # noqa: PLC0415

        new_indices: dict[str, list[int]] = {}
        for entry in self._entries:
            if entry.is_broken:
                continue
            try:
                model = load_model_file(entry.yaml_path)
            except Exception:
                continue
            new_indices[entry.name] = gpu_module.gpus_for_model(model.env)
        self._gpu_indices = new_indices

    def _entry_for(self, name: str) -> service.CatalogEntry | None:
        for entry in self._entries:
            if entry.name == name:
                return entry
        return None

    def _sync_log_attachment(self) -> None:
        name = self._models.selected_model_name
        if name == self._last_attached_log:
            return
        self._last_attached_log = name
        if name is None:
            self._logs.attach(None, "(no model selected)")
            self._metrics.show_no_selection()
            return
        paths = service.runtime_paths_for(self._options.project, name)
        self._logs.attach(paths.log_path, name)

    def _scrape_all_async(self) -> None:
        self.run_worker(self._scrape_all(), exclusive=False)

    async def _scrape_all(self) -> None:
        entries = await asyncio.to_thread(service.list_catalog_entries, self._options.project)
        for entry in entries:
            if entry.is_broken or entry.status is None:
                continue
            status = entry.status
            if status.metrics_port is None:
                continue
            if not status.running:
                self._histories.pop(status.name, None)
                continue
            url = f"http://{self._options.health_host}:{status.metrics_port}/metrics"
            history = self._history_for(status.name)
            try:
                samples = await asyncio.to_thread(metrics.scrape_vllm_metrics, url)
                history.ingest(samples)
            except metrics.VllmUnreachableError as exc:
                history.record_error(str(exc))

        self._gpu_snapshots = await asyncio.to_thread(gpu_module.query_gpus)

        self._render_metrics_panel()

    def _show_metrics_pane(self) -> None:
        self._errors.display = False
        self._metrics.display = True
        self._gpu_panel.display = True

    def _show_errors_pane(self) -> None:
        self._metrics.display = False
        self._gpu_panel.display = False
        self._errors.display = True

    def _render_metrics_panel(self) -> None:
        name = self._models.selected_model_name
        if name is None:
            self._show_metrics_pane()
            self._metrics.show_no_selection()
            self._gpu_panel.render_gpus(None, [])
            return

        entry = self._entry_for(name)
        if entry is not None and entry.is_broken:
            self._show_errors_pane()
            self._errors.show_yaml_error(
                name,
                entry.error or "invalid YAML",
                entry.yaml_path,
            )
            return

        self._show_metrics_pane()
        self._render_gpu_for(name)

        status = entry.status if entry is not None else None
        if status is not None and not status.running:
            self._metrics.show_stopped()
            return

        history = self._histories.get(name)
        if history is None or history.last_scrape_at is None:
            self._metrics.show_pending()
            return
        snapshot = snapshot_from_history(history)
        self._metrics.render_metrics(snapshot)

    def _render_gpu_for(self, name: str) -> None:
        configured = self._gpu_indices.get(name, [])
        if not configured:
            self._gpu_panel.render_gpus(configured, [])
            return
        wanted = set(configured)
        filtered = [g for g in self._gpu_snapshots if g.index in wanted]
        self._gpu_panel.render_gpus(configured, filtered)

    # --- table selection events ---

    @on(ModelsTree.NodeHighlighted)
    def _on_node_highlighted(self, event: ModelsTree.NodeHighlighted) -> None:
        del event
        self._sync_log_attachment()
        self._render_metrics_panel()
        self.refresh_bindings()

    # --- key actions ---

    def action_start_model(self) -> None:
        self._launch_user_action("start", self._do_start, self._do_start_profile)

    def action_stop_model(self) -> None:
        self._launch_user_action("stop", self._do_stop, self._do_stop_profile)

    def action_restart_model(self) -> None:
        self._launch_user_action("restart", self._do_restart, self._do_restart_profile)

    def action_refresh_now(self) -> None:
        self._refresh_statuses()
        self.run_worker(self._scrape_all(), exclusive=False)

    def action_edit_model(self) -> None:
        """Open the selected model's YAML in $EDITOR. Validates the catalog after exit."""
        name = self._models.selected_model_name
        if name is None:
            self.notify("no model selected", severity="warning", timeout=2)
            return
        project = self._options.project
        entry = self._entry_for(name)
        if entry is not None:
            config_path = entry.yaml_path
        else:
            config_path = service.resolve_models_dir(project, None) / f"{name}.yaml"
        if not config_path.is_file():
            self.notify(f"no config file: {config_path}", severity="warning", timeout=3)
            return

        editor = _resolve_editor(project.config.defaults.editor)
        if editor is None:
            self.notify(
                "no editor available, set defaults.editor in .vllmctl/config.yaml,"
                " export $EDITOR, or install nano/vim/micro",
                severity="error",
                timeout=6,
            )
            return

        editor_cmd = shlex.split(editor) + [str(config_path)]
        editor_error: str | None = None
        with self.suspend():
            try:
                subprocess.run(editor_cmd, check=False)
            except OSError as exc:
                editor_error = str(exc)

        if editor_error is not None:
            self.notify(f"editor failed: {editor_error}", severity="error", timeout=5)
            return

        try:
            service.load_catalog_for(project)
            self.notify(f"saved {config_path.name}", timeout=2)
        except Exception as exc:
            self.notify(
                f"YAML invalid after edit:\n{exc}",
                severity="error",
                timeout=10,
                markup=False,
            )

        self._refresh_statuses()
        self.run_worker(self._scrape_all(), exclusive=False)

    def action_copy_logs(self) -> None:
        """Copy the selected model's log file to the system clipboard via OSC 52."""
        name = self._models.selected_model_name
        if name is None:
            self.notify("no model selected", severity="warning", timeout=2)
            return
        paths = service.runtime_paths_for(self._options.project, name)
        if not paths.log_path.is_file():
            self.notify(f"no log yet: {paths.log_path}", severity="warning", timeout=3)
            return
        try:
            text = paths.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self.notify(f"failed to read log: {exc}", severity="error", timeout=4, markup=False)
            return

        truncated = False
        if len(text.encode("utf-8")) > self.LOG_CLIPBOARD_LIMIT_BYTES:
            text = text[-self.LOG_CLIPBOARD_LIMIT_BYTES:]
            text = "[truncated to the last ~1 MB]\n" + text
            truncated = True

        self.copy_to_clipboard(text)
        line_count = text.count("\n")
        suffix = " (truncated)" if truncated else ""
        self.notify(f"copied {line_count} lines to clipboard{suffix}", timeout=2)

    def action_smoke_test(self) -> None:
        if self._busy:
            self.notify("another action is running", severity="warning", timeout=2)
            return
        name = self._models.selected_model_name
        if name is None:
            self.notify("no model selected", severity="warning", timeout=2)
            return
        self.notify(f"smoke testing {name}...", timeout=3)
        self._busy = True
        self.run_worker(self._do_smoke_test(name), exclusive=True)

    async def _do_smoke_test(self, name: str) -> None:
        try:
            result = await asyncio.to_thread(
                service.smoke_test_model,
                self._options.project,
                name,
                host=self._options.health_host,
            )
        except service.SmokeTestError as exc:
            self.notify(
                f"smoke test failed: {exc}",
                severity="error",
                timeout=10,
                markup=False,
            )
        except Exception as exc:
            self.notify(
                f"smoke test error: {exc}",
                severity="error",
                timeout=10,
                markup=False,
            )
        else:
            self.notify(
                f"{result.model} · {result.latency_seconds:.2f}s",
                severity="information",
                timeout=4,
            )
        finally:
            self._busy = False

    def action_cycle_theme(self) -> None:
        favorites = [
            "tokyo-night",
            "monokai",
            "dracula",
            "catppuccin-mocha",
            "nord",
            "gruvbox",
            "textual-light",
            "solarized-light",
        ]
        current = self.theme
        try:
            index = favorites.index(current)
        except ValueError:
            index = -1
        self.theme = favorites[(index + 1) % len(favorites)]
        self.notify(f"theme: {self.theme}", timeout=1.5)

    def _launch_user_action(
        self,
        label: str,
        work_model: Callable[[str, str], Awaitable[None]],
        work_profile: Callable[[str, str], Awaitable[None]],
    ) -> None:
        if self._busy:
            self.notify("another action is already running", severity="warning", timeout=2)
            return
        profile = self._models.selected_profile_name
        if profile is not None:
            self._busy = True
            self.run_worker(work_profile(profile, label), exclusive=True)
            return
        name = self._models.selected_model_name
        if name is None:
            self.notify("no model selected", severity="warning", timeout=2)
            return
        entry = self._entry_for(name)
        # Stop is PID-based and doesn't read the YAML, so it stays usable
        # even when the file is broken (the process may still be running).
        if entry is not None and entry.is_broken and label != "stop":
            self.notify(
                f"can't {label}: YAML is invalid, press 'e' to fix",
                severity="warning",
                timeout=4,
            )
            return
        self._busy = True
        self.run_worker(work_model(name, label), exclusive=True)

    async def _do_start(self, name: str, label: str) -> None:
        try:
            await asyncio.to_thread(service.start_model, self._options.project, name)
            self.notify(f"{label}: {name} spawned", severity="information", timeout=2)
        except ModelAlreadyRunningError:
            self.notify(f"{name} is already running", severity="warning", timeout=3)
        except UnknownModelError:
            self.notify(f"unknown model: {name}", severity="error", timeout=3)
        except Exception as exc:
            self.notify(f"start failed: {exc}", severity="error", timeout=5, markup=False)
        finally:
            self._busy = False
            self._refresh_statuses()

    async def _do_stop(self, name: str, label: str) -> None:
        try:
            await asyncio.to_thread(service.stop_model, self._options.project, name, 30.0)
            self.notify(f"{label}: {name} stopped", severity="information", timeout=2)
        except ModelNotRunningError:
            self.notify(f"{name} is not running", severity="warning", timeout=3)
        except Exception as exc:
            self.notify(f"stop failed: {exc}", severity="error", timeout=5, markup=False)
        finally:
            self._busy = False
            self._histories.pop(name, None)
            self._refresh_statuses()

    async def _do_restart(self, name: str, label: str) -> None:
        try:
            await asyncio.to_thread(service.restart_model, self._options.project, name)
            self.notify(f"{label}: {name} respawned", severity="information", timeout=2)
        except UnknownModelError:
            self.notify(f"unknown model: {name}", severity="error", timeout=3)
        except Exception as exc:
            self.notify(f"restart failed: {exc}", severity="error", timeout=5, markup=False)
        finally:
            self._busy = False
            self._histories.pop(name, None)
            self._refresh_statuses()

    async def _do_start_profile(self, profile: str, label: str) -> None:
        del label
        try:
            result = await asyncio.to_thread(
                service.start_profile, self._options.project, profile
            )
            self._notify_bulk(result)
        except service.UnknownProfileError:
            self.notify(f"unknown profile: {profile}", severity="error", timeout=3)
        except Exception as exc:
            self.notify(
                f"profile start failed: {exc}", severity="error", timeout=5, markup=False
            )
        finally:
            self._busy = False
            self._refresh_statuses()

    async def _do_stop_profile(self, profile: str, label: str) -> None:
        del label
        try:
            result = await asyncio.to_thread(
                service.stop_profile, self._options.project, profile
            )
            self._notify_bulk(result)
            for name in result.succeeded:
                self._histories.pop(name, None)
        except service.UnknownProfileError:
            self.notify(f"unknown profile: {profile}", severity="error", timeout=3)
        except Exception as exc:
            self.notify(
                f"profile stop failed: {exc}", severity="error", timeout=5, markup=False
            )
        finally:
            self._busy = False
            self._refresh_statuses()

    async def _do_restart_profile(self, profile: str, label: str) -> None:
        del label
        try:
            result = await asyncio.to_thread(
                service.restart_profile, self._options.project, profile
            )
            self._notify_bulk(result)
            for name in result.succeeded:
                self._histories.pop(name, None)
        except service.UnknownProfileError:
            self.notify(f"unknown profile: {profile}", severity="error", timeout=3)
        except Exception as exc:
            self.notify(
                f"profile restart failed: {exc}", severity="error", timeout=5, markup=False
            )
        finally:
            self._busy = False
            self._refresh_statuses()

    def _notify_bulk(self, result: service.BulkResult) -> None:
        parts: list[str] = []
        if result.succeeded:
            parts.append(f"{len(result.succeeded)} {result.action}ed")
        if result.skipped:
            parts.append(f"{len(result.skipped)} skipped")
        if result.failed:
            parts.append(f"{len(result.failed)} failed")
        if not parts:
            parts.append("nothing to do")
        message = f"{result.profile}: " + " · ".join(parts)
        severity: Literal["information", "warning", "error"]
        if result.failed:
            severity = "error"
            details = ", ".join(name for name, _ in result.failed)
            message += f"\nfailed: {details}"
        elif result.skipped and not result.succeeded:
            severity = "warning"
        else:
            severity = "information"
        self.notify(message, severity=severity, timeout=4, markup=False)


def _resolve_editor(configured: str | None = None) -> str | None:
    """Pick an editor by precedence: project config > $VISUAL > $EDITOR > common fallbacks.

    The returned string can include arguments (e.g. ``"code --wait"``); only the
    first token has to be on PATH. Returns None if no editor is reachable.
    """
    sources: list[str] = []
    if configured:
        sources.append(configured)
    for var in ("VISUAL", "EDITOR"):
        value = os.environ.get(var)
        if value:
            sources.append(value)
    sources.extend(["nano", "vim", "vi", "micro", "emacs"])

    for raw in sources:
        if not raw.strip():
            continue
        first = raw.split(maxsplit=1)[0]
        if shutil.which(first) is not None:
            return raw
    return None
