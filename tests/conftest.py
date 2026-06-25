"""Shared pytest fixtures."""

from __future__ import annotations

import socket
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from vllmctl.project import Project, init_project, load_project

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """An initialized vllmctl project rooted at tmp_path."""
    init_project(tmp_path, force=False)
    return load_project(tmp_path)


def write_model_yaml(project: Project, name: str, payload: dict) -> Path:
    """Write a model YAML directly under the project's models dir, bypassing service helpers."""
    project.models_dir.mkdir(parents=True, exist_ok=True)
    path = project.models_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def sleeper_payload(name: str, port: int, with_metrics: bool = True) -> dict:
    """A minimal model that runs `python -c "time.sleep(60)"` instead of vLLM."""
    payload: dict = {
        "name": name,
        "env": {},
        "vllm": {
            "executable": sys.executable,
            "subcommand": "-c",
            "model": "import time; time.sleep(60)",
            "args": {},
            "flags": [],
            "extra_args": [],
        },
    }
    if with_metrics:
        payload["metrics"] = {"port": port, "path": "/metrics"}
    return payload


def fast_exit_payload(name: str, port: int) -> dict:
    """A model whose process exits immediately. For startup-failed paths."""
    payload = sleeper_payload(name, port)
    payload["vllm"]["model"] = "import sys; sys.exit(7)"
    return payload


def free_port() -> int:
    """OS-assigned free TCP port (small race window)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _SilentHandler(BaseHTTPRequestHandler):
    response_status = 200
    response_body = b"ok"

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(self.response_status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(self.response_body)))
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def mock_health_server() -> Iterator[tuple[int, type[_SilentHandler]]]:
    """A tiny HTTP server bound to a random port that returns 200 on any GET."""
    server = HTTPServer(("127.0.0.1", 0), _SilentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], _SilentHandler
    finally:
        server.shutdown()
        server.server_close()


class MockVllmMetricsHandler(BaseHTTPRequestHandler):
    """Configurable mock vLLM /metrics endpoint.

    Tests mutate the class-level `response_body` to simulate different states.
    """

    response_status = 200
    response_body: bytes = b""

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(self.response_status)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(self.response_body)))
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def mock_vllm_metrics() -> Iterator[tuple[str, type[MockVllmMetricsHandler]]]:
    """A configurable mock /metrics server. Yields (base_url, handler_class)."""
    MockVllmMetricsHandler.response_status = 200
    MockVllmMetricsHandler.response_body = b""
    server = HTTPServer(("127.0.0.1", 0), MockVllmMetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield f"http://127.0.0.1:{port}", MockVllmMetricsHandler
    finally:
        server.shutdown()
        server.server_close()


class MockCompletionsHandler(BaseHTTPRequestHandler):
    """Mock vLLM /v1/completions endpoint for smoke-test integration tests."""

    response_status = 200
    response_body: bytes = b""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(self.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.response_body)))
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def mock_completions() -> Iterator[tuple[int, type[MockCompletionsHandler]]]:
    """A configurable mock /v1/completions server. Yields (port, handler_class)."""
    MockCompletionsHandler.response_status = 200
    MockCompletionsHandler.response_body = b""
    server = HTTPServer(("127.0.0.1", 0), MockCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], MockCompletionsHandler
    finally:
        server.shutdown()
        server.server_close()
