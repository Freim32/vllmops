"""Tests for vllmctl.config (pydantic validation, catalog logic)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vllmctl.config import (
    Catalog,
    ModelConfig,
    VllmConfig,
    create_default_model_config,
    dump_model_file,
    load_catalog,
    load_catalog_or_empty,
    load_model_file,
    tensor_parallel_size_from_gpus,
)


def _valid_model(name: str = "m1", port: int = 8001) -> dict:
    return {
        "name": name,
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "vllm": {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "args": {"--port": port, "--host": "0.0.0.0"},
        },
    }


# --- VllmConfig ---


def test_vllm_arg_must_start_with_double_dash() -> None:
    with pytest.raises(ValidationError, match="must start with '--'"):
        VllmConfig.model_validate({"model": "x", "args": {"port": 8000}})


def test_vllm_flag_must_start_with_double_dash() -> None:
    with pytest.raises(ValidationError, match="must start with '--'"):
        VllmConfig.model_validate({"model": "x", "flags": ["trust-remote-code"]})


def test_vllm_command_args_renders_dict_args() -> None:
    cfg = VllmConfig.model_validate(
        {
            "model": "m",
            "args": {"--port": 8001, "--dtype": "auto"},
            "flags": ["--trust-remote-code"],
            "extra_args": ["--lora-modules", "a=/p"],
        }
    )
    cmd = cfg.command_args()
    assert cmd[:3] == ["vllm", "serve", "m"]
    assert "--port" in cmd and "8001" in cmd
    assert "--trust-remote-code" in cmd
    assert cmd[-2:] == ["--lora-modules", "a=/p"]


def test_vllm_command_args_bool_flag_only_if_true() -> None:
    cfg = VllmConfig.model_validate(
        {"model": "m", "args": {"--enable": True, "--disable": False}}
    )
    cmd = cfg.command_args()
    assert "--enable" in cmd
    assert "--disable" not in cmd


def test_vllm_command_args_repeats_list_values() -> None:
    cfg = VllmConfig.model_validate(
        {"model": "m", "args": {"--lora-modules": ["a=/p", "b=/q"]}}
    )
    cmd = cfg.command_args()
    assert cmd.count("--lora-modules") == 2


# --- ModelConfig ---


def test_model_name_pattern_rejects_invalid() -> None:
    payload = _valid_model(name="bad name!")
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(payload)


def test_model_rejects_secret_env_keys() -> None:
    payload = _valid_model()
    payload["env"]["HF_TOKEN"] = "leak"
    with pytest.raises(ValidationError, match="HF_TOKEN"):
        ModelConfig.model_validate(payload)


def test_model_rejects_invalid_env_name() -> None:
    payload = _valid_model()
    payload["env"]["1BAD"] = "x"
    with pytest.raises(ValidationError, match="invalid environment variable"):
        ModelConfig.model_validate(payload)


def test_model_metrics_port_from_metrics_overrides_args() -> None:
    payload = _valid_model(port=8001)
    payload["metrics"] = {"port": 9999}
    model = ModelConfig.model_validate(payload)
    assert model.metrics_port == 9999


def test_model_metrics_port_falls_back_to_args_port() -> None:
    payload = _valid_model(port=8001)
    model = ModelConfig.model_validate(payload)
    assert model.metrics_port == 8001


def test_model_without_any_port_has_metrics_port_none() -> None:
    payload = _valid_model()
    payload["vllm"]["args"] = {}
    model = ModelConfig.model_validate(payload)
    assert model.metrics_port is None


def test_model_metrics_path_default() -> None:
    payload = _valid_model()
    model = ModelConfig.model_validate(payload)
    assert model.metrics_path == "/metrics"


def test_model_metrics_path_custom() -> None:
    payload = _valid_model()
    payload["metrics"] = {"path": "/custom"}
    model = ModelConfig.model_validate(payload)
    assert model.metrics_path == "/custom"


# --- Catalog ---


def test_catalog_rejects_duplicate_names() -> None:
    a = ModelConfig.model_validate(_valid_model("dup", port=8001))
    b = ModelConfig.model_validate(_valid_model("dup", port=8002))
    with pytest.raises(ValidationError, match="duplicate model name"):
        Catalog(models=[a, b])


def test_catalog_allows_duplicate_metrics_ports() -> None:
    """Two models can share a metrics_port at config time. Conflict is enforced
    at runtime when both try to bind (see `start_model` + `PortConflictError`)."""
    a = ModelConfig.model_validate(_valid_model("a", port=8001))
    b = ModelConfig.model_validate(_valid_model("b", port=8001))
    catalog = Catalog(models=[a, b])
    assert len(catalog.models) == 2


def test_catalog_get_returns_model_or_none() -> None:
    a = ModelConfig.model_validate(_valid_model("a", port=8001))
    catalog = Catalog(models=[a])
    assert catalog.get("a") is a
    assert catalog.get("missing") is None


def test_catalog_next_available_port_skips_used() -> None:
    a = ModelConfig.model_validate(_valid_model("a", port=8001))
    b = ModelConfig.model_validate(_valid_model("b", port=8002))
    catalog = Catalog(models=[a, b])
    assert catalog.next_available_port(start=8001) == 8003


def test_catalog_next_available_port_returns_start_when_free() -> None:
    catalog = Catalog(models=[])
    assert catalog.next_available_port(start=9000) == 9000


# --- helpers ---


def test_tensor_parallel_size_from_gpus() -> None:
    assert tensor_parallel_size_from_gpus("0") == 1
    assert tensor_parallel_size_from_gpus("0,1") == 2
    assert tensor_parallel_size_from_gpus("0, 1, 2, 3") == 4
    assert tensor_parallel_size_from_gpus("") == 0


def test_create_default_model_config_sets_tensor_parallel_size_to_gpu_count() -> None:
    model = create_default_model_config(
        name="m", hf_model="hf/m", gpus="0,1,2", port=8001
    )
    assert model.vllm.args["--tensor-parallel-size"] == 3


def test_create_default_model_config_silences_polling_endpoints() -> None:
    """Default args drop access log spam from vllmctl's own /health and /metrics polling."""
    model = create_default_model_config(name="m", hf_model="hf/m", gpus="0", port=8001)
    silenced = model.vllm.args.get("--disable-access-log-for-endpoints")
    assert silenced is not None
    assert "/health" in silenced
    assert "/metrics" in silenced
    assert "/ping" in silenced


# --- file IO ---


def test_dump_and_load_model_roundtrip(tmp_path: Path) -> None:
    model = create_default_model_config(
        name="m1", hf_model="hf/m", gpus="0", port=8001
    )
    path = tmp_path / "m1.yaml"
    dump_model_file(path, model)
    loaded = load_model_file(path)
    assert loaded.name == "m1"
    assert loaded.metrics_port == 8001


def test_load_catalog_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "nope")


def test_load_catalog_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    with pytest.raises(FileNotFoundError, match="no model YAML"):
        load_catalog(tmp_path / "models")


def test_load_catalog_or_empty_returns_empty_when_missing(tmp_path: Path) -> None:
    catalog = load_catalog_or_empty(tmp_path / "missing")
    assert catalog.models == []


def test_load_catalog_picks_up_yaml_files(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    a = create_default_model_config(name="a", hf_model="hf/a", gpus="0", port=8001)
    b = create_default_model_config(name="b", hf_model="hf/b", gpus="0", port=8002)
    dump_model_file(models_dir / "a.yaml", a)
    dump_model_file(models_dir / "b.yaml", b)
    catalog = load_catalog(models_dir)
    assert {m.name for m in catalog.models} == {"a", "b"}
