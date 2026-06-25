from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SECRET_ENV_KEYS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}


class VllmConfig(BaseModel):
    """Pass-through description of a vLLM command."""

    model_config = ConfigDict(extra="forbid")

    executable: str = "vllm"
    subcommand: str = "serve"
    model: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("args")
    @classmethod
    def validate_arg_names(cls, value: dict[str, Any]) -> dict[str, Any]:
        for key in value:
            if not key.startswith("--"):
                raise ValueError(f"vllm arg must start with '--': {key}")
        return value

    @field_validator("flags")
    @classmethod
    def validate_flags(cls, value: list[str]) -> list[str]:
        for flag in value:
            if not flag.startswith("--"):
                raise ValueError(f"vllm flag must start with '--': {flag}")
        return value

    def command_args(self) -> list[str]:
        command = [self.executable, self.subcommand, self.model]

        for key, value in self.args.items():
            if isinstance(value, bool):
                if value:
                    command.append(key)
                continue

            if isinstance(value, list):
                for item in value:
                    command.extend([key, str(item)])
                continue

            command.extend([key, str(value)])

        command.extend(self.flags)
        command.extend(self.extra_args)
        return command


class MetricsConfig(BaseModel):
    """Where to scrape this model's Prometheus-format /metrics endpoint."""

    model_config = ConfigDict(extra="forbid")

    port: int | None = Field(default=None, ge=1, le=65535)
    path: str = "/metrics"


class ModelConfig(BaseModel):
    """Declarative configuration for one bare-metal vLLM server."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    env: dict[str, str] = Field(default_factory=dict)
    vllm: VllmConfig
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value:
            if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
                raise ValueError(f"invalid environment variable name: {key}")
            if key in SECRET_ENV_KEYS:
                raise ValueError(f"{key} must come from the shell or a local .env, not model YAML")
        return {key: str(env_value) for key, env_value in value.items()}

    @property
    def metrics_port(self) -> int | None:
        if self.metrics.port is not None:
            return self.metrics.port

        raw_port = self.vllm.args.get("--port")
        if raw_port is None or isinstance(raw_port, (bool, list)):
            return None

        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            return None

        if 1 <= port <= 65535:
            return port
        return None

    @property
    def metrics_path(self) -> str:
        return self.metrics.path

    def command_args(self) -> list[str]:
        return self.vllm.command_args()


class Catalog(BaseModel):
    models: list[ModelConfig]

    @model_validator(mode="after")
    def validate_unique_fields(self) -> "Catalog":
        names: set[str] = set()
        for model in self.models:
            if model.name in names:
                raise ValueError(f"duplicate model name: {model.name}")
            names.add(model.name)
        return self

    def get(self, model_name: str) -> ModelConfig | None:
        for model in self.models:
            if model.name == model_name:
                return model
        return None

    def next_available_port(self, start: int = 8001) -> int:
        used_ports = {
            model.metrics_port for model in self.models if model.metrics_port is not None
        }
        port = start
        while port in used_ports:
            port += 1
        return port


def tensor_parallel_size_from_gpus(gpus: str) -> int:
    return len([gpu for gpu in gpus.split(",") if gpu.strip()])


def create_default_model_config(
    *,
    name: str,
    hf_model: str,
    gpus: str,
    port: int,
    host: str = "0.0.0.0",
    hf_home: str = "data/huggingface",
    vllm_executable: str = "vllm",
    vllm_subcommand: str = "serve",
    log_level: str = "INFO",
) -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "name": name,
            "env": {
                "CUDA_VISIBLE_DEVICES": gpus,
                "HF_HOME": hf_home,
                "VLLM_LOGGING_LEVEL": log_level,
            },
            "vllm": {
                "executable": vllm_executable,
                "subcommand": vllm_subcommand,
                "model": hf_model,
                "args": {
                    "--host": host,
                    "--port": port,
                    "--tensor-parallel-size": tensor_parallel_size_from_gpus(gpus),
                    "--dtype": "auto",
                    "--served-model-name": name,
                    # Silence access-log spam from vllmctl's own polling
                    # without losing the access lines for real /v1/* requests.
                    # Edit or remove this entry if you want to see all access logs.
                    "--disable-access-log-for-endpoints": "/health,/metrics,/ping",
                },
                "flags": [],
                "extra_args": [],
            },
            "metrics": {
                "path": "/metrics",
            },
        }
    )


def dump_model_file(path: Path, model: ModelConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.model_dump(mode="python", exclude_none=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def load_model_file(path: Path) -> ModelConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return ModelConfig.model_validate(raw)


def load_catalog(config_dir: Path) -> Catalog:
    if not config_dir.exists():
        raise FileNotFoundError(f"config directory not found: {config_dir}")
    if not config_dir.is_dir():
        raise NotADirectoryError(f"config path is not a directory: {config_dir}")

    files = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    if not files:
        raise FileNotFoundError(f"no model YAML files found in {config_dir}")

    models = [load_model_file(path) for path in files]
    return Catalog(models=models)


def load_catalog_or_empty(config_dir: Path) -> Catalog:
    if not config_dir.exists():
        return Catalog(models=[])

    files = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    if not files:
        return Catalog(models=[])

    return load_catalog(config_dir)
