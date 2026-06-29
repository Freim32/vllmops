import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_DIR = ".vllmops"
PROJECT_CONFIG = "config.yaml"
DEFAULT_PYTHON_VERSION = "3.10"
NAME_PATTERN_STR = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
NAME_PATTERN = re.compile(NAME_PATTERN_STR)
GENERAL_PROFILE = "general"

ENV_EXAMPLE = """# HuggingFace authentication. Required for gated models (Llama, Gemma, ...).
HF_TOKEN=
HUGGING_FACE_HUB_TOKEN=
"""

GITIGNORE_TEMPLATE = """# vllmops runtime artifacts (PIDs, logs, internal state)
runtime/

# HuggingFace cache and locally-downloaded model weights
data/

# Project-local virtual environment with vLLM installed
.venv/

# Local secrets, never commit
.env
.env.local
"""

PYPROJECT_TEMPLATE = """[project]
name = "{name}"
version = "0.1.0"
description = "vllmops-managed vLLM workspace"
requires-python = ">={python_version}"
dependencies = [
    "vllm>=0.6",
]
"""

PYTHON_VERSION_TEMPLATE = "{python_version}\n"


def sanitize_project_name(folder_name: str) -> str:
    """Turn an arbitrary folder name into a valid project name.

    Replaces runs of invalid characters with a dash, strips leading and
    trailing punctuation, falls back to a sentinel when nothing usable remains.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", folder_name)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_.")
    if not cleaned or not cleaned[0].isalnum():
        return "vllmops-project"
    return cleaned


class ProjectPaths(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models_dir: str = "configs/models"
    logs_dir: str = "runtime/logs"
    pids_dir: str = "runtime/pids"
    hf_home: str = "data/huggingface"


class ProjectDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port_start: int = Field(default=8001, ge=1, le=65535)
    vllm_executable: str = "vllm"
    vllm_subcommand: str = "serve"
    log_level: str = "INFO"
    editor: str | None = Field(
        default=None,
        description="Editor command for `vllmops tui` 'e' binding. Overrides $VISUAL / $EDITOR.",
    )


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str | None = Field(
        default=None,
        pattern=NAME_PATTERN_STR,
        description="Display name for the project. Falls back to folder name if absent.",
    )
    paths: ProjectPaths = Field(default_factory=ProjectPaths)
    defaults: ProjectDefaults = Field(default_factory=ProjectDefaults)
    profiles: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Named groupings of model names. A model may appear in any number"
            " of profiles; unassigned models fall into the synthetic 'general'"
            " catch-all rendered by the TUI."
        ),
    )

    @model_validator(mode="after")
    def _validate_profiles(self) -> "ProjectConfig":
        if GENERAL_PROFILE in self.profiles:
            raise ValueError(f"{GENERAL_PROFILE!r} is a reserved profile name (synthetic catch-all)")
        for profile_name in self.profiles:
            if not NAME_PATTERN.match(profile_name):
                raise ValueError(f"invalid profile name: {profile_name!r} (must match {NAME_PATTERN.pattern})")
        return self


@dataclass(frozen=True)
class Project:
    root: Path
    config: ProjectConfig

    def resolve(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.root / path

    @property
    def name(self) -> str:
        """Project display name. Uses configured value if set, else folder name."""
        return self.config.name or self.root.name

    @property
    def config_path(self) -> Path:
        return self.root / PROJECT_DIR / PROJECT_CONFIG

    @property
    def models_dir(self) -> Path:
        return self.resolve(self.config.paths.models_dir)

    @property
    def venv_executable(self) -> Path | None:
        """Path to `.venv/bin/vllm` (POSIX) or `.venv/Scripts/vllm.exe` (Windows), if present."""
        candidates = [
            self.root / ".venv" / "bin" / "vllm",
            self.root / ".venv" / "Scripts" / "vllm.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None


def find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / PROJECT_DIR / PROJECT_CONFIG).is_file():
            return candidate
    return None


def load_project(start: Path | None = None) -> Project:
    start_path = Path.cwd() if start is None else start
    root = find_project_root(start_path) or start_path.resolve()
    config_path = root / PROJECT_DIR / PROJECT_CONFIG

    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = ProjectConfig.model_validate(raw)
    else:
        config = ProjectConfig()

    return Project(root=root, config=config)


def init_project(
    target: Path,
    force: bool = False,
    name: str | None = None,
    python_version: str = DEFAULT_PYTHON_VERSION,
) -> list[Path]:
    root = target.resolve()
    resolved_name = name if name is not None else sanitize_project_name(root.name)
    if not NAME_PATTERN.match(resolved_name):
        raise ValueError(
            f"invalid project name: {resolved_name!r} (must match {NAME_PATTERN.pattern}; pass --name to override)"
        )

    config = ProjectConfig(name=resolved_name)
    project = Project(root=root, config=config)
    config_path = project.config_path

    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} already exists")

    directories = [
        root / PROJECT_DIR,
        project.models_dir,
        project.resolve(config.paths.logs_dir),
        project.resolve(config.paths.pids_dir),
        project.resolve(config.paths.hf_home),
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    written.append(config_path)

    env_example = root / ".env.example"
    if not env_example.exists() or force:
        env_example.write_text(ENV_EXAMPLE, encoding="utf-8")
        written.append(env_example)

    gitignore = root / ".gitignore"
    if not gitignore.exists() or force:
        gitignore.write_text(GITIGNORE_TEMPLATE, encoding="utf-8")
        written.append(gitignore)

    pyproject = root / "pyproject.toml"
    if not pyproject.exists() or force:
        pyproject.write_text(
            PYPROJECT_TEMPLATE.format(name=resolved_name, python_version=python_version),
            encoding="utf-8",
        )
        written.append(pyproject)

    py_version_file = root / ".python-version"
    if not py_version_file.exists() or force:
        py_version_file.write_text(
            PYTHON_VERSION_TEMPLATE.format(python_version=python_version),
            encoding="utf-8",
        )
        written.append(py_version_file)

    return written
