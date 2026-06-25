"""Tests for vllmctl.project (workspace init, config loading)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllmctl.project import (
    PROJECT_CONFIG,
    PROJECT_DIR,
    ProjectConfig,
    find_project_root,
    init_project,
    load_project,
    sanitize_project_name,
)


def test_init_creates_expected_files(tmp_path: Path) -> None:
    written = init_project(tmp_path)
    paths = {p.name for p in written}
    assert PROJECT_CONFIG in paths
    assert ".env.example" in paths
    assert ".gitignore" in paths
    assert "pyproject.toml" in paths
    assert ".python-version" in paths


def test_init_pyproject_contains_vllm_dep(tmp_path: Path) -> None:
    init_project(tmp_path, name="my-proj")
    contents = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "my-proj"' in contents
    assert "vllm>=0.6" in contents
    assert 'requires-python = ">=3.10"' in contents


def test_init_python_version_file(tmp_path: Path) -> None:
    init_project(tmp_path)
    assert (tmp_path / ".python-version").read_text(encoding="utf-8").strip() == "3.10"


def test_init_default_name_is_folder_name(tmp_path: Path) -> None:
    target = tmp_path / "my-llms"
    target.mkdir()
    init_project(target)
    project = load_project(target)
    assert project.name == "my-llms"
    assert project.config.name == "my-llms"


def test_init_explicit_name_overrides_folder(tmp_path: Path) -> None:
    target = tmp_path / "weird name with spaces"
    target.mkdir()
    init_project(target, name="custom-name")
    project = load_project(target)
    assert project.name == "custom-name"


def test_init_sanitizes_invalid_folder_name(tmp_path: Path) -> None:
    target = tmp_path / "weird name with spaces"
    target.mkdir()
    init_project(target)
    project = load_project(target)
    assert project.config.name == "weird-name-with-spaces"


def test_init_rejects_explicit_invalid_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid project name"):
        init_project(tmp_path, name="has spaces")


def test_project_name_falls_back_to_folder_when_config_lacks_it(tmp_path: Path) -> None:
    init_project(tmp_path)
    cfg = tmp_path / PROJECT_DIR / PROJECT_CONFIG
    contents = cfg.read_text(encoding="utf-8").splitlines()
    cfg.write_text(
        "\n".join(line for line in contents if not line.startswith("name:")),
        encoding="utf-8",
    )
    project = load_project(tmp_path)
    assert project.config.name is None
    assert project.name == tmp_path.resolve().name


def test_sanitize_project_name_handles_edge_cases() -> None:
    assert sanitize_project_name("hello") == "hello"
    assert sanitize_project_name("hello world") == "hello-world"
    assert sanitize_project_name("---weird---") == "weird"
    assert sanitize_project_name("") == "vllmctl-project"
    assert sanitize_project_name("123abc") == "123abc"
    assert sanitize_project_name("...") == "vllmctl-project"


def test_venv_executable_returns_none_when_missing(tmp_path: Path) -> None:
    init_project(tmp_path)
    project = load_project(tmp_path)
    assert project.venv_executable is None


def test_venv_executable_finds_posix_path(tmp_path: Path) -> None:
    init_project(tmp_path)
    fake = tmp_path / ".venv" / "bin" / "vllm"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\nexit 0\n")
    project = load_project(tmp_path)
    assert project.venv_executable == fake


def test_project_defaults_editor_is_optional(tmp_path: Path) -> None:
    """Newly-init'd projects don't pin an editor, env var or fallback is used."""
    init_project(tmp_path)
    project = load_project(tmp_path)
    assert project.config.defaults.editor is None


def test_project_defaults_editor_can_be_set(tmp_path: Path) -> None:
    """Manually setting `defaults.editor` in config.yaml round-trips."""
    import yaml as _yaml  # noqa: PLC0415, local to keep production imports clean

    init_project(tmp_path)
    cfg_path = tmp_path / PROJECT_DIR / PROJECT_CONFIG
    raw = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["defaults"]["editor"] = "nano"
    cfg_path.write_text(_yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    project = load_project(tmp_path)
    assert project.config.defaults.editor == "nano"


def test_init_gitignore_excludes_runtime_data_env(tmp_path: Path) -> None:
    init_project(tmp_path)
    contents = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "runtime/" in contents
    assert "data/" in contents
    assert ".env" in contents


def test_init_does_not_create_templates_or_observability_dirs(tmp_path: Path) -> None:
    """V1 has no docker/observability templates, direct vLLM scrape only."""
    init_project(tmp_path)
    assert not (tmp_path / "templates").exists()
    assert not (tmp_path / "generated").exists()


def test_init_creates_runtime_dirs(tmp_path: Path) -> None:
    init_project(tmp_path)
    for sub in ("runtime/logs", "runtime/pids", "configs/models"):
        assert (tmp_path / sub).is_dir(), sub


def test_init_refuses_existing_without_force(tmp_path: Path) -> None:
    init_project(tmp_path)
    with pytest.raises(FileExistsError):
        init_project(tmp_path, force=False)


def test_init_force_rewrites_config(tmp_path: Path) -> None:
    init_project(tmp_path)
    cfg = tmp_path / PROJECT_DIR / PROJECT_CONFIG
    cfg.write_text("# tampered\n", encoding="utf-8")
    init_project(tmp_path, force=True)
    assert "version" in cfg.read_text(encoding="utf-8")


def test_load_project_with_no_config_returns_defaults(tmp_path: Path) -> None:
    project = load_project(tmp_path)
    assert project.config == ProjectConfig()


def test_load_project_after_init_resolves_paths(tmp_path: Path) -> None:
    init_project(tmp_path)
    project = load_project(tmp_path)
    assert project.models_dir == tmp_path / "configs" / "models"


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    init_project(tmp_path)
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert find_project_root(deep) == tmp_path.resolve()


def test_find_project_root_returns_none_outside_project(tmp_path: Path) -> None:
    assert find_project_root(tmp_path) is None


def test_resolve_returns_absolute_paths(tmp_path: Path) -> None:
    init_project(tmp_path)
    project = load_project(tmp_path)
    assert project.resolve("configs/models").is_absolute()
    assert project.resolve("/already/absolute").is_absolute()


def _rewrite_config_profiles(tmp_path: Path, profiles: dict[str, list[str]]) -> None:
    """Set the `profiles` block in the project config.yaml. Helper for profile tests."""
    import yaml as _yaml  # noqa: PLC0415

    cfg_path = tmp_path / PROJECT_DIR / PROJECT_CONFIG
    raw = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["profiles"] = profiles
    cfg_path.write_text(_yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_profiles_default_is_empty(tmp_path: Path) -> None:
    init_project(tmp_path)
    project = load_project(tmp_path)
    assert project.config.profiles == {}


def test_profiles_round_trip_from_yaml(tmp_path: Path) -> None:
    init_project(tmp_path)
    _rewrite_config_profiles(tmp_path, {"dev": ["a", "b"], "prod": ["c"]})
    project = load_project(tmp_path)
    assert project.config.profiles == {"dev": ["a", "b"], "prod": ["c"]}


def test_profiles_reject_reserved_general_name(tmp_path: Path) -> None:
    init_project(tmp_path)
    _rewrite_config_profiles(tmp_path, {"general": ["a"]})
    with pytest.raises(Exception, match="reserved"):
        load_project(tmp_path)


def test_profiles_allow_same_model_in_multiple_profiles(tmp_path: Path) -> None:
    """A model can be listed in two profiles; lifecycle is idempotent so it stays safe."""
    init_project(tmp_path)
    _rewrite_config_profiles(tmp_path, {"dev": ["a", "b"], "prod": ["b"]})
    project = load_project(tmp_path)
    assert project.config.profiles == {"dev": ["a", "b"], "prod": ["b"]}


def test_profiles_reject_invalid_profile_name(tmp_path: Path) -> None:
    init_project(tmp_path)
    _rewrite_config_profiles(tmp_path, {"has spaces": ["a"]})
    with pytest.raises(Exception, match="invalid profile name"):
        load_project(tmp_path)


def test_profiles_empty_profile_is_valid(tmp_path: Path) -> None:
    """Declared but empty profile is allowed (placeholder); rendering filters it out."""
    init_project(tmp_path)
    _rewrite_config_profiles(tmp_path, {"placeholder": []})
    project = load_project(tmp_path)
    assert project.config.profiles == {"placeholder": []}
