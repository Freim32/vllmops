"""Tests for the YAML / pydantic error explainer."""

from __future__ import annotations

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vllmops.config_errors import explain_yaml_error

# --- minimal pydantic models for isolated testing ---------------------------------


class _Inner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    args: dict[str, int] = Field(default_factory=dict)

    @field_validator("args")
    @classmethod
    def _validate_args(cls, value: dict[str, int]) -> dict[str, int]:
        for key in value:
            if not key.startswith("--"):
                raise ValueError(f"vllm arg must start with '--': {key}")
        return value


class _Outer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    inner: _Inner


def _raise(payload: dict) -> ValidationError:
    try:
        _Outer.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


# --- per-error-type cases ---------------------------------------------------------


def test_explain_missing_required_field() -> None:
    exc = _raise({"name": "ok"})
    explanation = explain_yaml_error(exc)
    assert "missing required field" in explanation.summary
    assert "inner" in explanation.summary


def test_explain_extra_forbidden_field() -> None:
    exc = _raise({"name": "ok", "inner": {"model": "m"}, "rogue": 1})
    explanation = explain_yaml_error(exc)
    assert "unknown field" in explanation.summary
    assert "rogue" in explanation.summary


def test_explain_string_pattern_mismatch() -> None:
    exc = _raise({"name": "has spaces", "inner": {"model": "m"}})
    explanation = explain_yaml_error(exc)
    assert "invalid format" in explanation.summary
    assert "name" in explanation.summary


def test_explain_int_parsing() -> None:
    exc = _raise({"name": "ok", "inner": {"model": "m", "args": {"--port": "not-an-int"}}})
    explanation = explain_yaml_error(exc)
    assert "must be an integer" in explanation.summary


def test_explain_value_error_dash_prefix() -> None:
    """vllmops-specific custom validator message is recognized."""
    exc = _raise({"name": "ok", "inner": {"model": "m", "args": {"port": 1}}})
    explanation = explain_yaml_error(exc)
    assert "must start with `--`" in explanation.summary


def test_explain_multi_error_appends_more_suffix() -> None:
    """When several errors exist, the summary shows the first + count of the rest."""
    exc = _raise({})  # name missing AND inner missing
    explanation = explain_yaml_error(exc)
    assert "+1 more" in explanation.summary


def test_explain_yaml_syntax_error_with_line_info() -> None:
    bad = "name: ok\n  bad: indent: here\n"
    try:
        yaml.safe_load(bad)
    except yaml.YAMLError as exc:
        explanation = explain_yaml_error(exc)
    else:
        raise AssertionError("expected YAMLError")
    assert "syntax error" in explanation.summary
    assert "line" in explanation.summary


def test_explain_yaml_syntax_error_without_mark() -> None:
    """Some yaml.YAMLError subclasses don't carry a problem_mark."""

    class _Bare(yaml.YAMLError):
        pass

    explanation = explain_yaml_error(_Bare("something went wrong"))
    assert explanation.summary == "something went wrong"


def test_explain_unknown_exception_falls_back() -> None:
    explanation = explain_yaml_error(RuntimeError("disk on fire"))
    assert explanation.summary == "disk on fire"


def test_explain_unknown_exception_with_empty_message() -> None:
    explanation = explain_yaml_error(RuntimeError())
    assert explanation.summary == "RuntimeError"


@pytest.mark.parametrize(
    ("reason", "expected_phrase"),
    [
        ("reserved profile name", "profile name is reserved"),
        ("at most one profile", "more than one profile"),
        ("duplicate model name", "duplicate model name across YAMLs"),
        ("invalid profile name", "profile name must match"),
    ],
)
def test_explain_value_error_known_phrases(reason: str, expected_phrase: str) -> None:
    """The value_error mapper recognizes vllmops-specific custom messages."""

    class _Model(BaseModel):
        @field_validator("__class__", mode="before", check_fields=False)
        @classmethod
        def _noop(cls, v: object) -> object:
            return v

    # Build a synthetic ValidationError by going through pydantic with a custom validator.
    class _One(BaseModel):
        x: int = 0

        @field_validator("x")
        @classmethod
        def _v(cls, value: int) -> int:
            raise ValueError(reason)

    try:
        _One.model_validate({"x": 1})
    except ValidationError as exc:
        explanation = explain_yaml_error(exc)
    else:
        raise AssertionError("expected ValidationError")
    assert expected_phrase in explanation.summary
