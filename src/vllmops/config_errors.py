"""Human-friendly summaries for YAML / pydantic validation failures.

Pydantic v2's `ValidationError.errors()` exposes a list of dicts with `type`,
`loc`, `msg`, `input` keys. We map the most common `type` values to a short
summary the user can act on, and reproduce vllmops-specific custom-validator
messages (`vllm arg must start with '--'` etc.) when they appear.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import yaml
from pydantic import ValidationError


class YamlErrorExplanation(NamedTuple):
    """One-line summary of a YAML/validation error, no actionable hint attached.

    The TUI keeps its own static "press 'e' to edit" hint; the summary just
    replaces the cryptic raw pydantic message.
    """

    summary: str


def explain_yaml_error(exc: BaseException) -> YamlErrorExplanation:
    """Map a pydantic / PyYAML / generic error to a single human-friendly line."""
    if isinstance(exc, ValidationError):
        return _explain_validation_error(exc)
    if isinstance(exc, yaml.YAMLError):
        return _explain_yaml_syntax_error(exc)
    text = str(exc).strip().splitlines()
    return YamlErrorExplanation(summary=text[0] if text else exc.__class__.__name__)


def _explain_validation_error(exc: ValidationError) -> YamlErrorExplanation:
    errors = exc.errors()
    if not errors:
        return YamlErrorExplanation(summary=str(exc))
    primary = _format_error_entry(errors[0])
    if len(errors) > 1:
        primary = f"{primary}  (+{len(errors) - 1} more)"
    return YamlErrorExplanation(summary=primary)


def _format_error_entry(err: Mapping[str, Any]) -> str:
    err_type = str(err.get("type", ""))
    raw_loc = err.get("loc", ())
    loc: tuple[object, ...] = raw_loc if isinstance(raw_loc, tuple) else ()
    loc_str = ".".join(str(x) for x in loc) if loc else "<root>"
    msg = str(err.get("msg", ""))

    if err_type == "missing":
        return f"missing required field `{loc_str}`"
    if err_type == "extra_forbidden":
        return f"unknown field `{loc_str}` (remove it or check spelling)"
    if err_type == "string_pattern_mismatch":
        return f"`{loc_str}` has invalid format: must match the allowed name pattern"
    if err_type in {"int_parsing", "int_type"}:
        return f"`{loc_str}` must be an integer"
    if err_type in {"float_parsing", "float_type"}:
        return f"`{loc_str}` must be a number"
    if err_type == "string_type":
        return f"`{loc_str}` must be a string"
    if err_type == "bool_parsing" or err_type == "bool_type":
        return f"`{loc_str}` must be true or false"
    if err_type in {"greater_than_equal", "less_than_equal", "greater_than", "less_than"}:
        return f"`{loc_str}`: {msg}"
    if err_type == "value_error":
        return _format_value_error(loc_str, msg)
    if err_type == "list_type":
        return f"`{loc_str}` must be a list"
    if err_type == "dict_type":
        return f"`{loc_str}` must be a mapping"

    # Fallback: keep pydantic's raw msg, prefixed with location.
    return f"`{loc_str}`: {msg}"


def _format_value_error(loc_str: str, msg: str) -> str:
    """Strip the `Value error, ` prefix pydantic adds, reuse vllmops wording where present."""
    cleaned = msg.removeprefix("Value error, ")
    if "must start with '--'" in cleaned:
        return f"`{loc_str}` must start with `--` (vllm uses GNU-style flags)"
    if "reserved profile name" in cleaned:
        return f"`{loc_str}`: profile name is reserved (rename it)"
    if "at most one profile" in cleaned:
        return f"`{loc_str}`: model is listed in more than one profile"
    if "duplicate model name" in cleaned:
        return f"`{loc_str}`: duplicate model name across YAMLs"
    if "invalid profile name" in cleaned:
        return f"`{loc_str}`: profile name must match the allowed pattern"
    return f"`{loc_str}`: {cleaned}"


def _explain_yaml_syntax_error(exc: yaml.YAMLError) -> YamlErrorExplanation:
    mark = getattr(exc, "problem_mark", None)
    if mark is not None:
        problem = getattr(exc, "problem", None) or str(exc).strip().splitlines()[0]
        return YamlErrorExplanation(
            summary=f"YAML syntax error at line {mark.line + 1}, col {mark.column + 1}: {problem}"
        )
    text = str(exc).strip().splitlines()
    return YamlErrorExplanation(summary=text[0] if text else "YAML syntax error")
