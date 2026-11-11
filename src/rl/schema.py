"""Strict YAML -> dataclass loading.

Every field of a config dataclass must be present in the YAML (no defaults are
hidden in code), unknown keys are rejected, and types are checked. Errors carry
the dotted path of the offending key (e.g. ``ppo.learning_rate``).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised for missing, unknown or wrongly typed configuration values."""


def _where(path: str) -> str:
    return path or "<root>"


def _convert(value: Any, hint: Any, path: str) -> Any:
    origin = typing.get_origin(hint)

    if hint is Any:
        return value

    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(hint)
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != 1:
            raise ConfigError(f"{_where(path)}: unsupported union type {hint}")
        return _convert(value, non_none[0], path)

    if dataclasses.is_dataclass(hint):
        return from_dict(hint, value, path)

    if origin is list:
        (item_hint,) = typing.get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"{_where(path)}: expected a list, got {value!r}")
        return [_convert(v, item_hint, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{_where(path)}: expected a mapping, got {value!r}")
        return dict(value)

    if isinstance(hint, type) and issubclass(hint, Enum):
        try:
            return hint(value)
        except ValueError:
            valid = [e.value for e in hint]
            raise ConfigError(f"{_where(path)}: {value!r} is not one of {valid}") from None

    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{_where(path)}: expected true/false, got {value!r}")
        return value

    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{_where(path)}: expected an integer, got {value!r}")
        return value

    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            hint_text = ""
            if isinstance(value, str):
                try:
                    float(value)
                    hint_text = " (PyYAML reads 1e-4 as text; write 0.0001 or 1.0e-4)"
                except ValueError:
                    pass
            raise ConfigError(f"{_where(path)}: expected a number, got {value!r}{hint_text}")
        return float(value)

    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{_where(path)}: expected a string, got {value!r}")
        return value

    raise ConfigError(f"{_where(path)}: unsupported field type {hint}")


def from_dict(cls: type[T], data: Any, path: str = "") -> T:
    """Build ``cls`` from a mapping, requiring every field and rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"{_where(path)}: expected a mapping, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    fields = dataclasses.fields(cls)
    names = {f.name for f in fields}

    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{_where(path)}: unknown key(s) {unknown}; valid keys: {sorted(names)}")
    missing = sorted(
        f.name
        for f in fields
        if f.name not in data
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ConfigError(f"{_where(path)}: missing required key(s) {missing}")

    kwargs = {
        name: _convert(data[name], hints[name], f"{path}.{name}" if path else name)
        for name in names
        if name in data
    }
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    """YAML-serializable view of a (nested) dataclass."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``a.b.c=value`` (value is read as YAML, so ``1e-4``, ``true``, ``[1,2]`` work)."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like key.path=value")
    key, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # PyYAML treats "1e-4" as a string; accept plain numbers
        for convert in (int, float):
            try:
                return key.strip(), convert(raw)
            except ValueError:
                continue
    return key.strip(), value


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with dotted-key overrides applied (list indexes allowed)."""
    out = yaml.safe_load(yaml.safe_dump(data))  # cheap deep copy of plain data
    for dotted, value in overrides.items():
        node: Any = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node.get(part)
            if node is None:
                raise ConfigError(f"override {dotted!r}: no such section {part!r}")
        last = parts[-1]
        if isinstance(node, list):
            node[int(last)] = value
        elif isinstance(node, dict) and last in node:
            node[last] = value
        else:
            raise ConfigError(f"override {dotted!r}: unknown key {last!r}")
    return out


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: root must be a mapping")
    return data
