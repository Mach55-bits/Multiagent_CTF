"""Config parsing: single-source defaults, strict keys/types, lossless round trip."""

from pathlib import Path

import pytest

from ctf.config import (
    DEFAULT_VISIBILITY_RADIUS,
    CTFConfig,
    SpawnMode,
    config_from_dict,
    config_to_dict,
    load_config,
)

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown config key"):
        config_from_dict({"widht": 11})


@pytest.mark.parametrize(
    "bad",
    [
        {"width": "11"},
        {"width": True},
        {"tagging_enabled": "yes"},
        {"visibility_radius": 2.5},
        {"red_base": [1]},
        {"red_base": [1, "a"]},
    ],
)
def test_wrong_types_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        config_from_dict(bad)


def test_empty_dict_gives_dataclass_defaults() -> None:
    assert config_from_dict({}) == CTFConfig()


def test_default_yaml_matches_dataclass_defaults_where_not_overridden() -> None:
    from_yaml = load_config(DEFAULT_YAML)
    defaults = CTFConfig()
    assert from_yaml.visibility_radius == defaults.visibility_radius == DEFAULT_VISIBILITY_RADIUS
    assert from_yaml.spawn_mode is SpawnMode.FIXED
    assert from_yaml.width == defaults.width and from_yaml.height == defaults.height


def test_round_trip_is_lossless() -> None:
    cfg = load_config(DEFAULT_YAML)
    assert config_from_dict(config_to_dict(cfg)) == cfg
    custom = CTFConfig(spawn_mode=SpawnMode.NEAR_BASE, spawn_radius=2, visibility_radius=5)
    assert config_from_dict(config_to_dict(custom)) == custom


def test_spawn_mode_accepts_strings_and_rejects_garbage() -> None:
    assert CTFConfig(spawn_mode="shuffle").spawn_mode is SpawnMode.SHUFFLE
    with pytest.raises(ValueError):
        CTFConfig(spawn_mode="teleport")


@pytest.mark.parametrize("field", ["visibility_radius", "spawn_radius"])
def test_negative_radii_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        CTFConfig(**{field: -1})


def test_near_base_needs_enough_cells() -> None:
    with pytest.raises(ValueError, match="near_base"):
        CTFConfig(spawn_mode="near_base", spawn_radius=0, agents_per_team=3)
