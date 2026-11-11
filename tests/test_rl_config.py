import copy
from dataclasses import replace

import pytest
import yaml

from ctf.config import config_to_dict
from rl.config import (
    RLConfig,
    config_to_plain,
    load_base_env_config,
    load_rl_config,
    phase_env_config,
)
from rl.schema import ConfigError, apply_overrides, from_dict, parse_override
from rl_helpers import DEFAULT_RL_YAML, ROOT


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_RL_YAML.read_text())


def test_default_yaml_loads_and_round_trips() -> None:
    cfg = load_rl_config(DEFAULT_RL_YAML)
    assert from_dict(RLConfig, config_to_plain(cfg)) == cfg


def test_missing_key_is_rejected_with_path() -> None:
    raw = _raw()
    del raw["ppo"]["gamma"]
    with pytest.raises(ConfigError, match=r"ppo.*gamma"):
        from_dict(RLConfig, raw)


def test_unknown_key_is_rejected() -> None:
    raw = _raw()
    raw["ppo"]["learning_rte"] = 1e-3
    with pytest.raises(ConfigError, match="unknown key"):
        from_dict(RLConfig, raw)


@pytest.mark.parametrize(
    "path,value",
    [
        ("ppo.epochs", 2.5),
        ("ppo.epochs", True),
        ("ppo.normalize_advantages", "yes"),
        ("run.seed", "0"),
        ("roles.slots", ["attacker", "wizard", "scout"]),
        ("curriculum.learner_side", "green"),
    ],
)
def test_wrong_types_or_values_are_rejected(path: str, value) -> None:
    data = apply_overrides(_raw(), {path: value})
    with pytest.raises(ConfigError):
        from_dict(RLConfig, data)


def test_overrides_are_applied_and_still_validated() -> None:
    cfg = load_rl_config(DEFAULT_RL_YAML, {"ppo.learning_rate": 1e-4, "curriculum.phases.0.name": "x"})
    assert cfg.ppo.learning_rate == 1e-4 and cfg.curriculum.phases[0].name == "x"
    with pytest.raises(ConfigError, match="unknown key"):
        load_rl_config(DEFAULT_RL_YAML, {"ppo.nonsense": 1})
    with pytest.raises(ConfigError):
        load_rl_config(DEFAULT_RL_YAML, {"ppo.gamma": 1.5})


def test_parse_override_reads_yaml_values() -> None:
    assert parse_override("ppo.learning_rate=1e-4") == ("ppo.learning_rate", 1e-4)
    assert parse_override("features.mirror_blue=false") == ("features.mirror_blue", False)
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")


def test_cross_field_validation() -> None:
    cfg = load_rl_config(DEFAULT_RL_YAML)
    with pytest.raises(ConfigError, match="total_steps"):
        replace(cfg, rollout=replace(cfg.rollout, total_steps=100))
    with pytest.raises(ConfigError, match="num_workers"):
        replace(cfg, rollout=replace(cfg.rollout, num_workers=cfg.rollout.num_envs + 1))


def test_roles_must_match_agents_per_team() -> None:
    cfg = load_rl_config(DEFAULT_RL_YAML)
    bad = replace(cfg, roles=replace(cfg.roles, slots=cfg.roles.slots[:2]))
    with pytest.raises(ConfigError, match="roles.slots"):
        load_base_env_config(bad, ROOT)


def test_phase_env_overrides_merge_and_validate() -> None:
    cfg = load_rl_config(DEFAULT_RL_YAML)
    base = load_base_env_config(cfg, ROOT)
    phase = cfg.curriculum.phases[0]
    merged = phase_env_config(base, phase, fixed_spawns=False)
    assert merged.visibility_radius == phase.env_overrides["visibility_radius"]
    assert phase_env_config(base, phase, fixed_spawns=True).spawn_mode.value == "fixed"
    assert base.width == merged.width
    bad = replace(phase, env_overrides={"widht": 3})
    with pytest.raises(ConfigError, match="env_overrides"):
        phase_env_config(base, bad, fixed_spawns=False)
    assert config_to_dict(base)["visibility_radius"] == base.visibility_radius
