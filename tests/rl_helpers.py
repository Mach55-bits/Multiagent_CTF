"""Shared helpers for the RL tests (not collected as tests)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ctf.config import CTFConfig
from rl.config import RLConfig, load_rl_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RL_YAML = ROOT / "configs" / "rl" / "default.yaml"
OBSTACLES = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]


def rl_config(**section_overrides) -> RLConfig:
    """Default RL config with whole sections replaced, e.g. rewards=replace(cfg.rewards, ...)."""
    cfg = load_rl_config(DEFAULT_RL_YAML)
    return replace(cfg, **section_overrides) if section_overrides else cfg


def small_env_config(**overrides) -> CTFConfig:
    """5x3 open board: RED base (0,1), BLUE base (4,1); RED spawns on the right side."""
    base = dict(
        width=5,
        height=3,
        agents_per_team=3,
        red_base=(0, 1),
        blue_base=(4, 1),
        red_spawns=[(3, 1), (3, 0), (3, 2)],  # R1 (attacker) on the flag row
        blue_spawns=[(0, 0), (0, 2), (1, 2)],  # keeps row 1 (the route home) clear
        visibility_radius=2,
    )
    base.update(overrides)
    return CTFConfig(**base)
