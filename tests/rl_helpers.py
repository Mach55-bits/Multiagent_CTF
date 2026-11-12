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


def tiny_config(
    output_dir: str,
    *,
    num_envs: int = 2,
    num_workers: int = 0,
    rollout_length: int = 8,
    total_steps: int = 48,
    phases: int = 1,
    checkpoint_every_steps: int = 16,
    first_phase_max_steps: int | None = None,
) -> RLConfig:
    """A fast config for trainer tests: tiny network, few envs, a handful of updates."""
    cfg = rl_config()
    all_phases = cfg.curriculum.phases
    chosen = all_phases[1:2] if phases == 1 else all_phases[:phases]
    if first_phase_max_steps is not None:
        chosen = [replace(chosen[0], max_steps=first_phase_max_steps), *chosen[1:]]
    return replace(
        cfg,
        run=replace(cfg.run, output_dir=output_dir, name="t", tensorboard=False,
                    checkpoint_every_steps=checkpoint_every_steps, keep_checkpoints=2),
        env=replace(cfg.env, config_path=str(ROOT / "configs" / "default.yaml")),
        model=replace(cfg.model, conv_channels=[4], hidden_size=16),
        rollout=replace(cfg.rollout, num_envs=num_envs, num_workers=num_workers,
                        rollout_length=rollout_length, total_steps=total_steps),
        curriculum=replace(cfg.curriculum, phases=chosen),
    )
