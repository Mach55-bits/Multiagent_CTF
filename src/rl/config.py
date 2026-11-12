"""RL experiment configuration.

The YAML file is the single source of truth: every field below is required, so no
hyperparameter is ever hidden in code. Load with :func:`load_rl_config`, which also
accepts dotted overrides (``{"ppo.learning_rate": 1e-4}``) that go through the same
strict validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ctf.config import CTFConfig, config_from_dict, config_to_dict, load_config
from rl.schema import ConfigError, apply_overrides, from_dict, read_yaml
from strategies.roles import Role


class LearnerSide(str, Enum):
    RED = "red"
    BLUE = "blue"
    RANDOM = "random"


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    seed: int
    device: str  # "cpu", "cuda", "mps" or "auto"
    torch_threads: int
    checkpoint_every_steps: int
    keep_checkpoints: int  # periodic checkpoints to keep (0 = keep all); phase snapshots are never pruned
    log_every_updates: int
    tensorboard: bool


@dataclass(frozen=True)
class EnvSection:
    config_path: str  # base environment YAML (see configs/default.yaml)


@dataclass(frozen=True)
class RolesConfig:
    slots: list[Role]  # role of each team slot, in sorted agent-id order


@dataclass(frozen=True)
class FeaturesConfig:
    mirror_blue: bool  # flip the grid for BLUE so every agent sees "my base on the left"
    staleness_horizon: int  # steps after which a seen cell counts as fully stale


@dataclass(frozen=True)
class RewardsConfig:
    win: float
    loss: float
    timeout: float
    pickup: float
    tagged_while_carrying: float
    tag_enemy_carrier: float
    tag_team_share: float
    attacker_potential: float  # reward per cell of path distance saved while heading for the flag
    attacker_carry_potential: float  # same, while carrying the flag home
    carry_step_penalty: float  # per step while carrying the flag (makes idling with it costly)
    defender_potential: float
    scout_new_cell: float
    scout_anneal_steps: int  # scout bonus decays linearly to 0 over this many steps
    step_penalty: float


@dataclass(frozen=True)
class ModelConfig:
    conv_channels: list[int]
    kernel_size: int
    hidden_size: int


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float
    learning_rate_final: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coef: float
    entropy_coef_final: float
    value_coef: float
    max_grad_norm: float
    epochs: int
    num_minibatches: int
    adam_eps: float
    normalize_advantages: bool
    value_normalization: bool


@dataclass(frozen=True)
class RolloutConfig:
    num_envs: int
    num_workers: int
    rollout_length: int
    total_steps: int


@dataclass(frozen=True)
class OpponentSpec:
    kind: str
    weight: float
    params: dict[str, Any]


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    max_steps: int
    advance_win_rate: float
    window_episodes: int
    fixed_spawn_fraction: float  # share of episodes that use the default (fixed) spawns
    env_overrides: dict[str, Any]  # CTFConfig keys applied on top of the base env config
    opponent_noise: float  # probability that an opponent action is replaced by a random one
    opponents: list[OpponentSpec]


@dataclass(frozen=True)
class CurriculumConfig:
    learner_side: LearnerSide
    phases: list[PhaseConfig]


@dataclass(frozen=True)
class RLConfig:
    run: RunConfig
    env: EnvSection
    roles: RolesConfig
    features: FeaturesConfig
    rewards: RewardsConfig
    model: ModelConfig
    ppo: PPOConfig
    rollout: RolloutConfig
    curriculum: CurriculumConfig

    def __post_init__(self) -> None:
        _validate(self)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate(cfg: RLConfig) -> None:
    p, r = cfg.ppo, cfg.rollout
    _check(0.0 < p.gamma <= 1.0, "ppo.gamma must be in (0, 1]")
    _check(0.0 <= p.gae_lambda <= 1.0, "ppo.gae_lambda must be in [0, 1]")
    _check(p.clip_epsilon > 0, "ppo.clip_epsilon must be > 0")
    _check(p.epochs >= 1 and p.num_minibatches >= 1, "ppo.epochs/num_minibatches must be >= 1")
    _check(p.learning_rate > 0 and p.learning_rate_final >= 0, "learning rates must be positive")
    _check(r.num_envs >= 1 and r.rollout_length >= 1 and r.total_steps >= 1, "rollout sizes must be >= 1")
    _check(0 <= r.num_workers <= r.num_envs, "rollout.num_workers must be in [0, num_envs] (0 = run in the main process)")
    _check(cfg.run.keep_checkpoints >= 0 and cfg.run.checkpoint_every_steps >= 1, "run.keep_checkpoints must be >= 0 and run.checkpoint_every_steps >= 1")
    _check(len(cfg.model.conv_channels) >= 1, "model.conv_channels needs at least one layer")
    _check(cfg.model.kernel_size % 2 == 1, "model.kernel_size must be odd")
    _check(cfg.features.staleness_horizon >= 1, "features.staleness_horizon must be >= 1")
    _check(len(cfg.roles.slots) >= 1, "roles.slots must not be empty")
    _check(cfg.rewards.scout_anneal_steps >= 1, "rewards.scout_anneal_steps must be >= 1")

    phases = cfg.curriculum.phases
    _check(len(phases) >= 1, "curriculum.phases must not be empty")
    for i, ph in enumerate(phases):
        loc = f"curriculum.phases[{i}] ({ph.name})"
        _check(ph.max_steps >= 1, f"{loc}: max_steps must be >= 1")
        _check(0.0 <= ph.advance_win_rate <= 1.0, f"{loc}: advance_win_rate must be in [0, 1]")
        _check(ph.window_episodes >= 1, f"{loc}: window_episodes must be >= 1")
        _check(0.0 <= ph.fixed_spawn_fraction <= 1.0, f"{loc}: fixed_spawn_fraction must be in [0, 1]")
        _check(0.0 <= ph.opponent_noise <= 1.0, f"{loc}: opponent_noise must be in [0, 1]")
        _check(len(ph.opponents) >= 1, f"{loc}: needs at least one opponent")
        _check(all(o.weight > 0 for o in ph.opponents), f"{loc}: opponent weights must be > 0")
    capped = sum(ph.max_steps for ph in phases[:-1])
    _check(capped < r.total_steps, "phase max_steps (excluding the last phase) must be < rollout.total_steps")


def load_base_env_config(cfg: RLConfig, root: Path | None = None) -> CTFConfig:
    """Load the base environment config; relative paths resolve against ``root`` (default: cwd)."""
    path = Path(cfg.env.config_path)
    if not path.is_absolute() and root is not None:
        path = root / path
    env_cfg = load_config(path)
    _check(
        len(cfg.roles.slots) == env_cfg.agents_per_team,
        f"roles.slots has {len(cfg.roles.slots)} entries but the env has "
        f"{env_cfg.agents_per_team} agents per team",
    )
    return env_cfg


def phase_env_config(base: CTFConfig, phase: PhaseConfig, *, fixed_spawns: bool) -> CTFConfig:
    """Base env config with the phase's overrides; ``fixed_spawns`` forces spawn_mode=fixed."""
    merged = {**config_to_dict(base), **phase.env_overrides}
    if fixed_spawns:
        merged["spawn_mode"] = "fixed"
    try:
        return config_from_dict(merged)
    except ValueError as exc:
        raise ConfigError(f"phase {phase.name!r} env_overrides invalid: {exc}") from exc


def load_rl_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RLConfig:
    data = read_yaml(path)
    if overrides:
        data = apply_overrides(data, overrides)
    return from_dict(RLConfig, data)


def config_to_plain(cfg: RLConfig) -> dict[str, Any]:
    from rl.schema import to_dict

    return to_dict(cfg)


__all__ = [
    "ConfigError",
    "LearnerSide",
    "RLConfig",
    "PhaseConfig",
    "OpponentSpec",
    "load_rl_config",
    "load_base_env_config",
    "phase_env_config",
    "config_to_plain",
]
