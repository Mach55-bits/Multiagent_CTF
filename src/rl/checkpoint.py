"""Checkpoint saving/loading.

A checkpoint is self-contained: weights, optimizer, value normalizer, curriculum
state, counters, the resolved configs, the git commit, and a record of the
observation layout. Loading refuses a checkpoint whose observation layout does not
match the current code, instead of silently feeding the network scrambled inputs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from ctf.config import CTFConfig, config_from_dict, config_to_dict
from rl.canonical import ACTIONS
from rl.config import RLConfig, config_to_plain
from rl.curriculum import Curriculum
from rl.features import (
    ACTOR_CHANNELS,
    ACTOR_VECTOR,
    CRITIC_CHANNELS,
    CRITIC_VECTOR,
    ROLE_ORDER,
)
from rl.mappo import ValueNorm
from rl.networks import ActorCritic, NetworkSpec
from rl.schema import from_dict

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def observation_spec() -> dict[str, list[str]]:
    return {
        "actor_channels": list(ACTOR_CHANNELS),
        "actor_vector": list(ACTOR_VECTOR),
        "critic_channels": list(CRITIC_CHANNELS),
        "critic_vector": list(CRITIC_VECTOR),
        "roles": [r.value for r in ROLE_ORDER],
        "actions": [a.name for a in ACTIONS],
    }


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_checkpoint(
    path: Path,
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    value_norm: ValueNorm,
    curriculum: Curriculum,
    counters: dict[str, int],
    config: RLConfig,
    env_config: CTFConfig,
) -> None:
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "value_norm": value_norm.state_dict(),
        "curriculum": curriculum.state_dict(),
        "counters": dict(counters),
        "torch_rng": torch.get_rng_state(),
        "config": config_to_plain(config),
        "env_config": config_to_dict(env_config),
        "obs_spec": observation_spec(),
        "git_commit": git_commit(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(
            f"{path}: unsupported checkpoint format {payload.get('format_version')!r} "
            f"(this code reads format {FORMAT_VERSION})"
        )
    saved, current = payload["obs_spec"], observation_spec()
    if saved != current:
        diff = sorted(k for k in current if saved.get(k) != current[k])
        raise CheckpointError(
            f"{path}: observation layout differs from the current code in {diff}; "
            "this checkpoint was trained with a different feature set"
        )
    return payload


def restore_training_state(
    payload: dict[str, Any],
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    value_norm: ValueNorm,
    curriculum: Curriculum,
) -> dict[str, int]:
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as exc:
        raise CheckpointError(
            "checkpoint weights do not fit the configured model (did model.* change?)"
        ) from exc
    optimizer.load_state_dict(payload["optimizer"])
    value_norm.load_state_dict(payload["value_norm"])
    curriculum.load_state_dict(payload["curriculum"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    return {k: int(v) for k, v in payload["counters"].items()}


def load_policy(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ActorCritic, dict[str, Any]]:
    """Rebuild the trained network for inference (no training state needed)."""
    payload = load_checkpoint(path, map_location="cpu")
    config = from_dict(RLConfig, payload["config"])
    env_config = config_from_dict(payload["env_config"])
    model = ActorCritic(NetworkSpec.from_env(env_config), config.model)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    meta = {
        "config": config,
        "env_config": env_config,
        "counters": payload["counters"],
        "git_commit": payload["git_commit"],
        "value_norm": payload["value_norm"],
    }
    return model, meta


class CheckpointManager:
    """Writes periodic checkpoints (pruned), a ``latest.pt`` copy, and permanent phase snapshots."""

    def __init__(self, run_dir: Path, keep: int) -> None:
        self.dir = run_dir / "checkpoints"
        self.keep = keep

    @property
    def latest(self) -> Path:
        return self.dir / "latest.pt"

    def save_periodic(self, step: int, **state: Any) -> Path:
        path = self.dir / f"step_{step:010d}.pt"
        save_checkpoint(path, **state)
        tmp = self.latest.with_suffix(".pt.tmp")
        shutil.copyfile(path, tmp)
        os.replace(tmp, self.latest)
        self._prune()
        return path

    def save_phase_start(self, index: int, name: str, **state: Any) -> Path:
        path = self.dir / f"phase_{index}_{name}_start.pt"
        save_checkpoint(path, **state)
        return path

    def _prune(self) -> None:
        if self.keep <= 0:
            return
        periodic = sorted(self.dir.glob("step_*.pt"))
        for old in periodic[: max(len(periodic) - self.keep, 0)]:
            old.unlink()
