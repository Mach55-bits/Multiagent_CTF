"""Measure training throughput (rollout + PPO update) to choose ``rollout.num_workers``,
``run.torch_threads`` and the model size.

    python -m rl.benchmark --workers 0 1 2 4 --updates 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from rl.buffer import RolloutBuffer
from rl.config import load_base_env_config, load_rl_config
from rl.mappo import MAPPO, ValueNorm
from rl.networks import ActorCritic, NetworkSpec
from rl.rollout import collect
from rl.schema import parse_override
from rl.trainer import resolve_device
from rl.vec_env import make_vec_env


def measure(config_path: str, overrides: dict, workers: int, updates: int) -> dict[str, float]:
    """Time ``updates`` full training iterations; returns per-phase seconds and overall steps/s."""
    cfg = load_rl_config(config_path, {**overrides, "rollout.num_workers": workers})
    env_config = load_base_env_config(cfg, root=Path.cwd())
    device = resolve_device(cfg.run.device)
    torch.set_num_threads(cfg.run.torch_threads)
    spec = NetworkSpec.from_env(env_config)
    model = ActorCritic(spec, cfg.model).to(device)
    value_norm = ValueNorm(cfg.ppo.value_normalization).to(device)
    algo = MAPPO(model, cfg.ppo, value_norm)
    buffer = RolloutBuffer(cfg.rollout.rollout_length, cfg.rollout.num_envs, env_config.agents_per_team, spec)
    vec = make_vec_env(cfg, env_config, cfg.run.seed)
    rollout_time = update_time = 0.0

    def iteration() -> tuple[float, float]:
        nonlocal obs
        t0 = time.time()
        obs, _ = collect(vec, obs, model, value_norm, buffer, device)
        t1 = time.time()
        buffer.compute_gae(cfg.ppo.gamma, cfg.ppo.gae_lambda)
        model.train()
        algo.update(buffer.to_batch(device), cfg.ppo.learning_rate, cfg.ppo.entropy_coef)
        return t1 - t0, time.time() - t1

    try:
        vec.set_phase(len(cfg.curriculum.phases) - 1)  # the real (slowest) setting
        obs = vec.reset()
        iteration()  # warm-up
        for _ in range(updates):
            r, u = iteration()
            rollout_time += r
            update_time += u
    finally:
        vec.close()
    steps = cfg.rollout.rollout_length * cfg.rollout.num_envs
    return {
        "rollout_s": rollout_time / updates,
        "update_s": update_time / updates,
        "steps_per_sec": steps * updates / (rollout_time + update_time),
        "total_steps": float(cfg.rollout.total_steps),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rollout throughput benchmark (no learning).")
    p.add_argument("--config", default="configs/rl/default.yaml")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    p.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--updates", type=int, default=3, help="rollouts to time per setting")
    args = p.parse_args(argv)
    overrides = dict(parse_override(t) for t in args.overrides)
    print("workers | rollout s/upd | PPO update s/upd | steps/s | est. hours for run.total_steps")
    for w in args.workers:
        r = measure(args.config, overrides, w, args.updates)
        hours = r["total_steps"] / r["steps_per_sec"] / 3600
        print(f"{w:7d} | {r['rollout_s']:13.2f} | {r['update_s']:16.2f} | {r['steps_per_sec']:7.0f} | {hours:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
