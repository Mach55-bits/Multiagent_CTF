"""Vectorised training environments.

``EnvGroup`` runs several ``CTFTrainEnv`` and returns stacked arrays; it auto-resets
finished episodes (the returned observation is then the first of the next episode).
``SubprocVecEnv`` runs one group per worker process (spawn context, so it behaves the
same on macOS, Windows and Linux); ``InProcessVecEnv`` is the same thing without
processes, for debugging and tests. Workers never import torch.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass, field

import numpy as np

from ctf.config import CTFConfig
from rl.config import RLConfig
from rl.env_wrapper import CTFTrainEnv, EnvStep


@dataclass
class VecStep:
    grid: np.ndarray  # (E, A, C, H, W)
    vec: np.ndarray  # (E, A, V)
    mask: np.ndarray  # (E, A, NUM_ACTIONS)
    alive: np.ndarray  # (E, A)
    role_index: np.ndarray  # (E, A)
    critic_grid: np.ndarray  # (E, A, Cc, H, W)
    critic_vec: np.ndarray  # (E, A, Vc)
    rewards: np.ndarray  # (E, A)
    dones: np.ndarray  # (E,)
    infos: list[dict] = field(default_factory=list)  # summaries of episodes finished this step


def _pack(steps: list[EnvStep], infos: list[dict]) -> VecStep:
    return VecStep(
        grid=np.stack([s.actor.grid for s in steps]),
        vec=np.stack([s.actor.vec for s in steps]),
        mask=np.stack([s.actor.mask for s in steps]),
        alive=np.stack([s.actor.alive for s in steps]),
        role_index=np.stack([s.actor.role_index for s in steps]),
        critic_grid=np.stack([s.critic.grid for s in steps]),
        critic_vec=np.stack([s.critic.vec for s in steps]),
        rewards=np.stack([s.rewards for s in steps]),
        dones=np.array([float(s.done) for s in steps]),
        infos=infos,
    )


def _merge(parts: list[VecStep]) -> VecStep:
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts], axis=0)  # noqa: E731
    return VecStep(
        grid=cat("grid"), vec=cat("vec"), mask=cat("mask"), alive=cat("alive"),
        role_index=cat("role_index"), critic_grid=cat("critic_grid"), critic_vec=cat("critic_vec"),
        rewards=cat("rewards"), dones=cat("dones"),
        infos=[info for p in parts for info in p.infos],
    )


class EnvGroup:
    def __init__(self, cfg: RLConfig, base_env_config: CTFConfig, seeds: list[int]) -> None:
        self.envs = [CTFTrainEnv(cfg, base_env_config, seed) for seed in seeds]

    def reset(self) -> VecStep:
        return _pack([env.reset() for env in self.envs], [])

    def step(self, actions: np.ndarray) -> VecStep:
        steps: list[EnvStep] = []
        infos: list[dict] = []
        for env, act in zip(self.envs, actions):
            result = env.step(act)
            if result.done:
                infos.append(result.info)
                fresh = env.reset()
                result = EnvStep(fresh.actor, fresh.critic, result.rewards, True, result.info)
            steps.append(result)
        return _pack(steps, infos)

    def set_phase(self, index: int) -> None:
        for env in self.envs:
            env.set_phase(index)

    def set_scout_scale(self, scale: float) -> None:
        for env in self.envs:
            env.set_scout_scale(scale)


def _split(items: list[int], parts: int) -> list[list[int]]:
    size, extra = divmod(len(items), parts)
    out, start = [], 0
    for i in range(parts):
        end = start + size + (1 if i < extra else 0)
        out.append(items[start:end])
        start = end
    return out


class InProcessVecEnv:
    def __init__(self, cfg: RLConfig, base_env_config: CTFConfig, seed: int) -> None:
        self.num_envs = cfg.rollout.num_envs
        self.num_agents = base_env_config.agents_per_team
        self._group = EnvGroup(cfg, base_env_config, [seed + i for i in range(self.num_envs)])

    def reset(self) -> VecStep:
        return self._group.reset()

    def step(self, actions: np.ndarray) -> VecStep:
        return self._group.step(actions)

    def set_phase(self, index: int) -> None:
        self._group.set_phase(index)

    def set_scout_scale(self, scale: float) -> None:
        self._group.set_scout_scale(scale)

    def close(self) -> None:
        pass


def _worker_main(conn, cfg: RLConfig, base_env_config: CTFConfig, seeds: list[int]) -> None:
    try:
        group = EnvGroup(cfg, base_env_config, seeds)
        while True:
            command, payload = conn.recv()
            if command == "reset":
                conn.send(("ok", group.reset()))
            elif command == "step":
                conn.send(("ok", group.step(payload)))
            elif command == "set_phase":
                group.set_phase(payload)
                conn.send(("ok", None))
            elif command == "set_scout_scale":
                group.set_scout_scale(payload)
                conn.send(("ok", None))
            elif command == "close":
                break
    except Exception:  # report the traceback to the parent instead of dying silently
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class SubprocVecEnv:
    def __init__(self, cfg: RLConfig, base_env_config: CTFConfig, seed: int) -> None:
        self.num_envs = cfg.rollout.num_envs
        self.num_agents = base_env_config.agents_per_team
        ctx = mp.get_context("spawn")
        env_ids = list(range(self.num_envs))
        self._slices = _split(env_ids, cfg.rollout.num_workers)
        self._conns, self._procs = [], []
        for ids in self._slices:
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_main,
                args=(child, cfg, base_env_config, [seed + i for i in ids]),
                daemon=True,
            )
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)

    def _recv_all(self) -> list:
        results = []
        for conn in self._conns:
            status, payload = conn.recv()
            if status == "error":
                self.close()
                raise RuntimeError(f"environment worker crashed:\n{payload}")
            results.append(payload)
        return results

    def reset(self) -> VecStep:
        for conn in self._conns:
            conn.send(("reset", None))
        return _merge(self._recv_all())

    def step(self, actions: np.ndarray) -> VecStep:
        for conn, ids in zip(self._conns, self._slices):
            conn.send(("step", actions[ids[0] : ids[-1] + 1]))
        return _merge(self._recv_all())

    def _broadcast(self, command: str, payload) -> None:
        for conn in self._conns:
            conn.send((command, payload))
        self._recv_all()

    def set_phase(self, index: int) -> None:
        self._broadcast("set_phase", index)

    def set_scout_scale(self, scale: float) -> None:
        self._broadcast("set_scout_scale", float(scale))

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._conns, self._procs = [], []


def make_vec_env(cfg: RLConfig, base_env_config: CTFConfig, seed: int):
    """``rollout.num_workers == 0`` runs in this process; otherwise one process per worker."""
    if cfg.rollout.num_workers == 0:
        return InProcessVecEnv(cfg, base_env_config, seed)
    return SubprocVecEnv(cfg, base_env_config, seed)
