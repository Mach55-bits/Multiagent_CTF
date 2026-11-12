"""MAPPO training loop."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ctf.config import CTFConfig
from rl.buffer import RolloutBuffer
from rl.checkpoint import CheckpointManager, load_checkpoint, restore_training_state
from rl.config import RLConfig
from rl.curriculum import Curriculum
from rl.mappo import MAPPO, ValueNorm
from rl.metrics import MetricsLogger, summarize_episodes
from rl.networks import ActorCritic, NetworkSpec
from rl.rollout import collect
from rl.vec_env import make_vec_env


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(max(progress, 0.0), 1.0)


class Trainer:
    def __init__(
        self,
        cfg: RLConfig,
        env_config: CTFConfig,
        run_dir: Path,
        resume_from: Path | None = None,
    ) -> None:
        self.cfg, self.env_config, self.run_dir = cfg, env_config, run_dir
        self.device = resolve_device(cfg.run.device)
        torch.set_num_threads(cfg.run.torch_threads)
        torch.manual_seed(cfg.run.seed)

        spec = NetworkSpec.from_env(env_config)
        self.model = ActorCritic(spec, cfg.model).to(self.device)
        self.value_norm = ValueNorm(cfg.ppo.value_normalization).to(self.device)
        self.algo = MAPPO(self.model, cfg.ppo, self.value_norm)
        self.curriculum = Curriculum(cfg.curriculum.phases)
        self.checkpoints = CheckpointManager(run_dir, cfg.run.keep_checkpoints)
        self.counters = {"total_steps": 0, "update": 0}
        if resume_from is not None:
            payload = load_checkpoint(resume_from, map_location="cpu")
            self.counters = restore_training_state(
                payload,
                model=self.model,
                optimizer=self.algo.optimizer,
                value_norm=self.value_norm,
                curriculum=self.curriculum,
            )
        self.buffer = RolloutBuffer(
            cfg.rollout.rollout_length, cfg.rollout.num_envs, env_config.agents_per_team, spec
        )
        self.generator = None if self.device.type == "mps" else torch.Generator(device=self.device)
        if self.generator is not None:
            self.generator.manual_seed(cfg.run.seed + self.counters["update"])

    # ------------------------------------------------------------------ helpers

    def _state(self) -> dict:
        return {
            "model": self.model,
            "optimizer": self.algo.optimizer,
            "value_norm": self.value_norm,
            "curriculum": self.curriculum,
            "counters": self.counters,
            "config": self.cfg,
            "env_config": self.env_config,
        }

    def _scalars(self, stats, episodes, lr, entropy_coef, scout_scale, sps) -> dict[str, float]:
        roles = [r.value for r in self.cfg.roles.slots]
        scalars = {f"train/{k}": v for k, v in stats.items()}
        scalars.update({f"episode/{k}": v for k, v in summarize_episodes(episodes, roles).items()})
        scalars["curriculum/phase"] = float(self.curriculum.index)
        if self.curriculum.win_rate is not None:
            scalars["curriculum/rolling_win_rate"] = self.curriculum.win_rate
        scalars["schedule/learning_rate"] = lr
        scalars["schedule/entropy_coef"] = entropy_coef
        scalars["schedule/scout_scale"] = scout_scale
        scalars["perf/steps_per_sec"] = sps
        return scalars

    # --------------------------------------------------------------------- train

    def run(self) -> dict[str, int]:
        cfg = self.cfg
        steps_per_update = cfg.rollout.rollout_length * cfg.rollout.num_envs
        # Resumed runs use different env seeds than the segment they continue.
        seed = cfg.run.seed + self.counters["update"] * cfg.rollout.num_envs
        vec = make_vec_env(cfg, self.env_config, seed)
        logger = MetricsLogger(self.run_dir, cfg.run.tensorboard)
        pending: list[dict] = []
        window_start, window_steps = time.time(), 0
        next_checkpoint = (self.counters["total_steps"] // cfg.run.checkpoint_every_steps + 1) * (
            cfg.run.checkpoint_every_steps
        )
        try:
            vec.set_phase(self.curriculum.index)
            obs = vec.reset()
            while self.counters["total_steps"] < cfg.rollout.total_steps:
                total = self.counters["total_steps"]
                progress = total / cfg.rollout.total_steps
                lr = _lerp(cfg.ppo.learning_rate, cfg.ppo.learning_rate_final, progress)
                entropy_coef = _lerp(cfg.ppo.entropy_coef, cfg.ppo.entropy_coef_final, progress)
                scout_scale = max(0.0, 1.0 - total / cfg.rewards.scout_anneal_steps)
                vec.set_scout_scale(scout_scale)

                obs, episodes = collect(vec, obs, self.model, self.value_norm, self.buffer, self.device)
                self.buffer.compute_gae(cfg.ppo.gamma, cfg.ppo.gae_lambda)
                self.model.train()
                stats = self.algo.update(self.buffer.to_batch(self.device), lr, entropy_coef, self.generator)

                self.counters["total_steps"] += steps_per_update
                self.counters["update"] += 1
                window_steps += steps_per_update
                pending.extend(episodes)

                if self.curriculum.record([e["outcome"] for e in episodes], steps_per_update):
                    phase = self.curriculum.phase
                    vec.set_phase(self.curriculum.index)
                    logger.event(self.counters["total_steps"], "phase_advance",
                                 phase_index=self.curriculum.index, phase=phase.name)
                    self.checkpoints.save_phase_start(self.curriculum.index, phase.name, **self._state())
                    print(f"--> phase {self.curriculum.index}: {phase.name}", flush=True)

                if self.counters["update"] % cfg.run.log_every_updates == 0:
                    sps = window_steps / max(time.time() - window_start, 1e-9)
                    scalars = self._scalars(stats, pending, lr, entropy_coef, scout_scale, sps)
                    logger.log(self.counters["total_steps"], scalars)
                    self._print(scalars)
                    pending, window_start, window_steps = [], time.time(), 0

                if self.counters["total_steps"] >= next_checkpoint:
                    self.checkpoints.save_periodic(self.counters["total_steps"], **self._state())
                    next_checkpoint += cfg.run.checkpoint_every_steps
            self.checkpoints.save_periodic(self.counters["total_steps"], **self._state())
        except KeyboardInterrupt:
            print("\ninterrupted: saving checkpoint...", flush=True)
            self.checkpoints.save_periodic(self.counters["total_steps"], **self._state())
        finally:
            vec.close()
            logger.close()
        return dict(self.counters)

    def _print(self, s: dict[str, float]) -> None:
        phase = self.cfg.curriculum.phases[self.curriculum.index].name
        win = s.get("episode/win_rate")
        draw = s.get("episode/draw_rate")
        episode = f"win {win:.2f} draw {draw:.2f}" if win is not None else "no finished episodes"
        print(
            f"upd {self.counters['update']:6d} | step {self.counters['total_steps'] / 1e6:7.3f}M"
            f" | {s['perf/steps_per_sec']:6.0f} sps | {phase} | {episode}"
            f" | ent {s['train/entropy']:.2f} | kl {s['train/approx_kl']:.4f}"
            f" | ev {s['train/explained_variance']:.2f}",
            flush=True,
        )
