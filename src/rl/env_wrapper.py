"""Single-environment training wrapper.

Owns one CTFEnvironment per spawn style, one opponent per episode (fresh every time, so
no state leaks between episodes), the learner's features and rewards, and per-episode
statistics. Phase changes (curriculum) take effect at the next ``reset()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig
from ctf.environment import CTFEnvironment
from ctf.observations import ObservationMode
from rl.config import LearnerSide, PhaseConfig, RLConfig, phase_env_config
from rl.features import ActorFeatures, CriticFeatures, FeatureBuilder
from rl.navigation import GridNav
from rl.opponents import build_opponent
from rl.rewards import COMPONENTS, RewardCalculator
from strategies import BaseStrategy


@dataclass(frozen=True)
class EnvStep:
    actor: ActorFeatures
    critic: CriticFeatures
    rewards: np.ndarray  # (A,)
    done: bool
    info: dict  # empty until the episode ends, then the episode summary


@dataclass
class _PhaseRuntime:
    phase: PhaseConfig
    envs: dict[bool, CTFEnvironment]  # keyed by "uses fixed spawns"
    builders: dict[Team, FeatureBuilder]
    rewards: dict[Team, RewardCalculator]
    env_config: CTFConfig


class CTFTrainEnv:
    def __init__(self, cfg: RLConfig, base_env_config: CTFConfig, seed: int) -> None:
        self.cfg = cfg
        self.base_env_config = base_env_config
        self._rng = Random(seed)
        self._runtime: _PhaseRuntime | None = None
        self._pending_phase = 0
        self._scout_scale = 1.0
        # per-episode state
        self._env: CTFEnvironment | None = None
        self._team = Team.RED
        self._opponent: BaseStrategy | None = None
        self._opponent_kind = ""
        self._fixed_spawns = False
        self._prev_agents: dict = {}
        self._stats: dict = {}
        self.num_agents = base_env_config.agents_per_team
        self._build_runtime(0)

    # ------------------------------------------------------------------- phases

    def set_phase(self, index: int) -> None:
        """Switch curriculum phase; applied at the next ``reset()``."""
        if not 0 <= index < len(self.cfg.curriculum.phases):
            raise IndexError(f"phase index {index} out of range")
        self._pending_phase = index

    def set_scout_scale(self, scale: float) -> None:
        self._scout_scale = float(scale)

    @property
    def phase_index(self) -> int:
        assert self._runtime is not None
        return self.cfg.curriculum.phases.index(self._runtime.phase)

    def _build_runtime(self, index: int) -> None:
        phase = self.cfg.curriculum.phases[index]
        envs: dict[bool, CTFEnvironment] = {}
        env_config = phase_env_config(self.base_env_config, phase, fixed_spawns=False)
        for fixed in (True, False):
            conf = phase_env_config(self.base_env_config, phase, fixed_spawns=fixed)
            conf.seed = self._rng.randrange(2**31)
            envs[fixed] = CTFEnvironment(conf)
        traversable = [
            [(x, y) not in set(env_config.obstacles) for x in range(env_config.width)]
            for y in range(env_config.height)
        ]
        nav = GridNav(traversable)
        roles = list(self.cfg.roles.slots)
        builders = {
            t: FeatureBuilder(env_config, t, roles, self.cfg.features) for t in Team
        }
        reward_calcs = {
            t: RewardCalculator(env_config, t, roles, self.cfg.rewards, self.cfg.ppo.gamma, nav)
            for t in Team
        }
        self._runtime = _PhaseRuntime(phase, envs, builders, reward_calcs, env_config)

    # ------------------------------------------------------------------ episodes

    def _choose_team(self) -> Team:
        side = self.cfg.curriculum.learner_side
        if side is LearnerSide.RED:
            return Team.RED
        if side is LearnerSide.BLUE:
            return Team.BLUE
        return self._rng.choice([Team.RED, Team.BLUE])

    def _choose_opponent(self, phase: PhaseConfig, env_config: CTFConfig) -> tuple[BaseStrategy, str]:
        weights = [o.weight for o in phase.opponents]
        spec = self._rng.choices(phase.opponents, weights=weights, k=1)[0]
        return build_opponent(spec, env_config, Random(self._rng.randrange(2**31))), spec.kind

    def reset(self) -> EnvStep:
        if self._runtime is None or self.cfg.curriculum.phases[self._pending_phase] is not self._runtime.phase:
            self._build_runtime(self._pending_phase)
        rt = self._runtime
        assert rt is not None
        phase = rt.phase
        self._fixed_spawns = self._rng.random() < phase.fixed_spawn_fraction
        self._env = rt.envs[self._fixed_spawns]
        self._team = self._choose_team()
        self._opponent, self._opponent_kind = self._choose_opponent(phase, rt.env_config)
        self._opponent.reset()

        self._env.reset()
        builder = rt.builders[self._team]
        builder.reset()
        glob = self._env.observe(ObservationMode.GLOBAL)
        rt.rewards[self._team].reset(glob)
        self._prev_agents = dict(glob.agents)
        self._stats = {
            "returns": np.zeros(self.num_agents),
            "components": {name: np.zeros(self.num_agents) for name in COMPONENTS},
            "pickups": 0,
            "tags_inflicted": 0,
            "tags_suffered": 0,
            "length": 0,
        }
        actor = builder.actor_features(self._env.observe(ObservationMode.PARTIAL, team=self._team))
        critic = builder.critic_features(glob)
        return EnvStep(actor, critic, np.zeros(self.num_agents), False, {})

    def step(self, actions: np.ndarray) -> EnvStep:
        rt, env = self._runtime, self._env
        assert rt is not None and env is not None and self._opponent is not None
        team, opp_team = self._team, self._team.opponent
        builder = rt.builders[team]

        mine: dict[str, Action] = {}
        for slot, agent_id in enumerate(builder.agent_ids):
            if self._prev_agents[agent_id].active:
                mine[agent_id] = builder.canon.to_env_action(int(actions[slot]))

        opp_obs = env.observe(ObservationMode.PARTIAL, team=opp_team)
        theirs = self._opponent.act(opp_obs, opp_team)
        noise = rt.phase.opponent_noise
        if noise > 0:
            all_actions = list(Action)
            theirs = {
                aid: (self._rng.choice(all_actions) if self._rng.random() < noise else act)
                for aid, act in theirs.items()
            }

        _, _, done, info = env.step({**mine, **theirs})
        glob = env.observe(ObservationMode.GLOBAL)
        result = rt.rewards[team].step(glob, self._prev_agents, info.get("events", []), self._scout_scale)
        self._prev_agents = dict(glob.agents)

        self._stats["returns"] += result.total
        for name, values in result.components.items():
            self._stats["components"][name] += values
        self._stats["length"] += 1
        for event in info.get("events", []):
            if event.get("type") == "flag_pickup" and event["agent_id"] in builder.agent_ids:
                self._stats["pickups"] += 1
            elif event.get("type") == "tag":
                if event["agent_id"] in builder.agent_ids:
                    self._stats["tags_suffered"] += 1
                if event["by_agent_id"] in builder.agent_ids:
                    self._stats["tags_inflicted"] += 1

        actor = builder.actor_features(env.observe(ObservationMode.PARTIAL, team=team))
        critic = builder.critic_features(glob)
        return EnvStep(actor, critic, result.total, bool(done), self._summary(glob) if done else {})

    def _summary(self, glob) -> dict:
        if glob.winner is self._team:
            outcome = "win"
        elif glob.winner is None:
            outcome = "draw"
        else:
            outcome = "loss"
        return {
            "outcome": outcome,
            "length": self._stats["length"],
            "learner_team": self._team.value,
            "opponent": self._opponent_kind,
            "phase": self._runtime.phase.name,  # type: ignore[union-attr]
            "fixed_spawns": self._fixed_spawns,
            "pickups": self._stats["pickups"],
            "tags_inflicted": self._stats["tags_inflicted"],
            "tags_suffered": self._stats["tags_suffered"],
            "return": self._stats["returns"].tolist(),
            "components": {k: v.tolist() for k, v in self._stats["components"].items()},
        }
