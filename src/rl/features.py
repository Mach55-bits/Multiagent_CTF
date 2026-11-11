"""Observation -> tensors.

Actor input: what the *team* can know (fog of war, shared belief memory fed by every
active teammate, scouts included). Critic input: the true global state, used only
during training. Both are built in the team-relative orientation (see canonical.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coordination.shared_map import SharedTeamMap, update_shared_map_from_partial_team
from ctf.actions import Action
from ctf.agent import Agent, Team
from ctf.config import CTFConfig
from ctf.game_state import FlagStatus
from ctf.observations import GlobalObservation, PartialTeamObservation
from rl.canonical import ACTION_INDEX, NUM_ACTIONS, Canonicalizer
from rl.config import FeaturesConfig
from strategies.roles import Role

ROLE_ORDER: tuple[Role, ...] = tuple(Role)

ACTOR_CHANNELS = (
    "walls",
    "home_zone",
    "own_base",
    "enemy_base",
    "friendly_agents",
    "friendly_carriers",
    "visible_enemies",
    "visible_enemy_carriers",
    "own_flag",
    "enemy_flag_visible",
    "enemy_belief",
    "observed",
    "staleness",
    "currently_visible",
    "self",
)
CRITIC_CHANNELS = (
    "walls",
    "home_zone",
    "own_base",
    "enemy_base",
    "own_agents",
    "own_carriers",
    "enemy_agents",
    "enemy_carriers",
    "own_flag",
    "enemy_flag",
    "self",
)
_SCALARS = (
    "self_carrying",
    "own_flag_taken",
    "team_has_enemy_flag",
    "time_fraction",
    "friendly_alive_fraction",
    "enemy_info_fraction",  # actor: visible enemies / team size; critic: alive enemies / team size
    "team_flag",
)
ACTOR_VECTOR = tuple(f"role_{r.value}" for r in ROLE_ORDER) + _SCALARS
CRITIC_VECTOR = ACTOR_VECTOR

_STAY = ACTION_INDEX[Action.STAY]


@dataclass(frozen=True)
class ActorFeatures:
    grid: np.ndarray  # (A, C, H, W) float32
    vec: np.ndarray  # (A, V) float32
    mask: np.ndarray  # (A, NUM_ACTIONS) bool, True = allowed
    alive: np.ndarray  # (A,) bool
    role_index: np.ndarray  # (A,) int64, index into ROLE_ORDER


@dataclass(frozen=True)
class CriticFeatures:
    grid: np.ndarray  # (A, C, H, W) float32
    vec: np.ndarray  # (A, V) float32


class FeatureBuilder:
    """Per-team feature extractor holding the team's belief memory for one episode."""

    def __init__(
        self,
        env_config: CTFConfig,
        team: Team,
        roles: list[Role],
        cfg: FeaturesConfig,
    ) -> None:
        self.env_config = env_config
        self.team = team
        self.cfg = cfg
        self.width, self.height = env_config.width, env_config.height
        self.canon = Canonicalizer(self.width, team, cfg.mirror_blue)
        prefix = "R" if team is Team.RED else "B"
        self.agent_ids = [f"{prefix}{i}" for i in range(1, env_config.agents_per_team + 1)]
        if len(roles) != len(self.agent_ids):
            raise ValueError("one role per team slot is required")
        self.role_index = np.array([ROLE_ORDER.index(r) for r in roles], dtype=np.int64)
        self._static = self._build_static()
        self.belief = SharedTeamMap(self.width, self.height, team)

    # ----------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Forget everything seen so far (call at the start of every episode)."""
        self.belief = SharedTeamMap(self.width, self.height, self.team)

    # -------------------------------------------------------------------- static

    def _flip(self, arr: np.ndarray) -> np.ndarray:
        return arr[..., ::-1] if self.canon.flip else arr

    def _build_static(self) -> np.ndarray:
        cfg = self.env_config
        own = cfg.red_base if self.team is Team.RED else cfg.blue_base
        other = cfg.blue_base if self.team is Team.RED else cfg.red_base
        walls = np.zeros((self.height, self.width), dtype=np.float32)
        for x, y in cfg.obstacles:
            walls[y, x] = 1.0
        home = np.zeros_like(walls)
        for y in range(self.height):
            for x in range(self.width):
                d_own = abs(x - own[0]) + abs(y - own[1])
                d_other = abs(x - other[0]) + abs(y - other[1])
                home[y, x] = 1.0 if d_own < d_other else 0.0
        own_base = np.zeros_like(walls)
        own_base[own[1], own[0]] = 1.0
        enemy_base = np.zeros_like(walls)
        enemy_base[other[1], other[0]] = 1.0
        return self._flip(np.stack([walls, home, own_base, enemy_base]))

    # ---------------------------------------------------------------- helpers

    def _plane(self, cells: list[tuple[int, int]]) -> np.ndarray:
        plane = np.zeros((self.height, self.width), dtype=np.float32)
        for x, y in cells:
            plane[y, x] = 1.0
        return plane

    def _self_plane(self, agent: Agent) -> np.ndarray:
        plane = np.zeros((self.height, self.width), dtype=np.float32)
        cx, cy = self.canon.cell(agent.x, agent.y)
        plane[cy, cx] = 1.0
        return plane

    def _role_onehot(self, slot: int) -> np.ndarray:
        onehot = np.zeros(len(ROLE_ORDER), dtype=np.float32)
        onehot[self.role_index[slot]] = 1.0
        return onehot

    def _team_agents(self, agents: dict[str, Agent]) -> list[Agent]:
        return [agents[aid] for aid in self.agent_ids]

    # ------------------------------------------------------------------- actor

    def actor_features(self, obs: PartialTeamObservation) -> ActorFeatures:
        """Update the belief with this step's view, then build the actor tensors."""
        update_shared_map_from_partial_team(self.belief, obs)
        t = obs.timestep
        mine = self._team_agents(obs.agents)
        enemies = [a for a in obs.agents.values() if a.team is not self.team and a.active]
        own_flag = obs.flags[self.team]
        enemy_flag = obs.flags.get(self.team.opponent)

        observed = np.asarray(self.belief.observed, dtype=bool)
        last = np.array(
            [[-1 if v is None else v for v in row] for row in self.belief.last_observed_timestep],
            dtype=np.int64,
        )
        stale = np.clip((t - last) / self.cfg.staleness_horizon, 0.0, 1.0)
        stale = np.where(observed, stale, 1.0).astype(np.float32)

        dynamic = np.stack(
            [
                self._plane([(a.x, a.y) for a in mine if a.active]),
                self._plane([(a.x, a.y) for a in mine if a.active and a.carrying_flag]),
                self._plane([(a.x, a.y) for a in enemies]),
                self._plane([(a.x, a.y) for a in enemies if a.carrying_flag]),
                self._plane([(own_flag.x, own_flag.y)]),
                self._plane([] if enemy_flag is None else [(enemy_flag.x, enemy_flag.y)]),
                np.asarray(self.belief.enemy_probability, dtype=np.float32),
                observed.astype(np.float32),
                stale,
                (last == t).astype(np.float32),
            ]
        )
        dynamic = self._flip(dynamic).astype(np.float32)

        n_agents = len(self.agent_ids)
        channels = len(ACTOR_CHANNELS)
        grid = np.zeros((n_agents, channels, self.height, self.width), dtype=np.float32)
        grid[:, :4] = self._static
        grid[:, 4:-1] = dynamic

        team_has_flag = float(any(a.active and a.carrying_flag for a in mine))
        own_taken = float(own_flag.status is FlagStatus.CARRIED)
        scalars = [
            None,  # self_carrying (per agent)
            own_taken,
            team_has_flag,
            t / self.env_config.max_episode_steps,
            sum(a.active for a in mine) / n_agents,
            len(enemies) / n_agents,
            self.canon.team_flag,
        ]
        vec = np.zeros((n_agents, len(ACTOR_VECTOR)), dtype=np.float32)
        mask = np.zeros((n_agents, NUM_ACTIONS), dtype=bool)
        alive = np.zeros(n_agents, dtype=bool)
        for slot, agent in enumerate(mine):
            grid[slot, -1] = self._self_plane(agent)
            vec[slot, : len(ROLE_ORDER)] = self._role_onehot(slot)
            row = list(scalars)
            row[0] = float(agent.active and agent.carrying_flag)
            vec[slot, len(ROLE_ORDER) :] = row
            alive[slot] = agent.active
            mask[slot] = self._action_mask(agent, obs.traversable)
        return ActorFeatures(grid, vec, mask, alive, self.role_index.copy())

    def _action_mask(self, agent: Agent, traversable: list[list[bool]]) -> np.ndarray:
        env_mask = [False] * NUM_ACTIONS
        env_mask[_STAY] = True
        if agent.active:
            for action, idx in ACTION_INDEX.items():
                if idx == _STAY:
                    continue
                dx, dy = action.delta()
                nx, ny = agent.x + dx, agent.y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and traversable[ny][nx]:
                    env_mask[idx] = True
        return np.array(self.canon.mask_to_canonical(env_mask), dtype=bool)

    # ------------------------------------------------------------------ critic

    def critic_features(self, obs: GlobalObservation) -> CriticFeatures:
        """Privileged global-state tensors (training only; never used at inference)."""
        mine = self._team_agents(obs.agents)
        theirs = [a for a in obs.agents.values() if a.team is not self.team]
        own_flag = obs.flags[self.team]
        enemy_flag = obs.flags[self.team.opponent]

        dynamic = np.stack(
            [
                self._plane([(a.x, a.y) for a in mine if a.active]),
                self._plane([(a.x, a.y) for a in mine if a.active and a.carrying_flag]),
                self._plane([(a.x, a.y) for a in theirs if a.active]),
                self._plane([(a.x, a.y) for a in theirs if a.active and a.carrying_flag]),
                self._plane([(own_flag.x, own_flag.y)]),
                self._plane([(enemy_flag.x, enemy_flag.y)]),
            ]
        )
        dynamic = self._flip(dynamic).astype(np.float32)

        n_agents = len(self.agent_ids)
        grid = np.zeros(
            (n_agents, len(CRITIC_CHANNELS), self.height, self.width), dtype=np.float32
        )
        grid[:, :4] = self._static
        grid[:, 4:-1] = dynamic

        scalars = [
            None,
            float(own_flag.status is FlagStatus.CARRIED),
            float(any(a.active and a.carrying_flag for a in mine)),
            obs.timestep / self.env_config.max_episode_steps,
            sum(a.active for a in mine) / n_agents,
            sum(a.active for a in theirs) / n_agents,
            self.canon.team_flag,
        ]
        vec = np.zeros((n_agents, len(CRITIC_VECTOR)), dtype=np.float32)
        for slot, agent in enumerate(mine):
            grid[slot, -1] = self._self_plane(agent)
            vec[slot, : len(ROLE_ORDER)] = self._role_onehot(slot)
            row = list(scalars)
            row[0] = float(agent.active and agent.carrying_flag)
            vec[slot, len(ROLE_ORDER) :] = row
        return CriticFeatures(grid, vec)
