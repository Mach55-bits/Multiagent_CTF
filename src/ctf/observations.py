"""Observation builders for strategy code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Any

from ctf.agent import Agent, Team
from ctf.config import DEFAULT_VISIBILITY_RADIUS
from ctf.game_state import Flag, GameState

# Deprecated alias kept so existing imports keep working; prefer DEFAULT_VISIBILITY_RADIUS.
DEFAULT_PARTIAL_VISIBILITY_RADIUS = DEFAULT_VISIBILITY_RADIUS


class ObservationMode(str, Enum):
    GLOBAL = "global"
    PARTIAL = "partial"


@cache
def manhattan_cells(
    cx: int,
    cy: int,
    radius: int,
    width: int,
    height: int,
) -> frozenset[tuple[int, int]]:
    """Cells within Manhattan ``radius`` of (cx, cy). Cached: the result is immutable."""
    cells: set[tuple[int, int]] = set()
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if abs(x - cx) + abs(y - cy) <= radius:
                cells.add((x, y))
    return frozenset(cells)


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def _observation_dict(obs: "GlobalObservation | PartialTeamObservation") -> dict[str, Any]:
    return {
        "mode": obs.mode.value,
        "width": obs.width,
        "height": obs.height,
        "traversable": obs.traversable,
        "obstacles": list(obs.obstacles),
        "red_base": obs.red_base,
        "blue_base": obs.blue_base,
        "agents": {aid: agent.__dict__.copy() for aid, agent in obs.agents.items()},
        "flags": {
            team.value: {
                "team": flag.team.value,
                "status": flag.status.value,
                "x": flag.x,
                "y": flag.y,
                "carrier_id": flag.carrier_id,
            }
            for team, flag in obs.flags.items()
        },
        "timestep": obs.timestep,
        "winner": obs.winner.value if obs.winner else None,
        "terminated": obs.terminated,
        "truncation": obs.truncation,
        "tagging_enabled": obs.tagging_enabled,
        "visibility_radius": obs.visibility_radius,
    }


@dataclass(frozen=True)
class GlobalObservation:
    mode: ObservationMode
    width: int
    height: int
    traversable: list[list[bool]]
    obstacles: list[tuple[int, int]]
    red_base: tuple[int, int]
    blue_base: tuple[int, int]
    agents: dict[str, Agent]
    flags: dict[Team, Flag]
    timestep: int
    winner: Team | None
    terminated: bool
    truncation: bool
    tagging_enabled: bool
    # Not used to filter this observation; carried so strategies know the
    # radius the environment is configured with.
    visibility_radius: int = DEFAULT_VISIBILITY_RADIUS

    def to_dict(self) -> dict[str, Any]:
        return _observation_dict(self)


@dataclass(frozen=True)
class PartialTeamObservation:
    """
    Team-level partial view: full info for friendly agents; enemies and enemy
    flag only if within visibility_radius (Manhattan) of at least one *active*
    friendly agent.
    """

    mode: ObservationMode
    team: Team
    visibility_radius: int
    width: int
    height: int
    traversable: list[list[bool]]
    obstacles: list[tuple[int, int]]
    red_base: tuple[int, int]
    blue_base: tuple[int, int]
    agents: dict[str, Agent]
    flags: dict[Team, Flag]
    timestep: int
    winner: Team | None
    terminated: bool
    truncation: bool
    tagging_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        out = _observation_dict(self)
        out["team"] = self.team.value
        return out


TeamObservation = GlobalObservation | PartialTeamObservation


def _agent_visible_to_team(agent: Agent, team: Team, observers: list[Agent], radius: int) -> bool:
    if agent.team is team:
        return True
    if not agent.active:
        return False
    return any(
        manhattan_distance(o.x, o.y, agent.x, agent.y) <= radius for o in observers
    )


def _flag_visible_to_team(flag: Flag, team: Team, observers: list[Agent], radius: int) -> bool:
    if flag.team is team:
        return True
    return any(
        manhattan_distance(o.x, o.y, flag.x, flag.y) <= radius for o in observers
    )


def build_partial_team_observation(
    state: GameState,
    team: Team,
    visibility_radius: int | None = None,
) -> PartialTeamObservation:
    radius = state.visibility_radius if visibility_radius is None else visibility_radius
    # Eliminated agents see nothing: only active friendlies act as observers.
    observers = [a for a in state.agents.values() if a.team is team and a.active]
    visible_agents = {
        aid: agent
        for aid, agent in state.agents.items()
        if _agent_visible_to_team(agent, team, observers, radius)
    }
    visible_flags = {
        flag_team: flag
        for flag_team, flag in state.flags.items()
        if _flag_visible_to_team(flag, team, observers, radius)
    }
    return PartialTeamObservation(
        mode=ObservationMode.PARTIAL,
        team=team,
        visibility_radius=radius,
        width=state.grid.width,
        height=state.grid.height,
        traversable=state.grid.traversable_grid(),
        obstacles=sorted(state.grid.obstacles),
        red_base=state.red_base,
        blue_base=state.blue_base,
        agents=visible_agents,
        flags=visible_flags,
        timestep=state.timestep,
        winner=state.winner,
        terminated=state.terminated,
        truncation=state.truncation,
        tagging_enabled=state.tagging_enabled,
    )


def build_observation(
    state: GameState,
    mode: ObservationMode = ObservationMode.GLOBAL,
    *,
    team: Team | None = None,
    visibility_radius: int | None = None,
) -> TeamObservation:
    """Build an observation. ``visibility_radius`` defaults to the state's configured radius."""
    if mode is ObservationMode.GLOBAL:
        return GlobalObservation(
            mode=ObservationMode.GLOBAL,
            width=state.grid.width,
            height=state.grid.height,
            traversable=state.grid.traversable_grid(),
            obstacles=sorted(state.grid.obstacles),
            red_base=state.red_base,
            blue_base=state.blue_base,
            agents=dict(state.agents),
            flags=dict(state.flags),
            timestep=state.timestep,
            winner=state.winner,
            terminated=state.terminated,
            truncation=state.truncation,
            tagging_enabled=state.tagging_enabled,
            visibility_radius=(
                state.visibility_radius if visibility_radius is None else visibility_radius
            ),
        )
    if mode is ObservationMode.PARTIAL:
        if team is None:
            raise ValueError("team is required for PARTIAL observations")
        return build_partial_team_observation(state, team, visibility_radius)
    raise ValueError(f"Unsupported observation mode: {mode}")
