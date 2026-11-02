"""Observation builders for strategy code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ctf.agent import Agent, Team
from ctf.game_state import Flag, GameState

DEFAULT_PARTIAL_VISIBILITY_RADIUS = 3


class ObservationMode(str, Enum):
    GLOBAL = "global"
    PARTIAL = "partial"


def manhattan_cells(
    cx: int,
    cy: int,
    radius: int,
    width: int,
    height: int,
) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            if abs(x - cx) + abs(y - cy) <= radius:
                cells.add((x, y))
    return frozenset(cells)


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "width": self.width,
            "height": self.height,
            "traversable": self.traversable,
            "obstacles": list(self.obstacles),
            "red_base": self.red_base,
            "blue_base": self.blue_base,
            "agents": {aid: agent.__dict__.copy() for aid, agent in self.agents.items()},
            "flags": {
                team.value: {
                    "team": flag.team.value,
                    "status": flag.status.value,
                    "x": flag.x,
                    "y": flag.y,
                    "carrier_id": flag.carrier_id,
                }
                for team, flag in self.flags.items()
            },
            "timestep": self.timestep,
            "winner": self.winner.value if self.winner else None,
            "terminated": self.terminated,
            "truncation": self.truncation,
            "tagging_enabled": self.tagging_enabled,
        }


@dataclass(frozen=True)
class PartialTeamObservation:
    """
    Team-level partial view: full info for friendly agents; enemies and enemy
    flag only if within visibility_radius (Manhattan) of at least one friendly agent.
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
        return {
            "mode": self.mode.value,
            "team": self.team.value,
            "visibility_radius": self.visibility_radius,
            "width": self.width,
            "height": self.height,
            "traversable": self.traversable,
            "obstacles": list(self.obstacles),
            "red_base": self.red_base,
            "blue_base": self.blue_base,
            "agents": {aid: agent.__dict__.copy() for aid, agent in self.agents.items()},
            "flags": {
                team.value: {
                    "team": flag.team.value,
                    "status": flag.status.value,
                    "x": flag.x,
                    "y": flag.y,
                    "carrier_id": flag.carrier_id,
                }
                for team, flag in self.flags.items()
            },
            "timestep": self.timestep,
            "winner": self.winner.value if self.winner else None,
            "terminated": self.terminated,
            "truncation": self.truncation,
            "tagging_enabled": self.tagging_enabled,
        }


TeamObservation = GlobalObservation | PartialTeamObservation


def _friendly_agents(state: GameState, team: Team) -> list[Agent]:
    return [a for a in state.agents.values() if a.team is team]


def _agent_visible_to_team(agent: Agent, team: Team, friendly: list[Agent], radius: int) -> bool:
    if agent.team is team:
        return True
    if not agent.active:
        return False
    for observer in friendly:
        if manhattan_distance(observer.x, observer.y, agent.x, agent.y) <= radius:
            return True
    return False


def _flag_visible_to_team(flag: Flag, team: Team, friendly: list[Agent], radius: int) -> bool:
    if flag.team is team:
        return True
    for observer in friendly:
        if manhattan_distance(observer.x, observer.y, flag.x, flag.y) <= radius:
            return True
    return False


def build_partial_team_observation(
    state: GameState,
    team: Team,
    visibility_radius: int = DEFAULT_PARTIAL_VISIBILITY_RADIUS,
) -> PartialTeamObservation:
    friendly = _friendly_agents(state, team)
    visible_agents = {
        aid: agent
        for aid, agent in state.agents.items()
        if _agent_visible_to_team(agent, team, friendly, visibility_radius)
    }
    visible_flags = {
        flag_team: flag
        for flag_team, flag in state.flags.items()
        if _flag_visible_to_team(flag, team, friendly, visibility_radius)
    }
    return PartialTeamObservation(
        mode=ObservationMode.PARTIAL,
        team=team,
        visibility_radius=visibility_radius,
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
    visibility_radius: int = DEFAULT_PARTIAL_VISIBILITY_RADIUS,
) -> TeamObservation:
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
        )
    if mode is ObservationMode.PARTIAL:
        if team is None:
            raise ValueError("team is required for PARTIAL observations")
        return build_partial_team_observation(state, team, visibility_radius)
    raise ValueError(f"Unsupported observation mode: {mode}")
