"""Observation builders for strategy code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ctf.agent import Agent, Team
from ctf.game_state import Flag, GameState


class ObservationMode(str, Enum):
    GLOBAL = "global"


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


def build_observation(
    state: GameState,
    mode: ObservationMode = ObservationMode.GLOBAL,
) -> GlobalObservation:
    if mode is not ObservationMode.GLOBAL:
        raise ValueError(f"Unsupported observation mode: {mode}")
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
