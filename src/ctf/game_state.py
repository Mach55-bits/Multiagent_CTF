"""Full environment state representation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ctf.agent import Agent, Team
from ctf.config import CTFConfig
from ctf.map import GridMap


class FlagStatus(str, Enum):
    AT_BASE = "AT_BASE"
    CARRIED = "CARRIED"


@dataclass(frozen=True)
class Flag:
    team: Team
    status: FlagStatus
    x: int
    y: int
    carrier_id: str | None = None

    def at_base_position(self, config: CTFConfig) -> tuple[int, int]:
        return config.red_base if self.team is Team.RED else config.blue_base

    def with_at_base(self, config: CTFConfig) -> "Flag":
        bx, by = self.at_base_position(config)
        return Flag(team=self.team, status=FlagStatus.AT_BASE, x=bx, y=by, carrier_id=None)

    def with_carried(self, carrier_id: str, x: int, y: int) -> "Flag":
        return Flag(
            team=self.team,
            status=FlagStatus.CARRIED,
            x=x,
            y=y,
            carrier_id=carrier_id,
        )

    def with_position(self, x: int, y: int) -> "Flag":
        return replace(self, x=x, y=y)


@dataclass
class GameState:
    grid: GridMap
    agents: dict[str, Agent]
    flags: dict[Team, Flag]
    red_base: tuple[int, int]
    blue_base: tuple[int, int]
    timestep: int = 0
    winner: Team | None = None
    terminated: bool = False
    truncation: bool = False
    tagging_enabled: bool = True

    @classmethod
    def initial(cls, config: CTFConfig) -> "GameState":
        grid = GridMap.from_config(config)
        agents: dict[str, Agent] = {}
        for i, (x, y) in enumerate(config.default_spawns(Team.RED), start=1):
            agents[f"R{i}"] = Agent(f"R{i}", Team.RED, x, y)
        for i, (x, y) in enumerate(config.default_spawns(Team.BLUE), start=1):
            agents[f"B{i}"] = Agent(f"B{i}", Team.BLUE, x, y)

        flags = {
            Team.RED: Flag(
                Team.RED,
                FlagStatus.AT_BASE,
                config.red_base[0],
                config.red_base[1],
            ),
            Team.BLUE: Flag(
                Team.BLUE,
                FlagStatus.AT_BASE,
                config.blue_base[0],
                config.blue_base[1],
            ),
        }
        return cls(
            grid=grid,
            agents=agents,
            flags=flags,
            red_base=config.red_base,
            blue_base=config.blue_base,
            tagging_enabled=config.tagging_enabled,
        )

    def copy(self) -> "GameState":
        return GameState(
            grid=self.grid,
            agents=dict(self.agents),
            flags={k: v for k, v in self.flags.items()},
            red_base=self.red_base,
            blue_base=self.blue_base,
            timestep=self.timestep,
            winner=self.winner,
            terminated=self.terminated,
            truncation=self.truncation,
            tagging_enabled=self.tagging_enabled,
        )
