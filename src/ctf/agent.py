from dataclasses import dataclass, replace
from enum import Enum


class Team(str, Enum):
    RED = "RED"
    BLUE = "BLUE"

    @property
    def opponent(self) -> "Team":
        return Team.BLUE if self is Team.RED else Team.RED


@dataclass(frozen=True)
class Agent:
    agent_id: str
    team: Team
    x: int
    y: int
    active: bool = True
    carrying_flag: bool = False

    def with_position(self, x: int, y: int) -> "Agent":
        return replace(self, x=x, y=y)

    def with_active(self, active: bool) -> "Agent":
        return replace(self, active=active)

    def with_carrying_flag(self, carrying: bool) -> "Agent":
        return replace(self, carrying_flag=carrying)
