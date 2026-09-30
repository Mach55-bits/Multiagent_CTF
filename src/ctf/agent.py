from dataclasses import dataclass
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
        return Agent(
            agent_id=self.agent_id,
            team=self.team,
            x=x,
            y=y,
            active=self.active,
            carrying_flag=self.carrying_flag,
        )

    def with_active(self, active: bool) -> "Agent":
        return Agent(
            agent_id=self.agent_id,
            team=self.team,
            x=self.x,
            y=self.y,
            active=active,
            carrying_flag=self.carrying_flag,
        )

    def with_carrying_flag(self, carrying: bool) -> "Agent":
        return Agent(
            agent_id=self.agent_id,
            team=self.team,
            x=self.x,
            y=self.y,
            active=self.active,
            carrying_flag=carrying,
        )
