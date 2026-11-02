from abc import ABC, abstractmethod

from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import TeamObservation


class BaseStrategy(ABC):
    """Common interface for team controllers."""

    @abstractmethod
    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        """
        Choose actions for the given team's active agents.

        Returns a map of agent_id -> Action suitable for CTFEnvironment.step().
        """
