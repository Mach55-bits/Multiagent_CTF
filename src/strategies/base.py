from abc import ABC, abstractmethod

from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import TeamObservation


class BaseStrategy(ABC):
    """Common interface for team controllers."""

    def reset(self) -> None:
        """Clear any per-episode memory. Called at the start of every episode.

        Stateless strategies can ignore this. Stateful ones (belief maps, plans,
        recurrent state) must override it so one instance can be reused across
        episodes without leaking information from the previous one.
        """

    @abstractmethod
    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        """
        Choose actions for the given team's active agents.

        Returns a map of agent_id -> Action suitable for CTFEnvironment.step().
        """
