from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import TeamObservation

from strategies.base import BaseStrategy


class StayStrategy(BaseStrategy):
    """Minimal strategy: all active agents on the team STAY."""

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        return {
            agent_id: Action.STAY
            for agent_id, agent in observation.agents.items()
            if agent.team is team and agent.active
        }
