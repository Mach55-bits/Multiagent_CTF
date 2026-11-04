from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import TeamObservation

from coordination.shared_map import SharedTeamMap

from classical_ai.behavior_tree import decide_action
from classical_ai.influence_map import build_influence_map


class AttackerRole:

    def __init__(self, shared_map: SharedTeamMap | None = None):
        self.shared_map = shared_map

    def act(
        self,
        agent_id: str,
        observation: TeamObservation,
        team: Team,
    ) -> Action:

        agent = observation.agents.get(agent_id)

        if agent is None or not agent.active or agent.team is not team:
            return Action.STAY

        # Calculating dangerous areas using visible enemies
        influence = build_influence_map(
            observation=observation,
            team=team,
            shared_map=self.shared_map,
        )

        # Using the behavior tree to select the objective and A* to determine the next movement.
        return decide_action(
            agent_id=agent_id,
            observation=observation,
            team=team,
            influence_map=influence,
            algorithm="astar",
        )