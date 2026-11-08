
from ctf.actions import Action
from ctf.agent import Team
from ctf.game_state import FlagStatus
from ctf.observations import TeamObservation

from coordination.shared_map import SharedTeamMap

from classical_ai.influence_map import build_influence_map
from classical_ai.pathfinding import next_action


class DefenderRole:
    """
    Defensive role responsible for protecting the team's flag
    and intercepting visible enemy flag carriers.
    """

    def __init__(self, shared_map: SharedTeamMap | None = None):
        self.shared_map = shared_map

    def find_threat(
        self,
        observation: TeamObservation,
        team: Team,
        start: tuple[int, int],
    ) -> tuple[int, int] | None:

        enemy_carriers = [
            agent
            for agent in observation.agents.values()
            if agent.team is team.opponent
            and agent.active
            and agent.carrying_flag
        ]

        if enemy_carriers:
            nearest_enemy = min(
                enemy_carriers,
                key=lambda agent: (
                    abs(agent.x - start[0]) +
                    abs(agent.y - start[1])
                ),
            )

            return (nearest_enemy.x, nearest_enemy.y)

        own_flag = observation.flags.get(team)

        if own_flag is not None and own_flag.status == FlagStatus.CARRIED:
            return (own_flag.x, own_flag.y)

        return None

    def get_defensive_position(
        self,
        observation: TeamObservation,
        team: Team,
    ) -> tuple[int, int]:

        if team is Team.RED:
            return observation.red_base

        return observation.blue_base

    def act(
        self,
        agent_id: str,
        observation: TeamObservation,
        team: Team,
    ) -> Action:

        agent = observation.agents.get(agent_id)

        if agent is None or not agent.active or agent.team is not team:
            return Action.STAY

        start = (agent.x, agent.y)

        influence = build_influence_map(
            observation=observation,
            team=team,
            shared_map=self.shared_map,
        )

        target = self.find_threat(
            observation=observation,
            team=team,
            start=start,
        )

        if target is None:
            target = self.get_defensive_position(observation, team)

        return next_action(
            start=start,
            goal=target,
            observation=observation,
            algorithm="astar",
            influence_map=influence,
        )
