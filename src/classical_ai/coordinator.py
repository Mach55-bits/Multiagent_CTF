from __future__ import annotations

from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import (
    GlobalObservation,
    PartialTeamObservation,
    TeamObservation,
    DEFAULT_PARTIAL_VISIBILITY_RADIUS,
)

from coordination.shared_map import (
    TeamSharedMaps,
    build_scout_view,
    update_shared_map_from_partial_team,
)

from classical_ai.roles.attacker import AttackerRole
from classical_ai.roles.defender import DefenderRole
from classical_ai.roles.scout import ScoutRole


class ClassicalCoordinator:

    VALID_ROLES = {"attacker", "defender", "scout"}

    def __init__(
        self,
        width: int,
        height: int,
        role_assignments: dict[Team, dict[str, str]] | None = None,
    ) -> None:

        self.team_maps = TeamSharedMaps(width, height)
        self.role_assignments = role_assignments or {}
        self.role_controllers = {}

        self.width = width
        self.height = height

    def get_role_assignments(
        self,
        observation: TeamObservation,
        team: Team,
    ) -> dict[str, str]:

        if team in self.role_assignments:
            assignments = self.role_assignments[team]

            for role in assignments.values():
                if role not in self.VALID_ROLES:
                    raise ValueError(f"Invalid role: {role}")

            return assignments

        agent_ids = sorted(
            agent.agent_id
            for agent in observation.agents.values()
            if agent.team is team and agent.active
        )

        assignments = {}

        if not agent_ids:
            return assignments

        assignments[agent_ids[0]] = "attacker"

        if len(agent_ids) >= 2:
            assignments[agent_ids[1]] = "defender"

        for agent_id in agent_ids[2:]:
            assignments[agent_id] = "scout"

        return assignments

    def initialize_roles(self, team: Team) -> None:

        if team in self.role_controllers:
            return

        shared_map = self.team_maps.for_team(team)

        self.role_controllers[team] = {
            "attacker": AttackerRole(shared_map),
            "defender": DefenderRole(shared_map),
            "scout": ScoutRole(shared_map),
        }

    def update_team_knowledge(
        self,
        observation: TeamObservation,
        team: Team,
    ) -> None:

        shared_map = self.team_maps.for_team(team)

        if isinstance(observation, PartialTeamObservation):

            if observation.team is not team:
                raise ValueError("Observation belongs to a different team")

            update_shared_map_from_partial_team(
                shared_map,
                observation,
            )

        elif isinstance(observation, GlobalObservation):

            for agent_id, agent in observation.agents.items():

                if agent.team is not team or not agent.active:
                    continue

                scout_view = build_scout_view(
                    observation=observation,
                    team=team,
                    scout_agent_id=agent_id,
                    visibility_radius=DEFAULT_PARTIAL_VISIBILITY_RADIUS,
                )

                shared_map.update_from_scout(scout_view)

    def act(
        self,
        observation: TeamObservation,
        team: Team,
    ) -> dict[str, Action]:

        if observation.width != self.width or observation.height != self.height:
            raise ValueError("Observation dimensions do not match coordinator map")

        self.initialize_roles(team)

        self.update_team_knowledge(observation, team)

        assignments = self.get_role_assignments(observation, team)

        controllers = self.role_controllers[team]

        actions = {}

        for agent_id, role_name in assignments.items():

            agent = observation.agents.get(agent_id)

            if agent is None or not agent.active or agent.team is not team:
                continue

            role = controllers[role_name]

            actions[agent_id] = role.act(
                agent_id=agent_id,
                observation=observation,
                team=team,
            )

        return actions