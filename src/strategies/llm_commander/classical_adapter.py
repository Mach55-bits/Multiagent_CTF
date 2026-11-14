"""Controller backed by Team 1's classical AI (mixed team: LLM orders + behaviour-tree agents).

Agents that have an order from the commander follow it using Team 1's A* pathfinding with the
influence map (danger-aware routes). Agents without an order run Team 1's own role logic
(attacker / defender / scout), so the LLM only has to intervene when it wants a change.

Requires the classical_ai branch to be merged (imports classical_ai.*).
"""
from __future__ import annotations

from ctf.actions import Action

from .controller import Controller


class ClassicalControllerAdapter(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.coordinator = None
        self.observation = None
        self.team = None
        self._roles: dict[str, str] = {}

    def reset(self) -> None:
        super().reset()
        self.coordinator = None          # a new game gets a fresh shared map

    def attach(self, observation, team, roles) -> None:
        from classical_ai.coordinator import ClassicalCoordinator

        self.observation, self.team, self._roles = observation, team, roles
        if self.coordinator is None:
            self.coordinator = ClassicalCoordinator(
                width=observation.width, height=observation.height, role_assignments={team: dict(roles)})
            self.coordinator.initialize_roles(team)
        self.coordinator.update_team_knowledge(observation, team)

    def actions(self, view) -> dict[str, Action]:
        from classical_ai.influence_map import build_influence_map
        from classical_ai.pathfinding import next_action

        obs, team = self.observation, self.team
        shared_map = self.coordinator.team_maps.for_team(team)
        influence = build_influence_map(obs, team, shared_map)
        role_controllers = self.coordinator.role_controllers[team]
        result: dict[str, Action] = {}
        for agent in view.agents:
            if not agent.active:
                continue
            if agent.id in self.orders:
                goal, _ = self._goal(view, agent, self.orders[agent.id])
                result[agent.id] = (Action.STAY if goal is None else next_action(
                    start=agent.pos, goal=goal, observation=obs, algorithm="astar", influence_map=influence))
            else:                                     # no order: Team 1's own role behaviour
                result[agent.id] = role_controllers[agent.role].act(agent_id=agent.id, observation=obs, team=team)
        return result
