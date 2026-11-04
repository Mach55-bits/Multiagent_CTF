from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import (
    GlobalObservation,
    PartialTeamObservation,
    TeamObservation,
)

from ctf.game_state import FlagStatus

from coordination.shared_map import (
    SharedTeamMap,
    build_scout_view,
    build_scout_view_from_partial,
)

from classical_ai.intents import Intent, IntentType
from classical_ai.pathfinding import astar_path, next_action
from classical_ai.influence_map import build_influence_map
from classical_ai.behavior_tree import decide_action


class ScoutRole:
    """
    Exploration role responsible for discovering enemy positions
    and updating the team's shared map.
    """

    def __init__(self, shared_map: SharedTeamMap):
        self.shared_map = shared_map

    def update_knowledge(
        self,
        agent_id: str,
        observation: TeamObservation,
        team: Team,
    ) -> None:

        # Build a scout observation without leaking hidden
        # enemy information when partial observations are used.
        if isinstance(observation, PartialTeamObservation):

            scout_view = build_scout_view_from_partial(
                observation,
                agent_id,
            )

        elif isinstance(observation, GlobalObservation):

            scout_view = build_scout_view(
                observation,
                team,
                agent_id,
                visibility_radius=3,
            )

        else:
            return

        # Merge newly discovered information into the team map.
        self.shared_map.update_from_scout(scout_view)

    def find_exploration_target(
        self,
        observation: TeamObservation,
        start: tuple[int, int],
        influence_map: list[list[float]],
    ) -> tuple[int, int] | None:

        width = observation.width
        height = observation.height

        frontier_cells = []

        # Find unexplored traversable cells adjacent to
        # already explored cells.
        for y in range(height):
            for x in range(width):

                if self.shared_map.observed[y][x]:
                    continue

                if not observation.traversable[y][x]:
                    continue

                for dx, dy in [
                    (0, -1),
                    (0, 1),
                    (-1, 0),
                    (1, 0),
                ]:

                    nx = x + dx
                    ny = y + dy

                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and self.shared_map.observed[ny][nx]
                        and observation.traversable[ny][nx]
                    ):
                        frontier_cells.append((x, y))
                        break

        # If there are no frontier cells, consider any
        # unexplored traversable cell.
        if not frontier_cells:

            for y in range(height):
                for x in range(width):

                    if (
                        not self.shared_map.observed[y][x]
                        and observation.traversable[y][x]
                    ):
                        frontier_cells.append((x, y))

        # Find the reachable frontier with the shortest path.
        best_target = None
        best_path_length = float("inf")

        for target in frontier_cells:

            path = astar_path(
                start=start,
                goal=target,
                observation=observation,
                influence_map=influence_map,
            )

            if path is not None and len(path) < best_path_length:

                best_path_length = len(path)
                best_target = target

        return best_target

    def act(
        self,
        agent_id: str,
        observation: TeamObservation,
        team: Team,
    ) -> Action:

        agent = observation.agents.get(agent_id)

        if agent is None or not agent.active or agent.team is not team:
            return Action.STAY

        # Update the team's knowledge first.
        self.update_knowledge(
            agent_id,
            observation,
            team,
        )

        # If the scout happens to carry the enemy flag,
        # prioritize returning it to its own base.
        if agent.carrying_flag:

            return decide_action(
                agent_id=agent_id,
                observation=observation,
                team=team,
                influence_map=build_influence_map(
                    observation,
                    team,
                    self.shared_map,
                ),
            )

        start = (agent.x, agent.y)

        influence = build_influence_map(
            observation=observation,
            team=team,
            shared_map=self.shared_map,
        )

        target = self.find_exploration_target(
            observation,
            start,
            influence,
        )

        if target is None:
            return Action.STAY

        intent = Intent(
            type=IntentType.SCOUT,
            target=target,
        )

        return next_action(
            start=start,
            goal=intent.target,
            observation=observation,
            algorithm="astar",
            influence_map=influence,
        )