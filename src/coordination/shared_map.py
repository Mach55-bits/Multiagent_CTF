from __future__ import annotations

from dataclasses import dataclass

from ctf.agent import Team
from ctf.observations import GlobalObservation, PartialTeamObservation, manhattan_cells


@dataclass(frozen=True)
class ScoutObservation:
    """Local view contributed by a scout to the team's shared map."""

    timestep: int
    visible_cells: frozenset[tuple[int, int]]
    enemy_cells: frozenset[tuple[int, int]]


class SharedTeamMap:
    """Team-specific accumulated spatial knowledge (not part of the game engine)."""

    def __init__(self, width: int, height: int, team: Team) -> None:
        if width < 1 or height < 1:
            raise ValueError("width and height must be >= 1")
        self.width = width
        self.height = height
        self.team = team
        self.observed = [[False] * width for _ in range(height)]
        self.last_observed_timestep: list[list[int | None]] = [
            [None] * width for _ in range(height)
        ]
        self.enemy_probability = [[0.0] * width for _ in range(height)]

    def _validate_cell(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"cell ({x}, {y}) is outside map bounds")

    def update_from_scout(self, scout: ScoutObservation) -> None:
        """Merge a scout view into this team's shared map."""
        for x, y in scout.visible_cells:
            self._validate_cell(x, y)
        for x, y in scout.enemy_cells:
            self._validate_cell(x, y)
            if (x, y) not in scout.visible_cells:
                raise ValueError("enemy_cells must be a subset of visible_cells")

        for x, y in scout.visible_cells:
            self.observed[y][x] = True
            self.last_observed_timestep[y][x] = scout.timestep
            if (x, y) in scout.enemy_cells:
                self.enemy_probability[y][x] = 1.0
            else:
                self.enemy_probability[y][x] = 0.0

    def is_observed(self, x: int, y: int) -> bool:
        self._validate_cell(x, y)
        return self.observed[y][x]

    def enemy_probability_at(self, x: int, y: int) -> float:
        self._validate_cell(x, y)
        return self.enemy_probability[y][x]


class TeamSharedMaps:
    """Separate shared maps for RED and BLUE."""

    def __init__(self, width: int, height: int) -> None:
        self._maps = {
            Team.RED: SharedTeamMap(width, height, Team.RED),
            Team.BLUE: SharedTeamMap(width, height, Team.BLUE),
        }

    def for_team(self, team: Team) -> SharedTeamMap:
        return self._maps[team]


def build_scout_view(
    observation: GlobalObservation,
    team: Team,
    scout_agent_id: str,
    visibility_radius: int,
) -> ScoutObservation:
    """
    Build a scout observation from a global snapshot and scout position.
    """
    if visibility_radius < 0:
        raise ValueError("visibility_radius must be >= 0")
    scout = observation.agents.get(scout_agent_id)
    if scout is None:
        raise KeyError(f"unknown agent id: {scout_agent_id}")
    if scout.team is not team:
        raise ValueError(f"{scout_agent_id} is not on team {team.value}")

    visible = manhattan_cells(
        scout.x, scout.y, visibility_radius, observation.width, observation.height
    )
    enemy = frozenset(
        (agent.x, agent.y)
        for agent in observation.agents.values()
        if agent.team is team.opponent and agent.active and (agent.x, agent.y) in visible
    )
    return ScoutObservation(
        timestep=observation.timestep,
        visible_cells=visible,
        enemy_cells=enemy,
    )


def build_scout_view_from_partial(
    observation: PartialTeamObservation,
    scout_agent_id: str,
) -> ScoutObservation:
    """Build a scout view from a partial team observation (no hidden enemy leakage)."""
    scout = observation.agents.get(scout_agent_id)
    if scout is None:
        raise KeyError(f"unknown or non-visible agent id: {scout_agent_id}")
    if scout.team is not observation.team:
        raise ValueError(f"{scout_agent_id} is not on team {observation.team.value}")

    visible = manhattan_cells(
        scout.x,
        scout.y,
        observation.visibility_radius,
        observation.width,
        observation.height,
    )
    enemy = frozenset(
        (agent.x, agent.y)
        for agent in observation.agents.values()
        if agent.team is observation.team.opponent and (agent.x, agent.y) in visible
    )
    return ScoutObservation(
        timestep=observation.timestep,
        visible_cells=visible,
        enemy_cells=enemy,
    )


def update_shared_map_from_partial_team(
    team_map: SharedTeamMap,
    observation: PartialTeamObservation,
) -> None:
    """Merge each friendly agent's partial view into the team shared map."""
    for agent_id, agent in observation.agents.items():
        if agent.team is not observation.team:
            continue
        team_map.update_from_scout(build_scout_view_from_partial(observation, agent_id))
