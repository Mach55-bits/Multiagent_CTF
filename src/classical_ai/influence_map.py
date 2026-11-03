from ctf.agent import Team
from ctf.observations import TeamObservation
from coordination.shared_map import SharedTeamMap


def build_influence_map(
    observation: TeamObservation,
    team: Team,
    shared_map: SharedTeamMap | None = None,
    enemy_radius: int = 3,
    enemy_penalty: float = 5.0,
    probability_weight: float = 3.0,
) -> list[list[float]]:

    width = observation.width
    height = observation.height

    influence = [
        [0.0 for _ in range(width)]
        for _ in range(height)
    ]

    enemies = [
        agent
        for agent in observation.agents.values()
        if agent.team is team.opponent and agent.active
    ]

    for enemy in enemies:
        for y in range(height):
            for x in range(width):

                distance = abs(enemy.x - x) + abs(enemy.y - y)

                if distance <= enemy_radius:
                    penalty = enemy_penalty * (
                        (enemy_radius + 1 - distance)
                        / (enemy_radius + 1)
                    )

                    influence[y][x] = max(
                        influence[y][x],
                        penalty
                    )

    if shared_map is not None:

        if shared_map.width != width or shared_map.height != height:
            raise ValueError("Shared map dimensions do not match observation")

        if shared_map.team is not team:
            raise ValueError("Shared map belongs to a different team")

        for y in range(height):
            for x in range(width):

                probability = shared_map.enemy_probability_at(x, y)

                influence[y][x] += probability_weight * probability

    return influence
