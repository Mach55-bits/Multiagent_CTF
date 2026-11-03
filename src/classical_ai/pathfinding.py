import heapq
from collections import deque

from ctf.actions import Action
from ctf.observations import TeamObservation


Position = tuple[int, int]


def manhattan_distance(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(position: Position, observation: TeamObservation) -> list[Position]:
    x, y = position
    neighbors = []

    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        dx, dy = action.delta()
        nx, ny = x + dx, y + dy

        if (
            0 <= nx < observation.width
            and 0 <= ny < observation.height
            and observation.traversable[ny][nx]
        ):
            neighbors.append((nx, ny))

    return neighbors


def reconstruct_path(
    came_from: dict[Position, Position],
    current: Position,
) -> list[Position]:
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    return path[::-1]


def bfs_path(
    start: Position,
    goal: Position,
    observation: TeamObservation,
) -> list[Position] | None:
    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, observation):
            if neighbor not in came_from:
                came_from[neighbor] = current
                queue.append(neighbor)

    return None


def astar_path(
    start: Position,
    goal: Position,
    observation: TeamObservation,
    influence_map: list[list[float]] | None = None,
) -> list[Position] | None:
    frontier = [(manhattan_distance(start, goal), 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, cost, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, current)

        if cost > cost_so_far[current]:
            continue

        for neighbor in get_neighbors(current, observation):
            nx, ny = neighbor
            extra_cost = influence_map[ny][nx] if influence_map else 0
            new_cost = cost + 1 + extra_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan_distance(neighbor, goal)
                heapq.heappush(frontier, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    return None


def next_action(
    start: Position,
    goal: Position,
    observation: TeamObservation,
    algorithm: str = "astar",
    influence_map: list[list[float]] | None = None,
) -> Action:
    if start == goal:
        return Action.STAY

    if algorithm == "bfs":
        path = bfs_path(start, goal, observation)
    elif algorithm == "astar":
        path = astar_path(start, goal, observation, influence_map)
    else:
        raise ValueError(f"Unknown pathfinding algorithm: {algorithm}")

    if path is None or len(path) < 2:
        return Action.STAY

    dx = path[1][0] - start[0]
    dy = path[1][1] - start[1]

    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        if action.delta() == (dx, dy):
            return action

    return Action.STAY