"""Grid helpers: shortest-path moves. (Team 1's A* can replace bfs_next_action later.)"""
from __future__ import annotations

from collections import deque

from ctf.actions import Action

Cell = tuple[int, int]
_MOVES = ((Action.UP, (0, -1)), (Action.DOWN, (0, 1)), (Action.LEFT, (-1, 0)), (Action.RIGHT, (1, 0)))


def in_bounds(width: int, height: int, c: Cell) -> bool:
    return 0 <= c[0] < width and 0 <= c[1] < height


def bfs_next_action(width: int, height: int, obstacles: set[Cell], start: Cell, goal: Cell) -> Action | None:
    """First move of a shortest path start->goal. STAY if already there, None if unreachable."""
    if start == goal:
        return Action.STAY
    if goal in obstacles or not in_bounds(width, height, goal):
        return None
    first: dict[Cell, Action] = {}
    queue = deque([start])
    seen = {start}
    while queue:
        cur = queue.popleft()
        for action, (dx, dy) in _MOVES:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in seen or nxt in obstacles or not in_bounds(width, height, nxt):
                continue
            seen.add(nxt)
            first[nxt] = first.get(cur, action) if cur != start else action
            if nxt == goal:
                return first[nxt]
            queue.append(nxt)
    return None


def reachable(width: int, height: int, obstacles: set[Cell], start: Cell, goal: Cell) -> bool:
    return bfs_next_action(width, height, obstacles, start, goal) is not None


def nearest_free(width: int, height: int, obstacles: set[Cell], p: Cell) -> Cell:
    """Snap a point to the closest open cell (used for zone names such as 'north')."""
    p = (min(max(p[0], 0), width - 1), min(max(p[1], 0), height - 1))
    queue, seen = deque([p]), {p}
    while queue:
        cur = queue.popleft()
        if cur not in obstacles:
            return cur
        for _, (dx, dy) in _MOVES:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt not in seen and in_bounds(width, height, nxt):
                seen.add(nxt)
                queue.append(nxt)
    return p
