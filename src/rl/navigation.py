"""Shortest-path distances on the grid (static walls only).

Used for reward shaping and by scripted opponents. Manhattan distance is wrong on
this map (the wall forces detours through the top/bottom gaps), so true BFS path
distances are precomputed for every cell pair once per map.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from random import Random

import numpy as np

from ctf.actions import Action

Cell = tuple[int, int]
MOVES: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


class GridNav:
    def __init__(self, traversable: Sequence[Sequence[bool]]) -> None:
        self.height = len(traversable)
        self.width = len(traversable[0])
        self._free = np.array(traversable, dtype=bool)
        n = self.height * self.width
        #: value used for "no path"; larger than any real shortest-path length
        self.unreachable = n
        self._dist = np.full((n, n), self.unreachable, dtype=np.int32)
        for y in range(self.height):
            for x in range(self.width):
                if self._free[y, x]:
                    self._bfs_from((x, y))

    def _index(self, cell: Cell) -> int:
        return cell[1] * self.width + cell[0]

    def is_free(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and bool(self._free[y, x])

    def _bfs_from(self, source: Cell) -> None:
        row = self._dist[self._index(source)]
        row[self._index(source)] = 0
        queue: deque[Cell] = deque([source])
        while queue:
            x, y = queue.popleft()
            d = row[self._index((x, y))]
            for move in MOVES:
                dx, dy = move.delta()
                nx, ny = x + dx, y + dy
                if self.is_free(nx, ny) and row[self._index((nx, ny))] == self.unreachable:
                    row[self._index((nx, ny))] = d + 1
                    queue.append((nx, ny))

    def distance(self, a: Cell, b: Cell) -> int:
        return int(self._dist[self._index(a), self._index(b)])

    def distance_map(self, target: Cell) -> np.ndarray:
        """(H, W) array of path distances from every cell to ``target``."""
        return self._dist[self._index(target)].reshape(self.height, self.width)

    @property
    def max_distance(self) -> int:
        reachable = self._dist[self._dist < self.unreachable]
        return int(reachable.max()) if reachable.size else 0

    def step_toward(self, src: Cell, dst: Cell, rng: Random | None = None) -> Action:
        """Move that shortens the path to ``dst``; STAY if already there or unreachable."""
        if src == dst or self.distance(src, dst) >= self.unreachable:
            return Action.STAY
        here = self.distance(src, dst)
        best: list[Action] = []
        for move in MOVES:
            dx, dy = move.delta()
            nxt = (src[0] + dx, src[1] + dy)
            if self.is_free(*nxt) and self.distance(nxt, dst) < here:
                best.append(move)
        if not best:
            return Action.STAY
        return rng.choice(best) if rng is not None else best[0]
