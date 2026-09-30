"""Grid map representation."""

from dataclasses import dataclass

from ctf.config import CTFConfig


@dataclass(frozen=True)
class GridMap:
    """2D grid with traversable cells and obstacles."""

    width: int
    height: int
    obstacles: frozenset[tuple[int, int]]

    @classmethod
    def from_config(cls, config: CTFConfig) -> "GridMap":
        return cls(
            width=config.width,
            height=config.height,
            obstacles=frozenset(config.obstacles),
        )

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_traversable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self.obstacles

    def traversable_grid(self) -> list[list[bool]]:
        """Return height x width matrix; True means traversable."""
        return [
            [
                self.is_traversable(x, y)
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]
