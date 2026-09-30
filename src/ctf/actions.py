from enum import Enum


class Action(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"

    def delta(self) -> tuple[int, int]:
        if self is Action.UP:
            return (0, -1)
        if self is Action.DOWN:
            return (0, 1)
        if self is Action.LEFT:
            return (-1, 0)
        if self is Action.RIGHT:
            return (1, 0)
        if self is Action.STAY:
            return (0, 0)
        raise ValueError(f"Unknown action: {self}")
