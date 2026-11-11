"""Team-relative ("canonical") view of the board.

With mirroring on, BLUE sees the grid flipped left-right so every agent sees its own
base on the left. One policy can then play both sides, and training data doubles.
Network outputs are mapped back to environment actions here.
"""

from __future__ import annotations

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig
from rl.schema import ConfigError

#: index of each action in the network's output layer
ACTIONS: tuple[Action, ...] = tuple(Action)
ACTION_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)
_LEFT, _RIGHT = ACTION_INDEX[Action.LEFT], ACTION_INDEX[Action.RIGHT]


def assert_mirror_symmetric(config: CTFConfig) -> None:
    """Mirroring BLUE is only valid if the map is left-right symmetric."""
    w = config.width
    mirrored_obstacles = {(w - 1 - x, y) for x, y in config.obstacles}
    mirrored_base = (w - 1 - config.red_base[0], config.red_base[1])
    if mirrored_obstacles != set(config.obstacles) or mirrored_base != tuple(config.blue_base):
        raise ConfigError(
            "features.mirror_blue=true needs a left-right symmetric map "
            "(mirrored obstacles and bases must coincide); set mirror_blue: false otherwise"
        )


class Canonicalizer:
    def __init__(self, width: int, team: Team, mirror: bool) -> None:
        self.width = width
        self.team = team
        self.flip = mirror and team is Team.BLUE
        #: constant fed to the network so it can tell sides apart when NOT mirroring
        self.team_flag = 0.0 if mirror else (1.0 if team is Team.BLUE else 0.0)

    def cell(self, x: int, y: int) -> tuple[int, int]:
        return (self.width - 1 - x, y) if self.flip else (x, y)

    def to_env_action(self, index: int) -> Action:
        if self.flip and index in (_LEFT, _RIGHT):
            index = _RIGHT if index == _LEFT else _LEFT
        return ACTIONS[index]

    def mask_to_canonical(self, env_mask: list[bool]) -> list[bool]:
        out = list(env_mask)
        if self.flip:
            out[_LEFT], out[_RIGHT] = out[_RIGHT], out[_LEFT]
        return out
