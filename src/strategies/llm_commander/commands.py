"""The command vocabulary: what the LLM may order, who may use what, and named zones."""
from __future__ import annotations

from dataclasses import dataclass, field

from ctf.agent import Team

from .pathing import Cell, nearest_free

# Commands the LLM may issue
COMMANDS = ("go_to", "capture_flag", "defend_base", "intercept", "scout")
NEEDS_TARGET = {"go_to", "intercept", "scout"}
ROLE_ALLOWED = {
    "attacker": {"go_to", "capture_flag", "intercept", "scout"},
    "defender": {"go_to", "defend_base", "intercept"},
    "scout": {"go_to", "scout", "intercept"},
}
ROLE_CYCLE = ("attacker", "defender", "scout")
# Used when an agent has no order. "hold" is internal (the LLM cannot order it).
ROLE_DEFAULT = {"attacker": "capture_flag", "defender": "defend_base", "scout": "hold"}
DEFAULT_ORDER_LIFETIME = 25  # turns before an order expires and the agent reverts to its default


def agent_index(agent_id: str) -> int:
    digits = "".join(ch for ch in agent_id if ch.isdigit())
    return int(digits) if digits else 0


def assign_roles(agent_ids: list[str]) -> dict[str, str]:
    """R1 = attacker, R2 = defender, R3 = scout (then repeats if more agents)."""
    ordered = sorted(agent_ids, key=agent_index)
    return {aid: ROLE_CYCLE[i % len(ROLE_CYCLE)] for i, aid in enumerate(ordered)}


@dataclass
class AgentView:
    id: str
    role: str
    pos: Cell
    carrying: bool
    active: bool


@dataclass
class FlagView:
    pos: Cell
    carried: bool
    carrier_id: str | None


@dataclass
class OrderStatus:
    agent_id: str
    command: str
    target: object
    status: str  # active | done | failed | blocked | expired


@dataclass
class View:
    """Everything the commander is allowed to know (built from the observation only)."""

    team: Team
    timestep: int
    max_steps: int | None
    width: int
    height: int
    obstacles: set[Cell]
    observed: set[Cell] | None  # None = whole map visible (global observability)
    own_base: Cell
    enemy_base: Cell
    own_flag: FlagView
    enemy_flag: FlagView | None
    agents: list[AgentView]
    enemies: list[tuple[str, Cell, int]]  # (id, last seen position, turn last seen)
    order_status: list[OrderStatus] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def agent(self, agent_id: str) -> AgentView | None:
        return next((a for a in self.agents if a.id == agent_id), None)


def zone_table(view: View) -> dict[str, Cell]:
    w, h, ob = view.width, view.height, view.obstacles
    raw = {
        "north": (w // 2, 0), "south": (w // 2, h - 1),
        "west": (0, h // 2), "east": (w - 1, h // 2),
        "northwest": (0, 0), "northeast": (w - 1, 0),
        "southwest": (0, h - 1), "southeast": (w - 1, h - 1),
        "center": (w // 2, h // 2),
        "own_base": view.own_base, "enemy_base": view.enemy_base,
    }
    return {k: nearest_free(w, h, ob, v) for k, v in raw.items()}


DYNAMIC_TARGETS = ("enemy_carrier",)  # resolved fresh every turn


def resolve_target(view: View, target) -> Cell | None:
    """Named zone / [x, y] -> cell. 'enemy_carrier' -> where our stolen flag is (if stolen)."""
    if isinstance(target, str):
        name = target.strip().lower()
        if name == "enemy_carrier":
            return view.own_flag.pos if view.own_flag.carried else None
        if name == "enemy_flag":
            return view.enemy_flag.pos if view.enemy_flag else view.enemy_base
        if name == "own_flag":
            return view.own_flag.pos
        return zone_table(view).get(name)
    if (isinstance(target, (list, tuple)) and len(target) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in target)):
        return (target[0], target[1])
    return None
