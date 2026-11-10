"""Environment configuration.

``CTFConfig`` is the single source of truth for every environment parameter.
YAML files only *override* these defaults; unknown keys and wrongly typed
values are rejected so a typo can never silently change an experiment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ctf.agent import Team

Cell = tuple[int, int]

# Radius (Manhattan) used when no other value is configured.
DEFAULT_VISIBILITY_RADIUS = 3


class SpawnMode(str, Enum):
    """How agents are placed at the start of every episode.

    FIXED      use the configured (or generated) spawn list as-is.
    SHUFFLE    same spawn cells, but randomly re-assigned among the team's agents.
    NEAR_BASE  sample distinct cells within ``spawn_radius`` of the base, restricted
               to the team's home zone (cells closer to own base than to the enemy
               base). Ignores any explicit spawn list.
    """

    FIXED = "fixed"
    SHUFFLE = "shuffle"
    NEAR_BASE = "near_base"


@dataclass
class CTFConfig:
    width: int = 11
    height: int = 7
    agents_per_team: int = 3
    tagging_enabled: bool = True
    max_episode_steps: int = 200
    seed: int = 42
    visibility_radius: int = DEFAULT_VISIBILITY_RADIUS
    spawn_mode: SpawnMode = SpawnMode.FIXED
    spawn_radius: int = 3
    spawn_allow_base: bool = True
    obstacles: list[Cell] = field(default_factory=list)
    red_base: Cell = (1, 3)
    blue_base: Cell = (9, 3)
    red_spawns: list[Cell] | None = None
    blue_spawns: list[Cell] | None = None

    def __post_init__(self) -> None:
        self.spawn_mode = SpawnMode(self.spawn_mode)

        if self.width < 1 or self.height < 1:
            raise ValueError("Grid width and height must be >= 1")
        if self.agents_per_team < 1:
            raise ValueError("agents_per_team must be >= 1")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1")
        if self.visibility_radius < 0:
            raise ValueError("visibility_radius must be >= 0")
        if self.spawn_radius < 0:
            raise ValueError("spawn_radius must be >= 0")

        obstacle_set = set(self.obstacles)
        self._validate_cell("red_base", self.red_base)
        self._validate_cell("blue_base", self.blue_base)
        self._validate_not_obstacle("red_base", self.red_base, obstacle_set)
        self._validate_not_obstacle("blue_base", self.blue_base, obstacle_set)

        for obs in self.obstacles:
            self._validate_cell("obstacle", obs)

        for label, spawns in (("red", self.red_spawns), ("blue", self.blue_spawns)):
            if spawns is None:
                continue
            if len(spawns) != self.agents_per_team:
                raise ValueError(
                    f"{label}_spawns has {len(spawns)} positions but "
                    f"agents_per_team is {self.agents_per_team}"
                )
            for cell in spawns:
                self._validate_cell(f"{label}_spawn", cell)
                self._validate_not_obstacle(f"{label}_spawn", cell, obstacle_set)

        for team in (Team.RED, Team.BLUE):
            has_explicit = (self.red_spawns if team is Team.RED else self.blue_spawns) is not None
            if not has_explicit:
                self._validate_generated_spawns(team, obstacle_set)
            if self.spawn_mode is SpawnMode.NEAR_BASE:
                self._validate_near_base_pool(team)

    # ------------------------------------------------------------------ validation

    def _validate_generated_spawns(self, team: Team, obstacle_set: set[Cell]) -> None:
        base = self._base(team)
        available = self._traversable_cells_sorted_from(base, obstacle_set)
        if len(available) < self.agents_per_team:
            raise ValueError(
                f"Not enough traversable cells near {team.value.lower()}_base {base} to spawn "
                f"{self.agents_per_team} agents (only {len(available)} available)"
            )

    def _validate_near_base_pool(self, team: Team) -> None:
        pool = self._near_base_pool(team)
        if len(pool) < self.agents_per_team:
            raise ValueError(
                f"spawn_mode=near_base needs {self.agents_per_team} home-zone cells within "
                f"spawn_radius={self.spawn_radius} of {team.value.lower()}_base, "
                f"found {len(pool)} (spawn_allow_base={self.spawn_allow_base})"
            )

    def _validate_cell(self, name: str, cell: Cell) -> None:
        x, y = cell
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{name} {cell} is outside grid bounds")

    def _validate_not_obstacle(self, name: str, cell: Cell, obstacle_set: set[Cell]) -> None:
        if cell in obstacle_set:
            raise ValueError(f"{name} {cell} must not be on an obstacle")

    # ---------------------------------------------------------------------- spawns

    def _base(self, team: Team) -> Cell:
        return self.red_base if team is Team.RED else self.blue_base

    def _traversable_cells_sorted_from(self, origin: Cell, obstacle_set: set[Cell]) -> list[Cell]:
        ox, oy = origin
        ranked: list[tuple[int, int, int, Cell]] = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in obstacle_set:
                    continue
                ranked.append((abs(x - ox) + abs(y - oy), y, x, (x, y)))
        ranked.sort()
        return [cell for _, _, _, cell in ranked]

    def _near_base_pool(self, team: Team) -> list[Cell]:
        """Home-zone cells within ``spawn_radius`` of the team's base (deterministic order)."""
        own = self._base(team)
        other = self._base(team.opponent)
        obstacle_set = set(self.obstacles)
        pool: list[Cell] = []
        for y in range(self.height):
            for x in range(self.width):
                cell = (x, y)
                if cell in obstacle_set:
                    continue
                if not self.spawn_allow_base and cell == own:
                    continue
                d_own = abs(x - own[0]) + abs(y - own[1])
                d_other = abs(x - other[0]) + abs(y - other[1])
                if d_own <= self.spawn_radius and d_own < d_other:
                    pool.append(cell)
        return pool

    def default_spawns(self, team: Team) -> list[Cell]:
        """Deterministic spawns: explicit list if given, else the cells closest to base."""
        explicit = self.red_spawns if team is Team.RED else self.blue_spawns
        if explicit is not None:
            return list(explicit)
        cells = self._traversable_cells_sorted_from(self._base(team), set(self.obstacles))
        return cells[: self.agents_per_team]

    def sample_spawns(self, team: Team, rng: random.Random) -> list[Cell]:
        """Spawn cells for one episode, drawn from ``rng`` according to ``spawn_mode``."""
        if self.spawn_mode is SpawnMode.FIXED:
            return self.default_spawns(team)
        if self.spawn_mode is SpawnMode.SHUFFLE:
            cells = self.default_spawns(team)
            rng.shuffle(cells)
            return cells
        return rng.sample(self._near_base_pool(team), self.agents_per_team)

    def agent_ids(self) -> list[str]:
        ids = [f"R{i}" for i in range(1, self.agents_per_team + 1)]
        ids += [f"B{i}" for i in range(1, self.agents_per_team + 1)]
        return ids


# ------------------------------------------------------------------------ YAML <-> config

_CELL_FIELDS = {"red_base", "blue_base"}
_CELL_LIST_FIELDS = {"obstacles", "red_spawns", "blue_spawns"}


def _parse_cell(raw: Any, label: str) -> Cell:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{label} must be a pair [x, y]")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in raw):
        raise ValueError(f"{label} must contain integers, got {raw!r}")
    return int(raw[0]), int(raw[1])


def _check_scalar(name: str, value: Any, default: Any) -> Any:
    """Reject values whose type does not match the field's default type."""
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, str):  # also covers str-based enums
        ok = isinstance(value, str)
    else:  # pragma: no cover - guards future field types
        ok = True
    if not ok:
        raise ValueError(f"{name} must be {type(default).__name__}, got {value!r}")
    return value


def config_from_dict(data: dict[str, Any]) -> CTFConfig:
    known = {f.name: f for f in fields(CTFConfig)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"Unknown config key(s): {unknown}. Valid keys: {sorted(known)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        if name in _CELL_FIELDS:
            kwargs[name] = _parse_cell(value, name)
        elif name in _CELL_LIST_FIELDS:
            if value is None:
                kwargs[name] = None if name != "obstacles" else []
            else:
                kwargs[name] = [_parse_cell(c, name) for c in value]
        else:
            fld = known[name]
            kwargs[name] = _check_scalar(name, value, fld.default)
    return CTFConfig(**kwargs)


def config_to_dict(config: CTFConfig) -> dict[str, Any]:
    """YAML-serializable view of a config (round-trips through ``config_from_dict``)."""
    out: dict[str, Any] = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if f.name in _CELL_FIELDS:
            out[f.name] = list(value)
        elif f.name in _CELL_LIST_FIELDS:
            out[f.name] = None if value is None else [list(c) for c in value]
        elif isinstance(value, Enum):
            out[f.name] = value.value
        else:
            out[f.name] = value
    return out


def load_config(path: str | Path) -> CTFConfig:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return config_from_dict(data)
