"""Environment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ctf.agent import Team


@dataclass
class CTFConfig:
    width: int = 11
    height: int = 7
    agents_per_team: int = 3
    tagging_enabled: bool = True
    max_episode_steps: int = 200
    seed: int = 42
    obstacles: list[tuple[int, int]] = field(default_factory=list)
    red_base: tuple[int, int] = (1, 3)
    blue_base: tuple[int, int] = (9, 3)
    red_spawns: list[tuple[int, int]] | None = None
    blue_spawns: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("Grid width and height must be >= 1")
        if self.agents_per_team < 1:
            raise ValueError("agents_per_team must be >= 1")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1")

        obstacle_set = set(self.obstacles)
        self._validate_cell("red_base", self.red_base)
        self._validate_cell("blue_base", self.blue_base)
        self._validate_not_obstacle("red_base", self.red_base, obstacle_set)
        self._validate_not_obstacle("blue_base", self.blue_base, obstacle_set)

        for obs in self.obstacles:
            self._validate_cell("obstacle", obs)

        if self.red_spawns is not None:
            if len(self.red_spawns) != self.agents_per_team:
                raise ValueError(
                    f"red_spawns has {len(self.red_spawns)} positions but "
                    f"agents_per_team is {self.agents_per_team}"
                )
            for cell in self.red_spawns:
                self._validate_cell("red_spawn", cell)
                self._validate_not_obstacle("red_spawn", cell, obstacle_set)

        if self.blue_spawns is not None:
            if len(self.blue_spawns) != self.agents_per_team:
                raise ValueError(
                    f"blue_spawns has {len(self.blue_spawns)} positions but "
                    f"agents_per_team is {self.agents_per_team}"
                )
            for cell in self.blue_spawns:
                self._validate_cell("blue_spawn", cell)
                self._validate_not_obstacle("blue_spawn", cell, obstacle_set)

        for team in (Team.RED, Team.BLUE):
            auto = self.red_spawns is None if team is Team.RED else self.blue_spawns is None
            if auto:
                self._validate_generated_spawns(team, obstacle_set)

    def _validate_generated_spawns(self, team: Team, obstacle_set: set[tuple[int, int]]) -> None:
        base = self.red_base if team is Team.RED else self.blue_base
        label = "red" if team is Team.RED else "blue"
        available = self._traversable_cells_sorted_from(base, obstacle_set)
        if len(available) < self.agents_per_team:
            raise ValueError(
                f"Not enough traversable cells near {label}_base {base} to spawn "
                f"{self.agents_per_team} agents (only {len(available)} available)"
            )

    def _validate_cell(self, name: str, cell: tuple[int, int]) -> None:
        x, y = cell
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{name} {cell} is outside grid bounds")

    def _validate_not_obstacle(
        self,
        name: str,
        cell: tuple[int, int],
        obstacle_set: set[tuple[int, int]],
    ) -> None:
        if cell in obstacle_set:
            raise ValueError(f"{name} {cell} must not be on an obstacle")

    def _traversable_cells_sorted_from(
        self,
        origin: tuple[int, int],
        obstacle_set: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        ox, oy = origin
        ranked: list[tuple[int, int, int, tuple[int, int]]] = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in obstacle_set:
                    continue
                dist = abs(x - ox) + abs(y - oy)
                ranked.append((dist, y, x, (x, y)))
        ranked.sort()
        return [cell for _, _, _, cell in ranked]

    def default_spawns(self, team: Team) -> list[tuple[int, int]]:
        if team is Team.RED and self.red_spawns is not None:
            return list(self.red_spawns)
        if team is Team.BLUE and self.blue_spawns is not None:
            return list(self.blue_spawns)

        base = self.red_base if team is Team.RED else self.blue_base
        obstacle_set = set(self.obstacles)
        cells = self._traversable_cells_sorted_from(base, obstacle_set)
        return cells[: self.agents_per_team]

    def agent_ids(self) -> list[str]:
        ids: list[str] = []
        for i in range(1, self.agents_per_team + 1):
            ids.append(f"R{i}")
        for i in range(1, self.agents_per_team + 1):
            ids.append(f"B{i}")
        return ids


def _parse_cell(raw: Any, label: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{label} must be a pair [x, y]")
    return int(raw[0]), int(raw[1])


def config_from_dict(data: dict[str, Any]) -> CTFConfig:
    obstacles = [_parse_cell(c, "obstacle") for c in data.get("obstacles", [])]
    red_base = _parse_cell(data.get("red_base", [1, 3]), "red_base")
    blue_base = _parse_cell(data.get("blue_base", [9, 3]), "blue_base")
    red_spawns_raw = data.get("red_spawns")
    blue_spawns_raw = data.get("blue_spawns")
    red_spawns = (
        [_parse_cell(c, "red_spawn") for c in red_spawns_raw] if red_spawns_raw else None
    )
    blue_spawns = (
        [_parse_cell(c, "blue_spawn") for c in blue_spawns_raw] if blue_spawns_raw else None
    )
    return CTFConfig(
        width=int(data.get("width", 11)),
        height=int(data.get("height", 7)),
        agents_per_team=int(data.get("agents_per_team", 3)),
        tagging_enabled=bool(data.get("tagging_enabled", True)),
        max_episode_steps=int(data.get("max_episode_steps", 200)),
        seed=int(data.get("seed", 42)),
        obstacles=obstacles,
        red_base=red_base,
        blue_base=blue_base,
        red_spawns=red_spawns,
        blue_spawns=blue_spawns,
    )


def load_config(path: str | Path) -> CTFConfig:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return config_from_dict(data)
