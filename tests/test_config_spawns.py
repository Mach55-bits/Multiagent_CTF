"""Configuration spawn count and validation tests."""

import pytest

from ctf.agent import Team
from ctf.config import CTFConfig
from ctf.environment import CTFEnvironment


def test_agents_per_team_one() -> None:
    config = CTFConfig(
        width=5,
        height=5,
        agents_per_team=1,
        red_base=(0, 2),
        blue_base=(4, 2),
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    state = env.reset()
    assert len(state.agents) == 2
    assert "R1" in state.agents
    assert "B1" in state.agents
    assert len(config.default_spawns(Team.RED)) == 1


def test_agents_per_team_three_default_spawns() -> None:
    config = CTFConfig(
        width=7,
        height=5,
        agents_per_team=3,
        red_base=(1, 2),
        blue_base=(5, 2),
        tagging_enabled=False,
    )
    assert len(config.default_spawns(Team.RED)) == 3
    assert len(config.default_spawns(Team.BLUE)) == 3
    env = CTFEnvironment(config)
    state = env.reset()
    assert len(state.agents) == 6


def test_agents_per_team_four_generated_spawns() -> None:
    config = CTFConfig(
        width=7,
        height=5,
        agents_per_team=4,
        red_base=(1, 2),
        blue_base=(5, 2),
        tagging_enabled=False,
    )
    red = config.default_spawns(Team.RED)
    assert len(red) == 4
    assert red[0] == (1, 2)
    env = CTFEnvironment(config)
    state = env.reset()
    assert len([a for a in state.agents if a.startswith("R")]) == 4


def test_invalid_explicit_red_spawn_count() -> None:
    with pytest.raises(ValueError, match="red_spawns has 2 positions"):
        CTFConfig(
            width=5,
            height=5,
            agents_per_team=3,
            red_base=(0, 2),
            blue_base=(4, 2),
            red_spawns=[(0, 0), (0, 1)],
            blue_spawns=[(4, 0), (4, 1), (4, 2)],
            tagging_enabled=False,
        )


def test_spawn_on_obstacle_rejected() -> None:
    with pytest.raises(ValueError, match="red_spawn \\(1, 1\\) must not be on an obstacle"):
        CTFConfig(
            width=5,
            height=5,
            agents_per_team=1,
            red_base=(0, 2),
            blue_base=(4, 2),
            red_spawns=[(1, 1)],
            blue_spawns=[(3, 2)],
            obstacles=[(1, 1)],
            tagging_enabled=False,
        )


def test_base_on_obstacle_rejected() -> None:
    with pytest.raises(ValueError, match="red_base \\(2, 2\\) must not be on an obstacle"):
        CTFConfig(
            width=5,
            height=5,
            agents_per_team=1,
            red_base=(2, 2),
            blue_base=(4, 2),
            obstacles=[(2, 2)],
            tagging_enabled=False,
        )


def test_impossible_generated_spawns() -> None:
    with pytest.raises(ValueError, match="Not enough traversable cells near red_base"):
        CTFConfig(
            width=2,
            height=2,
            agents_per_team=5,
            red_base=(0, 0),
            blue_base=(1, 1),
            tagging_enabled=False,
        )


def test_invalid_grid_dimensions() -> None:
    with pytest.raises(ValueError, match="Grid width and height must be >= 1"):
        CTFConfig(width=0, height=5, tagging_enabled=False)
