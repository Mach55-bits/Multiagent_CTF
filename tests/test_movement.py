"""Movement and collision tests."""

import pytest

from ctf.actions import Action
from ctf.config import CTFConfig
from ctf.environment import CTFEnvironment


def test_agent_movement_right() -> None:
    config = CTFConfig(
        width=5,
        height=5,
        agents_per_team=1,
        red_spawns=[(1, 2)],
        blue_spawns=[(3, 2)],
        red_base=(0, 2),
        blue_base=(4, 2),
        max_episode_steps=50,
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    env.reset()
    env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    assert env.state.agents["R1"].x == 2
    assert env.state.agents["R1"].y == 2


def test_boundary_collision_rejects_move() -> None:
    config = CTFConfig(
        width=5,
        height=5,
        agents_per_team=1,
        red_spawns=[(0, 2)],
        blue_spawns=[(4, 2)],
        red_base=(0, 2),
        blue_base=(4, 2),
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    env.reset()
    _, _, _, info = env.step({"R1": Action.LEFT, "B1": Action.STAY})
    assert env.state.agents["R1"].x == 0
    assert info["rejected_moves"]["R1"] == "LEFT"


def test_obstacle_collision_rejects_move() -> None:
    config = CTFConfig(
        width=5,
        height=5,
        agents_per_team=1,
        red_spawns=[(1, 2)],
        blue_spawns=[(4, 2)],
        red_base=(0, 2),
        blue_base=(4, 2),
        obstacles=[(2, 2)],
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    env.reset()
    _, _, _, info = env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    assert env.state.agents["R1"].x == 1
    assert info["rejected_moves"]["R1"] == "RIGHT"


def test_stay_action_always_valid_for_active_agent() -> None:
    config = CTFConfig(
        width=5,
        height=5,
        agents_per_team=1,
        red_spawns=[(0, 0)],
        blue_spawns=[(4, 4)],
        red_base=(0, 0),
        blue_base=(4, 4),
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    env.reset()
    env.step({"R1": Action.STAY, "B1": Action.STAY})
    assert env.state.agents["R1"].x == 0
    assert env.state.agents["R1"].y == 0


def test_invalid_agent_id_raises() -> None:
    env = CTFEnvironment(CTFConfig(agents_per_team=1, tagging_enabled=False))
    env.reset()
    with pytest.raises(KeyError):
        env.step({"X9": Action.STAY, "R1": Action.STAY, "B1": Action.STAY})


def test_missing_action_for_active_agent_raises() -> None:
    env = CTFEnvironment(CTFConfig(agents_per_team=1, tagging_enabled=False))
    env.reset()
    with pytest.raises(KeyError):
        env.step({"R1": Action.STAY})
