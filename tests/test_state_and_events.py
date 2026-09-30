"""State snapshot isolation and step info events."""

import pytest

from ctf.actions import Action
from ctf.agent import Agent, Team
from ctf.config import CTFConfig
from ctf.environment import CTFEnvironment


def test_env_state_snapshot_not_mutable() -> None:
    config = CTFConfig(
        width=5,
        height=5,
        agents_per_team=1,
        red_spawns=[(1, 2)],
        blue_spawns=[(3, 2)],
        red_base=(0, 2),
        blue_base=(4, 2),
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    env.reset()
    snap = env.state
    snap.terminated = True
    snap.agents["R1"] = Agent("R1", Team.RED, 9, 9, active=False)
    assert env.state.terminated is False
    assert env.state.agents["R1"].x == 1


def test_flag_pickup_event() -> None:
    config = CTFConfig(
        width=7,
        height=5,
        agents_per_team=1,
        red_spawns=[(0, 2)],
        blue_spawns=[(6, 2)],
        red_base=(0, 2),
        blue_base=(6, 2),
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    env.reset()
    for _ in range(6):
        _, _, _, info = env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    pickups = [e for e in info["events"] if e["type"] == "flag_pickup"]
    assert len(pickups) == 1
    assert pickups[0]["agent_id"] == "R1"
    assert pickups[0]["flag_team"] == "BLUE"


def test_tag_event() -> None:
    config = CTFConfig(
        width=7,
        height=5,
        agents_per_team=1,
        red_spawns=[(3, 2)],
        blue_spawns=[(6, 2)],
        red_base=(0, 2),
        blue_base=(6, 2),
        tagging_enabled=True,
    )
    env = CTFEnvironment(config)
    env.reset()
    env.step({"R1": Action.RIGHT, "B1": Action.UP})
    env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    _, _, _, info = env.step({"R1": Action.STAY, "B1": Action.DOWN})
    tags = [e for e in info["events"] if e["type"] == "tag"]
    assert len(tags) == 1
    assert tags[0]["agent_id"] == "R1"
    assert tags[0]["by_agent_id"] == "B1"
    assert tags[0]["flag_team"] == "BLUE"


def test_flag_capture_and_episode_end_events() -> None:
    config = CTFConfig(
        width=7,
        height=5,
        agents_per_team=1,
        red_spawns=[(0, 2)],
        blue_spawns=[(6, 2)],
        red_base=(0, 2),
        blue_base=(6, 2),
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    env.reset()
    info = {"events": []}
    for _ in range(6):
        _, _, _, info = env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    for _ in range(6):
        _, _, done, info = env.step({"R1": Action.LEFT, "B1": Action.STAY})
    assert done is True
    captures = [e for e in info["events"] if e["type"] == "flag_capture"]
    ends = [e for e in info["events"] if e["type"] == "episode_end"]
    assert len(captures) == 1
    assert captures[0]["agent_id"] == "R1"
    assert len(ends) == 1
    assert ends[0]["winner"] == "RED"
    assert ends[0]["reason"] == "flag_capture"


def test_max_steps_episode_end_event() -> None:
    config = CTFConfig(
        width=5,
        height=5,
        agents_per_team=1,
        red_spawns=[(0, 0)],
        blue_spawns=[(4, 4)],
        red_base=(0, 0),
        blue_base=(4, 4),
        max_episode_steps=2,
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    env.reset()
    _, _, _, info = env.step({"R1": Action.STAY, "B1": Action.STAY})
    _, _, done, info = env.step({"R1": Action.STAY, "B1": Action.STAY})
    assert done is True
    ends = [e for e in info["events"] if e["type"] == "episode_end"]
    assert len(ends) == 1
    assert ends[0]["winner"] is None
    assert ends[0]["reason"] == "max_steps"


def test_pickup_before_tag_prevents_score_same_step() -> None:
    """Pickup and tag on same step: pickup + tag fire; no flag_capture."""
    config = CTFConfig(
        width=7,
        height=5,
        agents_per_team=1,
        red_spawns=[(5, 2)],
        blue_spawns=[(6, 2)],
        red_base=(0, 2),
        blue_base=(6, 2),
        tagging_enabled=True,
    )
    env = CTFEnvironment(config)
    env.reset()
    _, _, _, info = env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    types = [e["type"] for e in info["events"]]
    assert types == ["flag_pickup", "tag"]
    assert "flag_capture" not in types
    assert env.state.winner is None
