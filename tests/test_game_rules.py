"""Flag, tagging, termination, and determinism tests."""

import pytest

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig
from ctf.environment import CTFEnvironment
from ctf.game_state import FlagStatus


def _minimal_config(**kwargs: object) -> CTFConfig:
    defaults = dict(
        width=7,
        height=5,
        agents_per_team=1,
        red_base=(0, 2),
        blue_base=(6, 2),
        red_spawns=[(0, 2)],
        blue_spawns=[(6, 2)],
        tagging_enabled=True,
        max_episode_steps=100,
        seed=7,
    )
    defaults.update(kwargs)
    return CTFConfig(**defaults)  # type: ignore[arg-type]


def test_flag_capture_when_entering_enemy_base() -> None:
    config = _minimal_config(tagging_enabled=False)
    env = CTFEnvironment(config)
    env.reset()
    # R1 walks to blue base at (6,2) from (0,2)
    for _ in range(6):
        env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    assert env.state.agents["R1"].carrying_flag is True
    assert env.state.flags[Team.BLUE].status is FlagStatus.CARRIED


def test_successful_flag_return_and_win() -> None:
    config = _minimal_config(tagging_enabled=False)
    env = CTFEnvironment(config)
    env.reset()
    for _ in range(6):
        env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    for _ in range(6):
        env.step({"R1": Action.LEFT, "B1": Action.STAY})
    assert env.state.winner is Team.RED
    assert env.done is True


def test_episode_terminates_after_capture_win() -> None:
    config = _minimal_config(tagging_enabled=False)
    env = CTFEnvironment(config)
    env.reset()
    for _ in range(12):
        _, _, done, _ = env.step({"R1": Action.RIGHT if _ < 6 else Action.LEFT, "B1": Action.STAY})
    assert done is True
    with pytest.raises(RuntimeError):
        env.step({"R1": Action.STAY, "B1": Action.STAY})


def test_tagging_eliminates_carrier() -> None:
    config = _minimal_config(
        red_spawns=[(3, 2)],
        blue_spawns=[(6, 2)],
        tagging_enabled=True,
    )
    env = CTFEnvironment(config)
    env.reset()
    env.step({"R1": Action.RIGHT, "B1": Action.UP})
    env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    env.step({"R1": Action.STAY, "B1": Action.DOWN})
    assert env.state.agents["R1"].active is False
    assert env.state.agents["R1"].carrying_flag is False


def test_flag_returns_to_base_after_tagging() -> None:
    config = _minimal_config(
        red_spawns=[(3, 2)],
        blue_spawns=[(6, 2)],
        tagging_enabled=True,
    )
    env = CTFEnvironment(config)
    env.reset()
    env.step({"R1": Action.RIGHT, "B1": Action.UP})
    env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    env.step({"R1": Action.RIGHT, "B1": Action.STAY})
    env.step({"R1": Action.STAY, "B1": Action.DOWN})
    flag = env.state.flags[Team.BLUE]
    assert flag.status is FlagStatus.AT_BASE
    assert (flag.x, flag.y) == config.blue_base


def test_max_episode_length_truncation() -> None:
    config = _minimal_config(max_episode_steps=3, tagging_enabled=False)
    env = CTFEnvironment(config)
    env.reset()
    for _ in range(3):
        _, _, done, _ = env.step({"R1": Action.STAY, "B1": Action.STAY})
    assert done is True
    assert env.state.winner is None
    assert env.state.truncation is True


def test_deterministic_with_same_seed() -> None:
    config = _minimal_config(seed=99, tagging_enabled=False)
    seq = [Action.RIGHT, Action.RIGHT, Action.LEFT, Action.STAY]

    def run() -> list[tuple[int, int]]:
        env = CTFEnvironment(config)
        env.reset(seed=99)
        positions: list[tuple[int, int]] = []
        for action in seq:
            env.step({"R1": action, "B1": Action.STAY})
            positions.append((env.state.agents["R1"].x, env.state.agents["R1"].y))
        return positions

    assert run() == run()
