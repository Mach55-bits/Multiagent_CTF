import pytest

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import CTFConfig
from ctf.environment import CTFEnvironment
from ctf.observations import build_observation
from strategies.base import BaseStrategy
from strategies.dummy import StayStrategy


def test_stay_strategy_is_base_strategy() -> None:
    assert isinstance(StayStrategy(), BaseStrategy)


def test_stay_strategy_returns_stay_for_team_active_agents_only() -> None:
    config = CTFConfig(
        width=5,
        height=5,
        agents_per_team=2,
        red_spawns=[(0, 0), (0, 1)],
        blue_spawns=[(4, 4), (4, 3)],
        red_base=(0, 0),
        blue_base=(4, 4),
        tagging_enabled=False,
    )
    env = CTFEnvironment(config)
    state = env.reset()
    obs = build_observation(state)

    red_actions = StayStrategy().act(obs, Team.RED)
    assert set(red_actions.keys()) == {"R1", "R2"}
    assert all(action is Action.STAY for action in red_actions.values())

    blue_actions = StayStrategy().act(obs, Team.BLUE)
    assert set(blue_actions.keys()) == {"B1", "B2"}


def test_stay_strategy_omits_inactive_agents() -> None:
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
    state = env.reset()
    obs = build_observation(state)
    inactive = obs.agents["R1"].with_active(False)
    obs = type(obs)(
        mode=obs.mode,
        width=obs.width,
        height=obs.height,
        traversable=obs.traversable,
        obstacles=obs.obstacles,
        red_base=obs.red_base,
        blue_base=obs.blue_base,
        agents={**obs.agents, "R1": inactive},
        flags=obs.flags,
        timestep=obs.timestep,
        winner=obs.winner,
        terminated=obs.terminated,
        truncation=obs.truncation,
        tagging_enabled=obs.tagging_enabled,
    )
    assert StayStrategy().act(obs, Team.RED) == {}


def test_two_stay_strategies_merge_for_env_step() -> None:
    config = CTFConfig(agents_per_team=1, tagging_enabled=False)
    env = CTFEnvironment(config)
    env.reset()
    strategy = StayStrategy()
    obs = env.observe()
    actions = {
        **strategy.act(obs, Team.RED),
        **strategy.act(obs, Team.BLUE),
    }
    _, _, done, _ = env.step(actions)
    assert done is False
    assert env.state.timestep == 1


def test_base_strategy_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        BaseStrategy()  # type: ignore[abstract]
