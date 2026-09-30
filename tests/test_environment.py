"""Environment smoke and observation tests."""

from pathlib import Path

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import load_config
from ctf.environment import CTFEnvironment
from ctf.observations import ObservationMode, build_observation


def test_reset_creates_six_agents_by_default() -> None:
    env = CTFEnvironment(load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    state = env.reset()
    assert len(state.agents) == 6
    assert sum(1 for a in state.agents.values() if a.team is Team.RED) == 3
    assert sum(1 for a in state.agents.values() if a.team is Team.BLUE) == 3


def test_global_observation_contains_map_and_flags() -> None:
    env = CTFEnvironment(load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    state = env.reset()
    obs = build_observation(state, mode=ObservationMode.GLOBAL)
    assert obs.width == state.grid.width
    assert obs.height == state.grid.height
    assert Team.RED in obs.flags
    assert obs.timestep == 0


def test_step_returns_rewards_dict() -> None:
    env = CTFEnvironment(load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    env.reset()
    actions = {aid: Action.STAY for aid in env.config.agent_ids()}
    _, rewards, done, info = env.step(actions)
    assert set(rewards.keys()) == set(env.config.agent_ids())
    assert all(v == 0.0 for v in rewards.values())
    assert done is False
    assert "rejected_moves" in info
