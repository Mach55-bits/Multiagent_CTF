from dataclasses import replace
from random import Random

import pytest

from ctf import CTFEnvironment, ObservationMode, Team
from ctf.actions import Action
from rl.config import OpponentSpec
from rl.opponents import GuardMode, ScoutBehavior, ScriptedParams, ScriptedTeamStrategy, build_opponent
from rl.schema import ConfigError
from rl_helpers import OBSTACLES, small_env_config
from ctf import CTFConfig
from strategies import Role, StayStrategy

SCRIPTED = {
    "roles": ["attacker", "defender", "scout"],
    "guard_mode": "on_flag",
    "guard_leash": 0,
    "scout_behavior": "stay",
}


def _params(**kw) -> ScriptedParams:
    base = dict(
        roles=[Role.ATTACKER, Role.DEFENDER, Role.SCOUT],
        guard_mode=GuardMode.ON_FLAG,
        guard_leash=0,
        scout_behavior=ScoutBehavior.STAY,
    )
    base.update(kw)
    return ScriptedParams(**base)


def _play(env, strategies, max_steps=100):
    for _ in range(max_steps):
        if env.done:
            break
        acts = {}
        for team, strat in strategies.items():
            acts.update(strat.act(env.observe(ObservationMode.PARTIAL, team=team), team))
        env.step(acts)
    return env.state


def test_random_and_stay_return_valid_actions_for_active_agents_only() -> None:
    cfg = small_env_config()
    env = CTFEnvironment(cfg)
    env.reset()
    obs = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    for strat in (build_opponent(OpponentSpec("random", 1.0, {}), cfg, Random(0)), StayStrategy()):
        acts = strat.act(obs, Team.RED)
        assert set(acts) == {"R1", "R2", "R3"} and all(isinstance(a, Action) for a in acts.values())


def test_scripted_attacker_captures_against_an_undefended_flag() -> None:
    cfg = small_env_config(visibility_radius=10)
    env = CTFEnvironment(cfg)
    env.reset()
    red = ScriptedTeamStrategy(cfg, _params(), Random(0))
    state = _play(env, {Team.RED: red, Team.BLUE: StayStrategy()})
    # BLUE's stationary agents are off the flag cell (4,1), so the attacker wins
    assert state.winner is Team.RED


def test_parked_defender_on_flag_is_unbeatable_by_a_scripted_attacker() -> None:
    cfg = small_env_config(red_spawns=[(3, 0), (3, 1), (3, 2)], blue_spawns=[(4, 0), (4, 1), (4, 2)], visibility_radius=10)
    env = CTFEnvironment(cfg)
    env.reset()
    red = ScriptedTeamStrategy(cfg, _params(), Random(0))
    blue = ScriptedTeamStrategy(cfg, _params(roles=[Role.SCOUT, Role.DEFENDER, Role.SCOUT]), Random(1))
    state = _play(env, {Team.RED: red, Team.BLUE: blue})
    assert state.winner is None  # B2 holds (4,1): every pickup is tagged at once
    assert not state.agents["R1"].active


def test_leash_post_is_in_front_of_the_flag_at_the_requested_distance() -> None:
    cfg = CTFConfig(obstacles=list(OBSTACLES))
    strat = ScriptedTeamStrategy(cfg, _params(guard_mode=GuardMode.LEASH, guard_leash=2), Random(0))
    for team, base, other in ((Team.RED, cfg.red_base, cfg.blue_base), (Team.BLUE, cfg.blue_base, cfg.red_base)):
        post = strat._post[team]
        assert strat._nav.distance(base, post) == 2
    assert strat._post[Team.RED] != cfg.red_base


def test_defender_steps_toward_a_visible_enemy_carrier() -> None:
    from ctf import build_observation

    cfg = small_env_config(visibility_radius=10, blue_spawns=[(0, 0), (4, 2), (0, 2)])
    env = CTFEnvironment(cfg)
    state = env.reset()
    carrier = state.agents["R1"].with_position(2, 2).with_carrying_flag(True)
    state = replace(state, agents={**state.agents, "R1": carrier})
    obs = build_observation(state, ObservationMode.PARTIAL, team=Team.BLUE)
    blue = ScriptedTeamStrategy(cfg, _params(), Random(0))  # slot 1 (B2) is the defender
    actions = blue.act(obs, Team.BLUE)
    assert actions["B2"] is Action.LEFT  # (4,2) -> (3,2) shortens the path to (2,2)
    assert set(actions) == {"B1", "B2", "B3"}


def test_factory_validation() -> None:
    cfg = small_env_config()
    with pytest.raises(ConfigError, match="unknown opponent kind"):
        build_opponent(OpponentSpec("wizard", 1.0, {}), cfg, Random(0))
    with pytest.raises(ConfigError, match="takes no params"):
        build_opponent(OpponentSpec("random", 1.0, {"x": 1}), cfg, Random(0))
    with pytest.raises(ConfigError, match="missing"):
        build_opponent(OpponentSpec("scripted", 1.0, {"roles": ["attacker"] * 3}), cfg, Random(0))
    with pytest.raises(ConfigError, match="roles"):
        build_opponent(OpponentSpec("scripted", 1.0, {**SCRIPTED, "roles": ["attacker"]}), cfg, Random(0))
    assert isinstance(build_opponent(OpponentSpec("scripted", 1.0, SCRIPTED), cfg, Random(0)), ScriptedTeamStrategy)
