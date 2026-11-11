from dataclasses import replace

import numpy as np
import pytest

from ctf import CTFEnvironment, ObservationMode, Team
from ctf.actions import Action
from rl.navigation import GridNav
from rl.rewards import RewardCalculator
from rl_helpers import rl_config, small_env_config
from strategies.roles import Role

ROLES = [Role.ATTACKER, Role.DEFENDER, Role.SCOUT]
STAY = Action.STAY


def _zero_rewards(**keep):
    r = rl_config().rewards
    zeroed = {f: 0.0 for f in r.__dataclass_fields__ if f != "scout_anneal_steps"}
    zeroed.update(keep)
    return replace(r, **zeroed)


def _setup(rewards, env_config=None, gamma=1.0, team=Team.RED):
    cfg = env_config or small_env_config()
    env = CTFEnvironment(cfg)
    env.reset()
    nav = GridNav([[(x, y) not in set(cfg.obstacles) for x in range(cfg.width)] for y in range(cfg.height)])
    calc = RewardCalculator(cfg, team, ROLES, rewards, gamma, nav)
    glob = env.observe(ObservationMode.GLOBAL)
    calc.reset(glob)
    return env, calc, glob


def _advance(env, calc, prev, red_actions, blue_actions=None, scout_scale=1.0):
    acts = {aid: STAY for aid in ("R1", "R2", "R3")}
    acts.update(red_actions)
    for aid in ("B1", "B2", "B3"):
        acts[aid] = (blue_actions or {}).get(aid, STAY)
    _, _, _, info = env.step(acts)
    glob = env.observe(ObservationMode.GLOBAL)
    res = calc.step(glob, prev, info["events"], scout_scale)
    return glob, res, info


def test_potential_shaping_telescopes_when_undiscounted() -> None:
    env, calc, glob = _setup(_zero_rewards(attacker_potential=0.1))
    total = 0.0
    # R1 starts at (3,1); BLUE flag is at (4,1): one step right = pickup
    glob2, res, _ = _advance(env, calc, glob.agents, {"R1": Action.RIGHT})
    total += res.total[0]
    # distance 1 -> 0 gives +0.1 and carrying continuity adds nothing at the flag cell
    assert total == pytest.approx(0.1)


def test_potential_is_continuous_across_pickup_and_rewards_carrying_home() -> None:
    env, calc, glob = _setup(_zero_rewards(attacker_potential=0.1))
    prev = glob.agents
    total = 0.0
    for action in [Action.RIGHT, Action.LEFT, Action.LEFT, Action.LEFT, Action.LEFT]:
        glob, res, info = _advance(env, calc, prev, {"R1": action})
        prev = glob.agents
        total += res.total[0]
    assert glob.winner is Team.RED
    # undiscounted telescoping: from distance-1-to-flag to capture at own base
    # start potential = -0.1*1; end potential = 0 (terminal)  => total = +0.1
    assert total == pytest.approx(0.1)


def test_discounted_shaping_matches_formula() -> None:
    gamma = 0.9
    env, calc, glob = _setup(_zero_rewards(attacker_potential=0.1), gamma=gamma)
    phi0 = -0.1 * 1
    _, res, _ = _advance(env, calc, glob.agents, {"R1": Action.RIGHT})
    assert res.components["potential"][0] == pytest.approx(gamma * 0.0 - phi0)


def test_pickup_and_tag_events_reward_the_right_agents() -> None:
    rewards = _zero_rewards(pickup=2.0, tagged_while_carrying=-2.0, tag_enemy_carrier=3.0, tag_team_share=0.5)
    cfg = small_env_config(blue_spawns=[(4, 0), (0, 0), (0, 2)])
    env, calc, glob = _setup(rewards, cfg, team=Team.RED)
    # R1 grabs the flag at (4,1) while B1 steps onto the same cell -> pickup then tag
    _, res, info = _advance(env, calc, glob.agents, {"R1": Action.RIGHT}, {"B1": Action.DOWN})
    kinds = [e["type"] for e in info["events"]]
    assert "flag_pickup" in kinds and "tag" in kinds
    assert res.components["pickup"][0] == 2.0
    assert res.components["tagged"][0] == -2.0
    assert res.components["tag"].sum() == 0.0  # RED did not tag anyone
    # Same game from BLUE's point of view
    env2, calc2, glob2 = _setup(rewards, cfg, team=Team.BLUE)
    _, res2, _ = _advance(env2, calc2, glob2.agents, {"R1": Action.RIGHT}, {"B1": Action.DOWN})
    assert res2.components["tag"][0] == 3.0
    assert res2.components["team_share"].tolist() == [0.0, 0.5, 0.5]
    assert res2.components["pickup"].sum() == 0.0


def test_terminal_outcomes_apply_to_every_teammate() -> None:
    rewards = _zero_rewards(win=10.0, loss=-10.0, timeout=-1.0)
    env, calc, glob = _setup(rewards)
    prev = glob.agents
    for action in [Action.RIGHT, Action.LEFT, Action.LEFT, Action.LEFT, Action.LEFT]:
        glob, res, _ = _advance(env, calc, prev, {"R1": action})
        prev = glob.agents
    assert res.components["outcome"].tolist() == [10.0, 10.0, 10.0]
    env, calc, glob = _setup(rewards, team=Team.BLUE)
    prev = glob.agents
    for action in [Action.RIGHT, Action.LEFT, Action.LEFT, Action.LEFT, Action.LEFT]:
        glob, res, _ = _advance(env, calc, prev, {"R1": action})
        prev = glob.agents
    assert res.components["outcome"].tolist() == [-10.0, -10.0, -10.0]
    # timeout
    cfg = small_env_config(max_episode_steps=2)
    env, calc, glob = _setup(rewards, cfg)
    prev = glob.agents
    for _ in range(2):
        glob, res, _ = _advance(env, calc, prev, {})
        prev = glob.agents
    assert glob.truncation and res.components["outcome"].tolist() == [-1.0, -1.0, -1.0]


def test_scout_bonus_counts_new_cells_once_and_respects_scale() -> None:
    cfg = small_env_config(visibility_radius=1)
    env, calc, glob = _setup(_zero_rewards(scout_new_cell=1.0), cfg)
    prev = glob.agents
    # R3 (scout, slot 2) at (3,2); walking up reveals new cells, standing still does not
    glob, res, _ = _advance(env, calc, prev, {"R3": Action.UP})
    first = res.components["scout"][2]
    prev = glob.agents
    glob, res, _ = _advance(env, calc, prev, {"R3": Action.STAY})
    assert first >= 0 and res.components["scout"][2] == 0.0
    assert res.components["scout"][0] == 0.0 and res.components["scout"][1] == 0.0
    # scale 0 disables the bonus entirely
    env, calc, glob = _setup(_zero_rewards(scout_new_cell=1.0), cfg)
    _, res, _ = _advance(env, calc, glob.agents, {"R3": Action.UP}, scout_scale=0.0)
    assert res.components["scout"][2] == 0.0


def test_step_penalty_applies_to_active_agents_only() -> None:
    cfg = small_env_config(blue_spawns=[(4, 0), (0, 0), (0, 2)])
    env, calc, glob = _setup(_zero_rewards(step_penalty=-0.1), cfg)
    glob, res, _ = _advance(env, calc, glob.agents, {"R1": Action.RIGHT}, {"B1": Action.DOWN})
    assert res.components["step"].tolist() == [-0.1, -0.1, -0.1]
    assert not glob.agents["R1"].active  # tagged carrier is out
    glob, res, _ = _advance(env, calc, glob.agents, {})
    assert res.components["step"].tolist() == [0.0, -0.1, -0.1]
    assert isinstance(res.total, np.ndarray) and res.total.shape == (3,)
