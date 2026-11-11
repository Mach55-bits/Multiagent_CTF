from dataclasses import replace

import numpy as np
import pytest

from ctf import CTFConfig, CTFEnvironment, ObservationMode, Team, build_observation
from ctf.actions import Action
from rl.canonical import ACTION_INDEX, NUM_ACTIONS, Canonicalizer, assert_mirror_symmetric
from rl.features import ACTOR_CHANNELS, ACTOR_VECTOR, CRITIC_CHANNELS, FeatureBuilder
from rl.schema import ConfigError
from rl_helpers import OBSTACLES, rl_config
from strategies.roles import Role

ROLES = [Role.ATTACKER, Role.DEFENDER, Role.SCOUT]


def _cfg(**kw) -> CTFConfig:
    # Explicit mirror-symmetric spawns (auto-generated spawns break distance ties
    # differently per side, so they are not symmetric).
    base = dict(
        obstacles=list(OBSTACLES),
        visibility_radius=3,
        red_spawns=[(1, 2), (1, 3), (1, 4)],
        blue_spawns=[(9, 2), (9, 3), (9, 4)],
    )
    base.update(kw)
    return CTFConfig(**base)


def _builders(cfg: CTFConfig, mirror: bool = True) -> dict[Team, FeatureBuilder]:
    feats = replace(rl_config().features, mirror_blue=mirror)
    return {t: FeatureBuilder(cfg, t, ROLES, feats) for t in Team}


def _features(env: CTFEnvironment, builders, team: Team):
    return builders[team].actor_features(env.observe(ObservationMode.PARTIAL, team=team))


def test_shapes_match_channel_names() -> None:
    cfg = _cfg()
    env = CTFEnvironment(cfg)
    env.reset()
    b = _builders(cfg)
    f = _features(env, b, Team.RED)
    assert f.grid.shape == (3, len(ACTOR_CHANNELS), cfg.height, cfg.width)
    assert f.vec.shape == (3, len(ACTOR_VECTOR))
    assert f.mask.shape == (3, NUM_ACTIONS) and f.alive.all()
    c = b[Team.RED].critic_features(env.observe(ObservationMode.GLOBAL))
    assert c.grid.shape == (3, len(CRITIC_CHANNELS), cfg.height, cfg.width)
    assert f.grid.dtype == np.float32 and f.vec.dtype == np.float32


def test_mirroring_makes_blue_see_the_same_as_red_on_a_symmetric_game() -> None:
    cfg = _cfg()
    env = CTFEnvironment(cfg)
    env.reset()
    b = _builders(cfg)
    rng = np.random.default_rng(3)
    for step in range(12):
        red, blue = _features(env, b, Team.RED), _features(env, b, Team.BLUE)
        np.testing.assert_array_equal(red.grid, blue.grid, err_msg=f"grid differs at step {step}")
        np.testing.assert_array_equal(red.vec, blue.vec)
        np.testing.assert_array_equal(red.mask, blue.mask)
        # same canonical action for both teams => mirrored real moves
        idx = rng.integers(0, NUM_ACTIONS, size=3)
        acts = {}
        for team in Team:
            for slot, aid in enumerate(b[team].agent_ids):
                acts[aid] = b[team].canon.to_env_action(int(idx[slot]))
        _, _, done, info = env.step(acts)
        if info["events"] or done:
            break
    assert step >= 3


def test_without_mirroring_blue_is_distinguishable() -> None:
    cfg = _cfg()
    env = CTFEnvironment(cfg)
    env.reset()
    b = _builders(cfg, mirror=False)
    red, blue = _features(env, b, Team.RED), _features(env, b, Team.BLUE)
    assert not np.array_equal(red.grid, blue.grid)
    team_flag = ACTOR_VECTOR.index("team_flag")
    assert red.vec[0, team_flag] == 0.0 and blue.vec[0, team_flag] == 1.0


def test_mirrored_action_mapping_round_trips() -> None:
    blue, red = Canonicalizer(11, Team.BLUE, True), Canonicalizer(11, Team.RED, True)
    assert red.to_env_action(ACTION_INDEX[Action.LEFT]) is Action.LEFT
    assert blue.to_env_action(ACTION_INDEX[Action.LEFT]) is Action.RIGHT
    assert blue.to_env_action(ACTION_INDEX[Action.RIGHT]) is Action.LEFT
    assert blue.to_env_action(ACTION_INDEX[Action.UP]) is Action.UP
    assert blue.cell(9, 3) == (1, 3) and red.cell(9, 3) == (9, 3)


def test_action_mask_blocks_walls_and_edges_in_canonical_orientation() -> None:
    cfg = _cfg(red_spawns=[(4, 3), (0, 0), (1, 3)], blue_spawns=[(6, 3), (10, 6), (9, 3)])
    env = CTFEnvironment(cfg)
    env.reset()
    b = _builders(cfg)
    right, left, up = (ACTION_INDEX[a] for a in (Action.RIGHT, Action.LEFT, Action.UP))
    red = _features(env, b, Team.RED)
    assert not red.mask[0, right] and red.mask[0, left]  # wall at (5,3)
    assert not red.mask[1, up] and not red.mask[1, left]  # corner (0,0)
    blue = _features(env, b, Team.BLUE)
    # B1 at (6,3): env LEFT is blocked; canonically that is RIGHT
    assert not blue.mask[0, right] and blue.mask[0, left]
    assert red.mask[:, ACTION_INDEX[Action.STAY]].all()


def test_hidden_enemies_do_not_leak_into_actor_features() -> None:
    cfg = _cfg(red_spawns=[(1, 2), (1, 3), (1, 4)], blue_spawns=[(9, 2), (9, 3), (9, 4)])
    env = CTFEnvironment(cfg)
    state = env.reset()

    def moved(pos):
        agents = dict(state.agents)
        agents["B1"] = agents["B1"].with_position(*pos)
        return replace(state, agents=agents)

    feats = []
    for hidden_pos in [(9, 2), (10, 6), (8, 0)]:  # all far outside radius 3 of RED
        builder = _builders(cfg)[Team.RED]
        obs = build_observation(moved(hidden_pos), ObservationMode.PARTIAL, team=Team.RED)
        assert "B1" not in obs.agents
        feats.append(builder.actor_features(obs))
    for other in feats[1:]:
        np.testing.assert_array_equal(feats[0].grid, other.grid)
        np.testing.assert_array_equal(feats[0].vec, other.vec)
    assert feats[0].grid[0, ACTOR_CHANNELS.index("visible_enemies")].sum() == 0
    assert feats[0].grid[0, ACTOR_CHANNELS.index("enemy_flag_visible")].sum() == 0


def test_visible_enemy_and_belief_memory_appear_then_age() -> None:
    cfg = _cfg(red_spawns=[(1, 2), (1, 3), (1, 4)], blue_spawns=[(9, 2), (9, 3), (9, 4)])
    env = CTFEnvironment(cfg)
    state = env.reset()
    near = replace(state, agents={**state.agents, "B1": state.agents["B1"].with_position(3, 3)})
    builder = _builders(cfg)[Team.RED]
    f1 = builder.actor_features(build_observation(near, ObservationMode.PARTIAL, team=Team.RED))
    vis = ACTOR_CHANNELS.index("visible_enemies")
    belief = ACTOR_CHANNELS.index("enemy_belief")
    assert f1.grid[0, vis, 3, 3] == 1.0 and f1.grid[0, belief, 3, 3] == 1.0
    # enemy moves away; its last-known position stays in memory but is no longer "visible"
    gone = replace(
        near, timestep=near.timestep + 5,
        agents={**near.agents, "B1": near.agents["B1"].with_position(9, 3)},
    )
    f2 = builder.actor_features(build_observation(gone, ObservationMode.PARTIAL, team=Team.RED))
    assert f2.grid[0, vis].sum() == 0
    assert f2.grid[0, ACTOR_CHANNELS.index("observed")].sum() >= 1
    # reset forgets everything
    builder.reset()
    f3 = builder.actor_features(build_observation(state, ObservationMode.PARTIAL, team=Team.RED))
    assert f3.grid[0, belief].sum() == 0


def test_dead_agents_are_masked_to_stay_and_not_alive() -> None:
    cfg = _cfg()
    env = CTFEnvironment(cfg)
    state = env.reset()
    dead = replace(state, agents={**state.agents, "R2": state.agents["R2"].with_active(False)})
    f = _builders(cfg)[Team.RED].actor_features(
        build_observation(dead, ObservationMode.PARTIAL, team=Team.RED)
    )
    assert not f.alive[1] and f.alive[0] and f.alive[2]
    assert f.mask[1].sum() == 1 and f.mask[1, ACTION_INDEX[Action.STAY]]


def test_critic_sees_true_enemy_positions() -> None:
    cfg = _cfg()
    env = CTFEnvironment(cfg)
    env.reset()
    c = _builders(cfg)[Team.RED].critic_features(env.observe(ObservationMode.GLOBAL))
    assert c.grid[0, CRITIC_CHANNELS.index("enemy_agents")].sum() == 3


def test_mirror_requires_symmetric_map() -> None:
    assert_mirror_symmetric(_cfg())
    with pytest.raises(ConfigError, match="symmetric"):
        assert_mirror_symmetric(CTFConfig(obstacles=[(5, 1), (4, 2)]))
