from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from rl.config import LearnerSide, load_base_env_config
from rl.env_wrapper import CTFTrainEnv
from rl.features import ACTOR_CHANNELS
from rl_helpers import ROOT, rl_config


def _single_phase_cfg(**phase_overrides):
    cfg = rl_config()
    phase = replace(cfg.curriculum.phases[1], **phase_overrides)
    curriculum = replace(cfg.curriculum, phases=[phase])
    return replace(cfg, curriculum=curriculum)


def _run_episode(env: CTFTrainEnv, rng: np.random.Generator):
    step = env.reset()
    rewards = []
    n = 0
    while not step.done:
        step = env.step(rng.integers(0, 5, size=env.num_agents))
        rewards.append(step.rewards.copy())
        n += 1
        assert n <= 1000
    return step, np.array(rewards)


def test_every_phase_runs_full_episodes_with_consistent_shapes() -> None:
    cfg = rl_config()
    env = CTFTrainEnv(cfg, load_base_env_config(cfg, ROOT), seed=0)
    rng = np.random.default_rng(0)
    for phase in range(len(cfg.curriculum.phases)):
        env.set_phase(phase)
        for _ in range(5):
            step, rewards = _run_episode(env, rng)
            assert step.info["phase"] == cfg.curriculum.phases[phase].name
            assert step.info["outcome"] in {"win", "loss", "draw"}
            assert step.actor.grid.shape[0] == env.num_agents
            np.testing.assert_allclose(rewards.sum(axis=0), step.info["return"], atol=1e-9)
            comp_total = np.sum(list(step.info["components"].values()), axis=0)
            np.testing.assert_allclose(comp_total, step.info["return"], atol=1e-9)


def test_same_seed_and_actions_reproduce_the_same_episodes() -> None:
    cfg = rl_config()
    base = load_base_env_config(cfg, ROOT)
    outputs = []
    for _ in range(2):
        env = CTFTrainEnv(cfg, base, seed=123)
        env.set_phase(2)
        rng = np.random.default_rng(7)
        outputs.append([_run_episode(env, rng)[1].tolist() for _ in range(4)])
    assert outputs[0] == outputs[1]


def test_different_seeds_give_different_starting_layouts() -> None:
    cfg = rl_config()
    base = load_base_env_config(cfg, ROOT)
    firsts = []
    for seed in range(8):
        env = CTFTrainEnv(cfg, base, seed=seed)
        env.set_phase(2)
        firsts.append(env.reset().actor.grid.tobytes())
    assert len(set(firsts)) > 1  # spawns / side / opponent are seeded, so seeds differ


def test_nothing_leaks_between_episodes() -> None:
    cfg = _single_phase_cfg(fixed_spawn_fraction=1.0, opponent_noise=0.0)
    cfg = replace(cfg, curriculum=replace(cfg.curriculum, learner_side=LearnerSide.RED))
    env = CTFTrainEnv(cfg, load_base_env_config(cfg, ROOT), seed=0)
    first = env.reset()
    rng = np.random.default_rng(1)
    for _ in range(25):
        step = env.step(rng.integers(0, 5, size=env.num_agents))
        if step.done:
            break
    again = env.reset()
    np.testing.assert_array_equal(first.actor.grid, again.actor.grid)
    np.testing.assert_array_equal(first.actor.vec, again.actor.vec)
    observed = ACTOR_CHANNELS.index("observed")
    assert again.actor.grid[0, observed].sum() == first.actor.grid[0, observed].sum()


def test_phase_switch_applies_at_the_next_reset() -> None:
    cfg = rl_config()
    env = CTFTrainEnv(cfg, load_base_env_config(cfg, ROOT), seed=0)
    env.reset()
    assert env.phase_index == 0
    env.set_phase(2)
    assert env.phase_index == 0  # current episode unaffected
    env.reset()
    assert env.phase_index == 2
    with pytest.raises(IndexError):
        env.set_phase(99)


def test_learner_side_policy() -> None:
    cfg = rl_config()
    base = load_base_env_config(cfg, ROOT)
    teams = Counter()
    env = CTFTrainEnv(cfg, base, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        teams[_run_episode(env, rng)[0].info["learner_team"]] += 1
    assert set(teams) == {"RED", "BLUE"}
    fixed = replace(cfg, curriculum=replace(cfg.curriculum, learner_side=LearnerSide.BLUE))
    env = CTFTrainEnv(fixed, base, seed=0)
    assert {_run_episode(env, rng)[0].info["learner_team"] for _ in range(5)} == {"BLUE"}


def test_scout_scale_changes_scout_reward_only() -> None:
    cfg = _single_phase_cfg(fixed_spawn_fraction=1.0, opponent_noise=0.0)
    base = load_base_env_config(cfg, ROOT)
    totals = {}
    for scale in (1.0, 0.0):
        env = CTFTrainEnv(cfg, base, seed=5)
        env.set_scout_scale(scale)
        step, _ = _run_episode(env, np.random.default_rng(2))
        totals[scale] = step.info["components"]["scout"]
    assert sum(totals[0.0]) == 0.0
    assert sum(totals[1.0]) > 0.0
