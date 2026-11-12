from dataclasses import replace

import numpy as np
import pytest

from rl.config import load_base_env_config
from rl.features import ACTOR_VECTOR
from rl.vec_env import InProcessVecEnv, SubprocVecEnv, make_vec_env
from rl_helpers import ROOT, rl_config

FIELDS = ("grid", "vec", "mask", "alive", "role_index", "critic_grid", "critic_vec", "rewards", "dones")


def _cfg(workers: int):
    cfg = rl_config()
    return replace(cfg, rollout=replace(cfg.rollout, num_envs=4, num_workers=workers))


def _same(a, b) -> None:
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.infos == b.infos


@pytest.mark.parametrize("workers", [2, 3])  # 3 workers for 4 envs = uneven split (2, 1, 1)
def test_subprocess_matches_in_process_exactly(workers: int) -> None:
    base = load_base_env_config(_cfg(0), ROOT)
    reference = make_vec_env(_cfg(0), base, seed=5)
    parallel = make_vec_env(_cfg(workers), base, seed=5)
    assert isinstance(reference, InProcessVecEnv) and isinstance(parallel, SubprocVecEnv)
    try:
        for vec in (reference, parallel):
            vec.set_phase(2)
            vec.set_scout_scale(0.5)
        _same(reference.reset(), parallel.reset())
        rng = np.random.default_rng(0)
        finished = 0
        for _ in range(60):
            actions = rng.integers(0, 5, size=(4, 3))
            a, b = reference.step(actions), parallel.step(actions)
            _same(a, b)
            finished += len(a.infos)
        assert finished > 0  # auto-reset path was exercised
    finally:
        reference.close()
        parallel.close()


def test_auto_reset_returns_the_next_episodes_first_observation() -> None:
    base = load_base_env_config(_cfg(0), ROOT)
    vec = make_vec_env(_cfg(0), base, seed=1)
    vec.set_phase(0)
    first = vec.reset()
    rng = np.random.default_rng(0)
    step = first
    for _ in range(400):
        step = vec.step(rng.integers(0, 5, size=(4, 3)))
        if step.infos:
            done_env = int(np.argmax(step.dones))
            assert step.dones[done_env] == 1.0
            # a fresh episode starts at t=0 with every agent alive
            assert step.alive[done_env].all()
            time_fraction = step.vec[done_env, 0, ACTOR_VECTOR.index("time_fraction")]
            assert time_fraction == 0.0
            return
    pytest.fail("no episode finished in 400 steps")


def test_worker_crash_surfaces_the_traceback() -> None:
    base = load_base_env_config(_cfg(2), ROOT)
    vec = make_vec_env(_cfg(2), base, seed=0)
    try:
        vec.reset()
        with pytest.raises(RuntimeError, match="environment worker crashed"):
            vec.step(np.full((4, 3), 99))  # not a valid action index
    finally:
        vec.close()
