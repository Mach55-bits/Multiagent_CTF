"""Seeded spawn randomization and episode reproducibility."""

from pathlib import Path

import pytest

from ctf import Action, CTFConfig, CTFEnvironment, Team
from ctf.config import SpawnMode, load_config

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
OBSTACLES = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]


def _cfg(**overrides) -> CTFConfig:
    return CTFConfig(obstacles=list(OBSTACLES), **overrides)


def _spawns(env: CTFEnvironment, team: Team) -> list[tuple[int, int]]:
    prefix = "R" if team is Team.RED else "B"
    st = env.state
    return [(a.x, a.y) for aid, a in sorted(st.agents.items()) if aid.startswith(prefix)]


def test_fixed_mode_is_unchanged_and_never_varies() -> None:
    env = CTFEnvironment(_cfg())
    env.reset()
    first = [_spawns(env, Team.RED)]
    for seed in range(5):
        env.reset(seed=seed)
        first.append(_spawns(env, Team.RED))
    assert all(s == first[0] for s in first)


def test_shuffle_keeps_cells_but_changes_assignment() -> None:
    cfg = _cfg(spawn_mode="shuffle")
    base = set(cfg.default_spawns(Team.RED))
    env = CTFEnvironment(cfg)
    seen = set()
    for seed in range(30):
        env.reset(seed=seed)
        spawns = _spawns(env, Team.RED)
        assert set(spawns) == base
        seen.add(tuple(spawns))
    assert len(seen) > 1


def test_near_base_respects_radius_home_zone_and_distinctness() -> None:
    cfg = _cfg(spawn_mode="near_base", spawn_radius=3, spawn_allow_base=False)
    env = CTFEnvironment(cfg)
    for seed in range(50):
        env.reset(seed=seed)
        for team, base, other in (
            (Team.RED, cfg.red_base, cfg.blue_base),
            (Team.BLUE, cfg.blue_base, cfg.red_base),
        ):
            spawns = _spawns(env, team)
            assert len(set(spawns)) == cfg.agents_per_team
            for (x, y) in spawns:
                d_own = abs(x - base[0]) + abs(y - base[1])
                d_other = abs(x - other[0]) + abs(y - other[1])
                assert d_own <= cfg.spawn_radius and d_own < d_other
                assert (x, y) != base and (x, y) not in OBSTACLES


def test_same_seed_same_spawns_different_seed_eventually_differs() -> None:
    cfg = _cfg(spawn_mode="near_base")
    a, b = CTFEnvironment(cfg), CTFEnvironment(cfg)
    a.reset(seed=123)
    b.reset(seed=123)
    assert a.state.agents == b.state.agents
    variants = set()
    for seed in range(20):
        a.reset(seed=seed)
        variants.add(tuple(sorted((k, v.x, v.y) for k, v in a.state.agents.items())))
    assert len(variants) > 1


def test_reset_without_seed_continues_stream_reproducibly() -> None:
    cfg = _cfg(spawn_mode="near_base", seed=7)

    def run() -> list:
        env = CTFEnvironment(cfg)
        out = []
        for _ in range(5):
            env.reset()
            out.append(tuple(sorted((k, v.x, v.y) for k, v in env.state.agents.items())))
        return out

    first, second = run(), run()
    assert first == second          # reproducible from config.seed
    assert len(set(first)) > 1      # but consecutive episodes differ


def test_step_is_deterministic_given_spawns_and_actions() -> None:
    cfg = _cfg(spawn_mode="near_base")
    actions = [Action.RIGHT, Action.UP, Action.STAY, Action.DOWN, Action.LEFT]

    def rollout() -> list:
        env = CTFEnvironment(cfg)
        env.reset(seed=99)
        snaps = []
        for a in actions:
            env.step({aid: a for aid in env.state.agents})
            snaps.append(dict(env.state.agents))
        return snaps

    assert rollout() == rollout()


def test_default_yaml_loads_and_stays_fixed() -> None:
    cfg = load_config(DEFAULT_YAML)
    assert cfg.spawn_mode is SpawnMode.FIXED
    env = CTFEnvironment(cfg)
    env.reset()
    assert _spawns(env, Team.RED) == [(1, 2), (1, 3), (1, 4)]


def test_unusable_near_base_config_fails_at_construction() -> None:
    with pytest.raises(ValueError):
        _cfg(spawn_mode="near_base", spawn_radius=1, spawn_allow_base=False, agents_per_team=5)
