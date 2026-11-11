import numpy as np

from ctf.actions import Action
from ctf.config import CTFConfig
from rl.navigation import GridNav
from rl_helpers import OBSTACLES


def _default_nav() -> GridNav:
    cfg = CTFConfig(obstacles=list(OBSTACLES))
    return GridNav([[(x, y) not in set(cfg.obstacles) for x in range(cfg.width)] for y in range(cfg.height)])


def test_distance_goes_around_the_wall_not_through_it() -> None:
    nav = _default_nav()
    # (1,3) -> (1,0): 3, along row 0 to (9,0): 8, down to (9,3): 3
    assert nav.distance((1, 3), (9, 3)) == 14
    assert nav.distance((1, 3), (9, 3)) > abs(1 - 9) + abs(3 - 3)  # Manhattan would say 8


def test_distance_is_symmetric_zero_on_diagonal_and_unreachable_for_walls() -> None:
    nav = _default_nav()
    for a, b in [((0, 0), (10, 6)), ((2, 3), (7, 1))]:
        assert nav.distance(a, b) == nav.distance(b, a)
    assert nav.distance((4, 4), (4, 4)) == 0
    assert nav.distance((5, 3), (4, 3)) == nav.unreachable  # (5,3) is a wall
    assert nav.max_distance >= 14


def test_distance_map_matches_pairwise_distance() -> None:
    nav = _default_nav()
    dmap = nav.distance_map((9, 3))
    assert dmap.shape == (nav.height, nav.width)
    assert dmap[3, 1] == nav.distance((1, 3), (9, 3))


def test_step_toward_always_shortens_the_path_and_reaches_the_goal() -> None:
    nav = _default_nav()
    pos, goal = (1, 3), (9, 3)
    steps = 0
    while pos != goal:
        before = nav.distance(pos, goal)
        action = nav.step_toward(pos, goal)
        assert action is not Action.STAY
        dx, dy = action.delta()
        pos = (pos[0] + dx, pos[1] + dy)
        assert nav.distance(pos, goal) == before - 1
        steps += 1
    assert steps == 14
    assert nav.step_toward(goal, goal) is Action.STAY


def test_random_tie_breaking_stays_on_shortest_paths() -> None:
    import random

    nav = _default_nav()
    rng = random.Random(0)
    seen = set()
    for _ in range(40):
        a = nav.step_toward((2, 0), (8, 2), rng)
        seen.add(a)
        dx, dy = a.delta()
        assert nav.distance((2 + dx, 0 + dy), (8, 2)) < nav.distance((2, 0), (8, 2))
    assert len(seen) >= 1
    assert isinstance(nav.distance_map((0, 0)), np.ndarray)
