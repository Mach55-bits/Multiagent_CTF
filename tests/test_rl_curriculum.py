from dataclasses import replace

from rl.curriculum import Curriculum
from rl_helpers import rl_config


def _curriculum(window: int = 4, advance: float = 0.75, max_steps: int = 1000) -> Curriculum:
    base = rl_config().curriculum.phases
    first = replace(base[0], window_episodes=window, advance_win_rate=advance, max_steps=max_steps)
    second = replace(base[1], window_episodes=window, advance_win_rate=advance, max_steps=max_steps)
    return Curriculum([first, second, replace(base[2], window_episodes=window)])


def test_advances_when_a_full_window_reaches_the_win_rate() -> None:
    c = _curriculum()
    assert not c.record(["win", "win", "win"], 30)  # window not full yet
    assert c.index == 0
    assert c.record(["win"], 10)  # 4/4 wins
    assert c.index == 1 and c.steps_in_phase == 0 and c.win_rate is None


def test_does_not_advance_below_the_threshold() -> None:
    c = _curriculum()
    assert not c.record(["win", "loss", "draw", "win"], 40)  # 0.5 < 0.75
    assert c.index == 0 and c.win_rate == 0.5


def test_rolling_window_forgets_old_episodes() -> None:
    c = _curriculum()
    c.record(["loss"] * 4, 40)
    assert c.index == 0
    assert c.record(["win"] * 4, 40)  # losses rolled out of the window


def test_advances_after_max_steps_even_with_a_poor_win_rate() -> None:
    c = _curriculum(max_steps=100)
    assert not c.record(["loss"], 60)
    assert c.record(["loss"], 60)
    assert c.index == 1


def test_last_phase_never_advances() -> None:
    c = _curriculum(max_steps=10)
    c.record(["win"] * 4, 100)
    c.record(["win"] * 4, 100)
    assert c.is_last and c.index == 2
    assert not c.record(["win"] * 50, 10**9) and c.index == 2


def test_state_round_trip_restores_progress() -> None:
    a = _curriculum()
    a.record(["win", "loss"], 50)
    state = a.state_dict()
    b = _curriculum()
    b.load_state_dict(state)
    assert (b.index, b.steps_in_phase, b.win_rate) == (a.index, a.steps_in_phase, a.win_rate)
    assert b.record(["win", "win"], 10) == a.record(["win", "win"], 10)
