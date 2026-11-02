from ctf.agent import Agent, Team
from ctf.config import CTFConfig
from ctf.environment import CTFEnvironment
from ctf.game_state import Flag, FlagStatus, GameState
from ctf.map import GridMap
from ctf.observations import ObservationMode, build_observation
from coordination.shared_map import (
    TeamSharedMaps,
    build_scout_view_from_partial,
    update_shared_map_from_partial_team,
)
from strategies.dummy import StayStrategy


def _state(
    *,
    red: list[tuple[int, int]],
    blue: list[tuple[int, int]],
    blue_flag_pos: tuple[int, int] = (9, 3),
) -> GameState:
    agents = {}
    for i, (x, y) in enumerate(red, start=1):
        agents[f"R{i}"] = Agent(f"R{i}", Team.RED, x, y)
    for i, (x, y) in enumerate(blue, start=1):
        agents[f"B{i}"] = Agent(f"B{i}", Team.BLUE, x, y)
    return GameState(
        grid=GridMap(11, 7, frozenset()),
        agents=agents,
        flags={
            Team.RED: Flag(Team.RED, FlagStatus.AT_BASE, 1, 3),
            Team.BLUE: Flag(Team.BLUE, FlagStatus.AT_BASE, blue_flag_pos[0], blue_flag_pos[1]),
        },
        red_base=(1, 3),
        blue_base=(9, 3),
    )


def test_global_mode_unchanged() -> None:
    env = CTFEnvironment(CTFConfig(tagging_enabled=False))
    env.reset()
    obs = env.observe(ObservationMode.GLOBAL)
    assert obs.mode is ObservationMode.GLOBAL
    assert len(obs.agents) == 6


def test_teammates_always_visible() -> None:
    state = _state(red=[(0, 0), (1, 3)], blue=[(10, 3)])
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    assert "R1" in partial.agents and "R2" in partial.agents
    assert partial.agents["R1"].x == 0 and partial.agents["R2"].x == 1


def test_enemy_visible_within_radius_three() -> None:
    state = _state(red=[(5, 3)], blue=[(8, 3)])
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    assert "B1" in partial.agents


def test_enemy_hidden_beyond_radius_three() -> None:
    state = _state(red=[(0, 3)], blue=[(10, 3)])
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    assert "B1" not in partial.agents


def test_enemy_visible_at_exactly_radius_three() -> None:
    state = _state(red=[(5, 3)], blue=[(8, 3)])
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    assert "B1" in partial.agents


def test_enemy_hidden_at_distance_four() -> None:
    state = _state(red=[(5, 3)], blue=[(9, 3)])
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    assert "B1" not in partial.agents


def test_own_flag_always_visible_enemy_flag_hidden_when_far() -> None:
    state = _state(red=[(0, 0)], blue=[(10, 0)], blue_flag_pos=(9, 3))
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    assert Team.RED in partial.flags
    assert Team.BLUE not in partial.flags


def test_enemy_flag_visible_when_within_radius() -> None:
    state = _state(red=[(6, 3)], blue=[(10, 3)], blue_flag_pos=(9, 3))
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    assert Team.BLUE in partial.flags
    assert partial.flags[Team.BLUE].x == 9


def test_no_leakage_in_to_dict() -> None:
    state = _state(red=[(0, 3)], blue=[(10, 3)])
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    assert "B1" not in partial.agents
    assert Team.BLUE not in partial.flags
    agent_payload = partial.to_dict()["agents"]
    assert "B1" not in agent_payload
    assert all(not aid.startswith("B") for aid in agent_payload)


def test_stay_strategy_with_partial_observation() -> None:
    state = _state(red=[(0, 3), (1, 3)], blue=[(10, 3)])
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    actions = StayStrategy().act(partial, Team.RED)
    assert set(actions.keys()) == {"R1", "R2"}


def test_partial_observe_via_environment() -> None:
    env = CTFEnvironment(CTFConfig(tagging_enabled=False))
    env.reset()
    partial = env.observe(ObservationMode.PARTIAL, team=Team.RED)
    assert partial.mode is ObservationMode.PARTIAL
    assert partial.team is Team.RED
    assert all(a.team is Team.RED for a in partial.agents.values() if a.agent_id.startswith("R"))


def test_shared_map_update_from_partial_without_hidden_enemy() -> None:
    state = _state(red=[(3, 3)], blue=[(10, 3)])
    partial = build_observation(state, ObservationMode.PARTIAL, team=Team.RED)
    maps = TeamSharedMaps(11, 7)
    update_shared_map_from_partial_team(maps.for_team(Team.RED), partial)
    scout = build_scout_view_from_partial(partial, "R1")
    assert (3, 3) in scout.visible_cells
    assert scout.enemy_cells == frozenset()
