from ctf.agent import Agent, Team
from ctf.game_state import Flag, FlagStatus
from ctf.observations import GlobalObservation, ObservationMode
from coordination.shared_map import (
    ScoutObservation,
    SharedTeamMap,
    TeamSharedMaps,
    build_scout_view,
)


def _global_obs(
    *,
    timestep: int = 0,
    agents: dict[str, Agent] | None = None,
) -> GlobalObservation:
    agents = agents or {
        "R1": Agent("R1", Team.RED, 2, 2),
        "B1": Agent("B1", Team.BLUE, 4, 2),
    }
    return GlobalObservation(
        mode=ObservationMode.GLOBAL,
        width=8,
        height=6,
        traversable=[[True] * 8 for _ in range(6)],
        obstacles=[],
        red_base=(0, 2),
        blue_base=(7, 2),
        agents=agents,
        flags={
            Team.RED: Flag(Team.RED, FlagStatus.AT_BASE, 0, 2),
            Team.BLUE: Flag(Team.BLUE, FlagStatus.AT_BASE, 7, 2),
        },
        timestep=timestep,
        winner=None,
        terminated=False,
        truncation=False,
        tagging_enabled=True,
    )


def test_shared_map_initialization() -> None:
    m = SharedTeamMap(4, 3, Team.RED)
    assert m.team is Team.RED
    assert not m.is_observed(0, 0)
    assert m.last_observed_timestep[0][0] is None
    assert m.enemy_probability_at(0, 0) == 0.0


def test_update_marks_observed_and_timestep() -> None:
    m = SharedTeamMap(5, 5, Team.RED)
    scout = ScoutObservation(
        timestep=7,
        visible_cells=frozenset({(1, 1), (2, 1)}),
        enemy_cells=frozenset(),
    )
    m.update_from_scout(scout)
    assert m.is_observed(1, 1)
    assert m.last_observed_timestep[1][1] == 7
    assert not m.is_observed(0, 0)
    assert m.enemy_probability_at(1, 1) == 0.0


def test_enemy_probability_in_visible_cells() -> None:
    m = SharedTeamMap(5, 5, Team.RED)
    scout = ScoutObservation(
        timestep=3,
        visible_cells=frozenset({(0, 0), (1, 0), (2, 0)}),
        enemy_cells=frozenset({(1, 0)}),
    )
    m.update_from_scout(scout)
    assert m.enemy_probability_at(1, 0) == 1.0
    assert m.enemy_probability_at(0, 0) == 0.0
    assert m.enemy_probability_at(2, 0) == 0.0


def test_later_scout_overwrites_timestamps() -> None:
    m = SharedTeamMap(3, 3, Team.BLUE)
    m.update_from_scout(
        ScoutObservation(1, frozenset({(0, 0)}), frozenset()),
    )
    m.update_from_scout(
        ScoutObservation(5, frozenset({(0, 0)}), frozenset({(0, 0)})),
    )
    assert m.last_observed_timestep[0][0] == 5
    assert m.enemy_probability_at(0, 0) == 1.0


def test_team_maps_are_separate() -> None:
    maps = TeamSharedMaps(6, 6)
    red = maps.for_team(Team.RED)
    blue = maps.for_team(Team.BLUE)
    red.update_from_scout(
        ScoutObservation(2, frozenset({(1, 1)}), frozenset()),
    )
    assert red.is_observed(1, 1)
    assert not blue.is_observed(1, 1)


def test_build_scout_view_from_global_observation() -> None:
    obs = _global_obs(timestep=4)
    scout_view = build_scout_view(obs, Team.RED, "R1", visibility_radius=2)
    assert scout_view.timestep == 4
    assert (2, 2) in scout_view.visible_cells
    assert (4, 2) in scout_view.enemy_cells

    maps = TeamSharedMaps(obs.width, obs.height)
    team_map = maps.for_team(Team.RED)
    team_map.update_from_scout(scout_view)
    assert team_map.is_observed(4, 2)
    assert team_map.enemy_probability_at(4, 2) == 1.0


def test_teammate_reads_same_shared_map() -> None:
    maps = TeamSharedMaps(8, 6)
    obs = _global_obs(
        agents={
            "R1": Agent("R1", Team.RED, 3, 2),
            "R2": Agent("R2", Team.RED, 0, 0),
            "B1": Agent("B1", Team.BLUE, 4, 2),
        }
    )
    scout_view = build_scout_view(obs, Team.RED, "R1", visibility_radius=1)
    shared = maps.for_team(Team.RED)
    shared.update_from_scout(scout_view)
    # Another red agent uses the same TeamSharedMaps instance.
    same = maps.for_team(Team.RED)
    assert same.is_observed(4, 2)
    assert same.enemy_probability_at(4, 2) == 1.0


def test_enemy_cells_must_be_visible() -> None:
    m = SharedTeamMap(3, 3, Team.RED)
    try:
        m.update_from_scout(
            ScoutObservation(1, frozenset({(0, 0)}), frozenset({(2, 2)})),
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
