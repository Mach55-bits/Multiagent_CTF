"""Visibility radius comes from config; eliminated agents see nothing."""

from ctf import (
    Agent,
    CTFConfig,
    CTFEnvironment,
    Flag,
    FlagStatus,
    GameState,
    ObservationMode,
    Team,
    build_observation,
)
from ctf.map import GridMap
from coordination.shared_map import SharedTeamMap, update_shared_map_from_partial_team
from strategies import BaseStrategy, Role, StayStrategy


def _state(agents: dict[str, Agent], radius: int = 3) -> GameState:
    return GameState(
        grid=GridMap(11, 7, frozenset()),
        agents=agents,
        flags={
            Team.RED: Flag(Team.RED, FlagStatus.AT_BASE, 1, 3),
            Team.BLUE: Flag(Team.BLUE, FlagStatus.AT_BASE, 9, 3),
        },
        red_base=(1, 3),
        blue_base=(9, 3),
        visibility_radius=radius,
    )


def test_env_observe_uses_configured_radius_by_default() -> None:
    env = CTFEnvironment(CTFConfig(visibility_radius=5))
    env.reset()
    assert env.observe(ObservationMode.PARTIAL, team=Team.RED).visibility_radius == 5
    assert env.observe(ObservationMode.GLOBAL).visibility_radius == 5
    assert env.observe(ObservationMode.PARTIAL, team=Team.RED, visibility_radius=2).visibility_radius == 2


def test_radius_changes_what_is_visible() -> None:
    agents = {"R1": Agent("R1", Team.RED, 4, 3), "B1": Agent("B1", Team.BLUE, 8, 3)}
    near = build_observation(_state(agents, radius=4), ObservationMode.PARTIAL, team=Team.RED)
    far = build_observation(_state(agents, radius=3), ObservationMode.PARTIAL, team=Team.RED)
    assert "B1" in near.agents and "B1" not in far.agents


def test_eliminated_agents_provide_no_vision() -> None:
    agents = {
        "R1": Agent("R1", Team.RED, 8, 3, active=False),  # dead, standing next to B1
        "R2": Agent("R2", Team.RED, 0, 0),
        "B1": Agent("B1", Team.BLUE, 9, 3),
    }
    obs = build_observation(_state(agents), ObservationMode.PARTIAL, team=Team.RED)
    assert "B1" not in obs.agents
    assert Team.BLUE not in obs.flags
    assert "R1" in obs.agents  # the team still knows its own fallen agent


def test_shared_map_ignores_eliminated_agents() -> None:
    agents = {
        "R1": Agent("R1", Team.RED, 8, 3, active=False),
        "R2": Agent("R2", Team.RED, 0, 0),
        "B1": Agent("B1", Team.BLUE, 20, 20),
    }
    obs = build_observation(_state(agents), ObservationMode.PARTIAL, team=Team.RED)
    team_map = SharedTeamMap(11, 7, Team.RED)
    update_shared_map_from_partial_team(team_map, obs)
    assert not team_map.observed[3][9]   # only the dead agent could have seen (9,3)
    assert team_map.observed[0][0]


def test_strategy_reset_is_an_optional_no_op() -> None:
    strat = StayStrategy()
    strat.reset()
    assert isinstance(strat, BaseStrategy)


def test_role_enum_is_string_compatible() -> None:
    assert Role("attacker") is Role.ATTACKER
    assert Role.SCOUT == "scout"


def test_legacy_radius_constant_alias_still_importable() -> None:
    from ctf import DEFAULT_PARTIAL_VISIBILITY_RADIUS, DEFAULT_VISIBILITY_RADIUS
    from ctf.observations import DEFAULT_PARTIAL_VISIBILITY_RADIUS as from_obs

    assert DEFAULT_PARTIAL_VISIBILITY_RADIUS == DEFAULT_VISIBILITY_RADIUS == from_obs
