"""Plays one full game and collects metrics. Shared by the single-game and experiment scripts."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ctf.agent import Team
from ctf.config import SpawnMode
from ctf.environment import CTFEnvironment
from ctf.observations import ObservationMode


@dataclass
class GameResult:
    result: str                 # "win" | "loss" | "draw" (from the commander team's point of view)
    winner: str | None
    turns: int
    my_team: str
    metrics: dict = field(default_factory=dict)
    game_events: list[str] = field(default_factory=list)


def with_spawn_mode(config, spawn_mode: str):
    """Copy of the config with another spawn mode ('fixed' gives identical starts in every game)."""
    return dataclasses.replace(config, spawn_mode=SpawnMode(spawn_mode))


def play_game(config, mine, theirs, my_team: Team, seed: int, partial: bool = False,
              radius: int | None = None) -> GameResult:
    from classical_ai.evaluation import StrategyEvaluator     # Team 1's evaluator (flags, tags, moves)

    env = CTFEnvironment(config)
    env.reset(seed=seed)
    for s in (mine, theirs):
        s.reset()
    other = my_team.opponent

    def observation_for(team: Team, strategy):
        """Global by default; partial when requested or when the strategy (the RL policy) needs it."""
        if partial or getattr(strategy, "needs_partial_observation", False):
            return env.observe(ObservationMode.PARTIAL, team=team, visibility_radius=radius)
        return env.observe(ObservationMode.GLOBAL, team=team)

    evaluator = StrategyEvaluator()
    team_of = {aid: a.team for aid, a in env.state.agents.items()}
    split = {"my_flag_pickups": 0, "my_flag_captures": 0, "my_agents_tagged": 0, "enemy_agents_tagged": 0}
    game_events: list[str] = []

    while not env.done:
        my_actions = mine.act(observation_for(my_team, mine), my_team)
        actions = {**my_actions, **theirs.act(observation_for(other, theirs), other)}
        _, _, _, info = env.step(actions)
        evaluator.record_step(my_actions, info, getattr(mine, "roles", None))
        for e in info["events"]:
            game_events.append(f"turn {env.state.timestep}: {e}")
            if e["type"] == "flag_pickup" and team_of[e["agent_id"]] is my_team:
                split["my_flag_pickups"] += 1
            elif e["type"] == "flag_capture" and team_of[e["agent_id"]] is my_team:
                split["my_flag_captures"] += 1
            elif e["type"] == "tag":
                split["my_agents_tagged" if team_of[e["agent_id"]] is my_team else "enemy_agents_tagged"] += 1

    state = env.state
    evaluator.record_episode(state)
    group = evaluator.get_results()                  # the group's standard fields
    result = "draw" if state.winner is None else ("win" if state.winner is my_team else "loss")
    metrics = {**split, "my_moves": sum(group["agent_moves"].values()),
               "rejected_moves": group["total_rejected_moves"]}
    commander = getattr(mine, "commander", None)
    if commander is not None:
        st = commander.stats
        metrics.update(llm_decisions=st["calls"], llm_requests=st["llm_requests"], orders=st["orders"],
                       illegal_orders=st["illegal_orders"], unreadable_replies=st["failed_replies"],
                       llm_seconds=round(st["latency_s"], 3), tokens_in=st["tokens_in"], tokens_out=st["tokens_out"])
    return GameResult(result, state.winner.value if state.winner else None, state.timestep,
                      my_team.value, metrics, game_events)
