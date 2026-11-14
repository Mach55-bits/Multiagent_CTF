"""Opponent teams for experiments, built from the other branches' code.

Names: aggressive | balanced | defensive | camper | random | stay   (RL branch scripted teams)
       classical                                                  (Team 1's classical AI)
       rl                                                         (Team 2's trained RL policy, needs a checkpoint)
       baseline                                                   (this package's role-default team)

TODO(6): "aggressive", "balanced" and "defensive" are provisional definitions (see PRESETS). Replace
them with the group's agreed ones. "camper" keeps its defender on its base cell, which makes that
base unbeatable (see TODO(8) in controller.py); it is kept as a named stress test.
"""
from __future__ import annotations

from random import Random

from strategies.base import BaseStrategy

# name -> (roles by slot, guard_mode, guard_leash, scout_behavior)
PRESETS = {
    "aggressive": (["attacker", "attacker", "attacker"], "leash", 2, "random"),
    "balanced":   (["attacker", "defender", "scout"], "leash", 2, "random"),
    "defensive":  (["attacker", "defender", "defender"], "leash", 2, "stay"),
    "camper":     (["attacker", "defender", "scout"], "on_flag", 0, "stay"),
}
SIMPLE_KINDS = ("random", "stay")
ALL_OPPONENTS = tuple(PRESETS) + SIMPLE_KINDS + ("classical", "rl", "baseline")


def make_opponent(name: str, env_config, seed: int = 0, rl_checkpoint: str | None = None) -> BaseStrategy:
    if name not in ALL_OPPONENTS:
        raise ValueError(f"unknown opponent '{name}'; choose from {', '.join(ALL_OPPONENTS)}")
    if name == "classical":
        from classical_ai.classical_strategy import ClassicalStrategy

        return ClassicalStrategy()
    if name == "rl":
        from .rl_adapter import RLPolicyStrategy

        return RLPolicyStrategy(rl_checkpoint)
    if name == "baseline":
        from .baseline import ScriptedStrategy

        return ScriptedStrategy("baseline")
    from rl.opponents import GuardMode, RandomStrategy, ScoutBehavior, ScriptedParams, ScriptedTeamStrategy
    from strategies import Role, StayStrategy

    rng = Random(seed)
    if name == "random":
        return RandomStrategy(rng)
    if name == "stay":
        return StayStrategy()
    roles, guard, leash, scout = PRESETS[name]
    params = ScriptedParams(roles=[Role(r) for r in roles], guard_mode=GuardMode(guard),
                            guard_leash=leash, scout_behavior=ScoutBehavior(scout))
    return ScriptedTeamStrategy(env_config, params, rng)
