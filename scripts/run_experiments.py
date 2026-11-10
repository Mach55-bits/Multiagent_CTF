#!/usr/bin/env python3
"""Experiment runner for the evaluation (skeleton, not implemented yet).

Planned design, following the evaluation section of the project proposal:
play every combination of the conditions below with the same maps and seeds, swap sides, and
write one CSV row per game. scripts/run_llm_commander.py already contains the single-game logic
(play()) to reuse.
"""
from __future__ import annotations

# TODO(2): conditions to sweep (at least 50 games per pairing, sides swapped for each pairing)
OPPONENTS = ["aggressive", "balanced", "defensive"]      # TODO(6): final definitions from the group
OBSERVABILITY = ["global", "partial"]
SIDES = ["red", "blue"]
GAMES_PER_PAIRING = 50

# TODO(4): comparison conditions to add to the sweep
#   - no commander (ScriptedStrategy("baseline"))
#   - random-orders commander (4a)
#   - validator on / off (4b)
#   - prompt with / without the map (4c)
#   - scout information sharing on / off (4d)
#   - trigger frequency: values of K for every_k_turns

# TODO(3): metrics to record per game
#   win/loss, flag captures, completion time (turns), movement efficiency (moves made),
#   decision time per LLM call, illegal-order rate, tokens / cost

# TODO(5): mixed-team configurations (LLM commanding Team 1 / Team 2 agents) once those exist

# TODO(7): demo scenario. Pick one seed/situation where the commander visibly changes the outcome,
# save its log, and also keep one failure case for the report.


def main() -> None:
    raise NotImplementedError("Experiment runner not implemented yet; see the TODO comments above.")


if __name__ == "__main__":
    main()
