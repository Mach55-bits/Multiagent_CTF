#!/usr/bin/env python3
"""Check that the RL agent works with the commander: plays the RL policy alone and as part of a mixed team.

  uv run --extra rl python scripts/check_rl.py                       # uses runs/mappo_default/checkpoints/latest.pt
  uv run --extra rl python scripts/check_rl.py --rl-checkpoint untrained     # random weights: tests the plumbing only
Needs the rl_agent branch merged and the RL extras installed (uv sync --extra dev --extra rl).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from strategies.llm_commander.conditions import build_condition  # noqa: E402
from strategies.llm_commander.llm import MockLLM  # noqa: E402
from strategies.llm_commander.opponents import make_opponent  # noqa: E402
from strategies.llm_commander.rl_adapter import resolve_checkpoint  # noqa: E402
from strategies.llm_commander.runner import play_game  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rl-checkpoint", default=None)
    p.add_argument("--opponent", default="classical")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    try:
        ckpt = resolve_checkpoint(args.rl_checkpoint)
    except RuntimeError as e:
        sys.exit(str(e))
    config = load_config(ROOT / "configs/default.yaml")
    print(f"RL checkpoint: {ckpt}\n")
    for cond in ("rl", "llm_rl"):
        for side in (Team.RED, Team.BLUE):
            mine = build_condition(cond, llm=MockLLM(), max_steps=config.max_episode_steps, seed=args.seed, rl_checkpoint=ckpt)
            theirs = make_opponent(args.opponent, config, args.seed, ckpt)
            res = play_game(config, mine, theirs, side, args.seed)
            m = res.metrics
            print(f"{cond:<7} as {side.value:<4} vs {args.opponent}: {res.result.upper():<5} in {res.turns:>3} turns | "
                  f"pickups {m['my_flag_pickups']}, tagged {m['my_agents_tagged']}, moves {m['my_moves']}")
    print("\nOK: the RL policy ran alone ('rl') and as part of a mixed team ('llm_rl'). "
          "The mixed team uses the offline mock LLM here; add --llm gemini in the other scripts for the real one.")


if __name__ == "__main__":
    main()
