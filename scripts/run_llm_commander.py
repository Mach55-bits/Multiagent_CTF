#!/usr/bin/env python3
"""Play full Capture the Flag games with a commander (or a baseline) controlling one team.

Examples (run from the repository root):
  uv run python scripts/run_llm_commander.py                            # offline mock LLM vs 'balanced'
  uv run --with google-genai python scripts/run_llm_commander.py --llm gemini --opponent classical
  uv run python scripts/run_llm_commander.py --condition classical      # Team 1's classical AI, no LLM
  uv run python scripts/run_llm_commander.py --condition llm_classical  # LLM orders on Team 1's agents
  uv run python scripts/run_llm_commander.py --condition llm_rl --rl-checkpoint untrained   # LLM + RL agent
  uv run python scripts/run_llm_commander.py --partial --radius 3       # fog of war
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from strategies.llm_commander.conditions import ALL_CONDITIONS, build_condition, needs_rl, parse_condition, uses_llm  # noqa: E402
from strategies.llm_commander.llm import make_llm  # noqa: E402
from strategies.llm_commander.opponents import ALL_OPPONENTS, make_opponent  # noqa: E402
from strategies.llm_commander.rl_adapter import resolve_checkpoint  # noqa: E402
from strategies.llm_commander.runner import play_game, with_spawn_mode  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", default="llm", help=f"one of: {', '.join(ALL_CONDITIONS)} (add @K to set the trigger period)")
    p.add_argument("--llm", choices=["mock", "gemini"], default="mock")
    p.add_argument("--model", default=None, help="Gemini model name (default: GEMINI_MODEL or built-in)")
    p.add_argument("--rpm", type=float, default=10.0, help="max Gemini requests per minute")
    p.add_argument("--team", choices=["red", "blue"], default="red", help="team the commander controls")
    p.add_argument("--opponent", choices=ALL_OPPONENTS, default="balanced")
    p.add_argument("--partial", action="store_true", help="partial observability (fog of war)")
    p.add_argument("--radius", type=int, default=None, help="visibility radius for partial observations (default: the environment's)")
    p.add_argument("--rl-checkpoint", default=None,
                   help="RL policy file from rl.train (default: $RL_CHECKPOINT or runs/mappo_default/checkpoints/latest.pt); "
                        "'untrained' = random weights, only to test the plumbing")
    p.add_argument("--k", type=int, default=5, help="ask the LLM at least every K turns")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spawn", default="fixed", choices=["fixed", "shuffle", "near_base"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--log", default="logs/llm_commander.jsonl", help="prompt/response log (one file per game)")
    p.add_argument("--cache", default=None, help="file to cache LLM answers (saves quota, repeatable)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    parse_condition(args.condition)
    if needs_rl(args.condition) or args.opponent == "rl":
        try:
            args.rl_checkpoint = resolve_checkpoint(args.rl_checkpoint)
        except RuntimeError as e:
            sys.exit(str(e))
    config = with_spawn_mode(load_config(ROOT / args.config), args.spawn)
    try:
        llm = make_llm(args.llm, args.model, args.rpm, args.cache) if uses_llm(args.condition) else None
    except RuntimeError as e:
        sys.exit(str(e))
    my_team = Team[args.team.upper()]
    wins = 0
    for g in range(args.games):
        seed = args.seed + g
        log = None
        if uses_llm(args.condition) and args.log:
            log = str(Path(args.log).with_name(f"{Path(args.log).stem}_game{g}.jsonl"))
            Path(log).unlink(missing_ok=True)
        mine = build_condition(args.condition, llm=llm, max_steps=config.max_episode_steps, seed=seed,
                               default_k=args.k, log_path=log, rl_checkpoint=args.rl_checkpoint)
        theirs = make_opponent(args.opponent, config, seed, args.rl_checkpoint)
        res = play_game(config, mine, theirs, my_team, seed, partial=args.partial, radius=args.radius)
        wins += res.result == "win"
        backend = llm.name if llm else "no LLM"
        print(f"\n=== Game {g + 1}: {args.condition} ({backend}) as {my_team.value} vs {args.opponent} | "
              f"{'partial' if args.partial else 'global'} observability | seed {seed} ===")
        if not args.quiet:
            for line in getattr(mine, "timeline", []):
                print("  " + line)
            for line in res.game_events:
                print("  GAME " + line)
        print(f"Result: {res.result.upper()} after {res.turns} turns")
        m = res.metrics
        print(f"Flag pickups {m['my_flag_pickups']}, own agents tagged {m['my_agents_tagged']}, "
              f"enemy agents tagged {m['enemy_agents_tagged']}, moves {m['my_moves']}")
        if "llm_requests" in m:
            n = m["llm_requests"] or 1
            print(f"LLM: {m['llm_decisions']} decisions, {m['llm_requests']} requests, {m['orders']} orders, "
                  f"{m['illegal_orders']} illegal rejected, {m['unreadable_replies']} unreadable, "
                  f"avg {m['llm_seconds'] / n:.2f}s/request, tokens in/out {m['tokens_in']}/{m['tokens_out']}")
    if args.games > 1:
        print(f"\nWon {wins} of {args.games} games.")
    if uses_llm(args.condition) and args.log:
        print(f"Prompt/response logs: {Path(args.log).parent}/")


if __name__ == "__main__":
    main()
