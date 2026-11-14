#!/usr/bin/env python3
"""Experiment runner for the evaluation.

Plays every combination of  condition x opponent x observability  with the same seeds, swaps sides
(the commander plays RED and BLUE on every seed), and writes one CSV row per game plus a summary table.
Results are appended game by game, so an interrupted run (quota, network) resumes where it stopped.

Examples:
  uv run python scripts/run_experiments.py --dry-run
  uv run python scripts/run_experiments.py --llm mock --games 50
  uv run --with google-genai python scripts/run_experiments.py --llm gemini --games 50 \
      --conditions llm,classical,random_commander --opponents balanced,classical --cache logs/cache.json
  uv run python scripts/run_experiments.py --conditions llm@3,llm@5,llm@10 --opponents balanced   # trigger period
  uv run python scripts/run_experiments.py --conditions classical,rl,llm_classical,llm_rl --opponents classical,rl
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from strategies.llm_commander.conditions import build_condition, needs_rl, parse_condition, uses_llm  # noqa: E402
from strategies.llm_commander.llm import make_llm  # noqa: E402
from strategies.llm_commander.opponents import ALL_OPPONENTS, make_opponent  # noqa: E402
from strategies.llm_commander.reporting import read_rows, summarise  # noqa: E402
from strategies.llm_commander.rl_adapter import resolve_checkpoint  # noqa: E402
from strategies.llm_commander.runner import play_game, with_spawn_mode  # noqa: E402

DEFAULT_CONDITIONS = "llm,classical,random_commander,llm_no_validator,llm_no_map"
FIELDS = ["condition", "opponent", "observability", "side", "seed", "result", "winner", "turns",
          "my_flag_pickups", "my_flag_captures", "my_agents_tagged", "enemy_agents_tagged", "my_moves",
          "rejected_moves", "llm_decisions", "llm_requests", "orders", "illegal_orders", "unreadable_replies",
          "llm_seconds", "tokens_in", "tokens_out", "llm_model"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conditions", default=DEFAULT_CONDITIONS, help="comma-separated; see conditions.py")
    p.add_argument("--opponents", default="balanced,defensive,aggressive", help=f"from: {', '.join(ALL_OPPONENTS)}")
    p.add_argument("--observability", default="global", help="global, partial or global,partial")
    p.add_argument("--games", type=int, default=50, help="games per pairing (half as RED, half as BLUE)")
    p.add_argument("--seed", type=int, default=1000, help="first seed")
    p.add_argument("--spawn", default="near_base", choices=["fixed", "shuffle", "near_base"],
                   help="'fixed' makes every game identical; random spawns create variety")
    p.add_argument("--radius", type=int, default=None, help="visibility radius for partial observability (default: the environment's)")
    p.add_argument("--rl-checkpoint", default=None,
                   help="RL policy file from rl.train (default: $RL_CHECKPOINT or runs/mappo_default/checkpoints/latest.pt); "
                        "'untrained' = random weights, only to test the plumbing")
    p.add_argument("--k", type=int, default=5, help="default trigger period (ask the LLM at least every K turns)")
    p.add_argument("--llm", choices=["mock", "gemini"], default="mock")
    p.add_argument("--model", default=None, help="Gemini model name (default: GEMINI_MODEL or built-in)")
    p.add_argument("--rpm", type=float, default=10.0, help="max Gemini requests per minute (free tiers are low)")
    p.add_argument("--cache", default=None, help="LLM answer cache file (saves quota, repeatable)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default="results/games.csv")
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--log-dir", default="logs/experiments", help="per-game prompt/response logs (LLM conditions)")
    p.add_argument("--max-llm-calls", type=int, default=None, help="abort if the estimated calls exceed this")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the estimated LLM calls only")
    args = p.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    modes = [m.strip() for m in args.observability.split(",") if m.strip()]
    for c in conditions:
        parse_condition(c)
    for o in opponents:
        if o not in ALL_OPPONENTS:
            sys.exit(f"unknown opponent '{o}'")
    pairs = math.ceil(args.games / 2)

    pairings = []
    for cond in conditions:
        for opp in opponents:
            for mode in modes:
                if parse_condition(cond)[0] == "llm_no_scout_share" and mode == "global":
                    print(f"skipping {cond} in global mode (everything is visible, sharing has no effect)")
                    continue
                pairings.append((cond, opp, mode))
    total_games = len(pairings) * pairs * 2
    llm_games = sum(2 * pairs for c, _, _ in pairings if uses_llm(c) and parse_condition(c)[0] != "random_commander")
    est_calls = llm_games * (200 // args.k + 15)
    print(f"Plan: {len(pairings)} pairings x {pairs * 2} games = {total_games} games "
          f"({args.games} requested per pairing, rounded up to an even number so sides are balanced).")
    print(f"LLM backend: {args.llm}. Games that call the LLM: {llm_games}; estimated LLM requests: ~{est_calls}.")
    if args.llm != "mock" and est_calls:
        print(f"At --rpm {args.rpm:g} that is roughly {est_calls / args.rpm / 60:.1f} hours of waiting. "
              "Reduce --games / --conditions / --opponents, raise --k, or use --cache.")
    if args.dry_run:
        return
    if any(needs_rl(c) for c in conditions) or "rl" in opponents:
        try:
            args.rl_checkpoint = resolve_checkpoint(args.rl_checkpoint)
        except RuntimeError as e:
            sys.exit(str(e))
    if args.max_llm_calls is not None and args.llm != "mock" and est_calls > args.max_llm_calls:
        sys.exit(f"Estimated {est_calls} requests exceeds --max-llm-calls {args.max_llm_calls}. Reduce the plan.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["condition"], r["opponent"], r["observability"], r["side"], r["seed"]) for r in read_rows(out)}
    new_file = not out.exists()
    config = with_spawn_mode(load_config(ROOT / args.config), args.spawn)
    try:
        llm = make_llm(args.llm, args.model, args.rpm, args.cache) if any(uses_llm(c) for c in conditions) else None
    except RuntimeError as e:
        sys.exit(str(e))
    model_name = llm.name if llm else ""

    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        n = 0
        for cond, opp, mode in pairings:
            for i in range(pairs):
                for side in ("red", "blue"):
                    seed = args.seed + i
                    n += 1
                    key = (cond, opp, mode, side, str(seed))
                    if key in done:
                        continue
                    my_team = Team[side.upper()]
                    log = (f"{args.log_dir}/{cond}__{opp}__{mode}__{side}__{seed}.jsonl"
                           if uses_llm(cond) and args.log_dir else None)
                    if log and Path(log).exists():
                        Path(log).unlink()
                    mine = build_condition(cond, llm=llm, max_steps=config.max_episode_steps, seed=seed,
                                           default_k=args.k, log_path=log, rl_checkpoint=args.rl_checkpoint)
                    theirs = make_opponent(opp, config, seed, args.rl_checkpoint)
                    try:
                        g = play_game(config, mine, theirs, my_team, seed, partial=(mode == "partial"), radius=args.radius)
                    except RuntimeError as e:        # e.g. LLM quota exhausted: results so far are saved
                        f.flush()
                        sys.exit(f"\nStopped: {e}\nProgress is saved in {out}. Run the same command again to resume.")
                    writer.writerow({"condition": cond, "opponent": opp, "observability": mode, "side": side,
                                     "seed": seed, "result": g.result, "winner": g.winner, "turns": g.turns,
                                     "llm_model": model_name if uses_llm(cond) else "", **g.metrics})
                    f.flush()
                    print(f"[{n}/{total_games}] {cond:<18} vs {opp:<10} {mode:<7} {side:<4} seed {seed}: "
                          f"{g.result.upper()} in {g.turns} turns")
    
    rows = read_rows(out)
    summary = summarise(rows)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print("\n=== Summary (commander team) ===")
    print(f"{'condition':<20}{'opponent':<12}{'obs':<9}{'games':>6}{'win%':>7}{'95% CI':>12}{'turns':>7}{'illegal':>9}")
    for s in summary:
        print(f"{s['condition']:<20}{s['opponent']:<12}{s['observability']:<9}{s['games']:>6}"
              f"{s['win_rate'] * 100:>6.0f}%{s['win_rate_ci95']:>12}{s['avg_turns']:>7}{str(s['illegal_order_rate']):>9}")
    print(f"\nPer-game rows: {out}\nSummary table: {args.summary}")


if __name__ == "__main__":
    main()
