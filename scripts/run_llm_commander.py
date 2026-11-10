#!/usr/bin/env python3
"""Play full Capture the Flag games with the LLM Commander controlling one team.

Examples (run from the repository root):
  uv run python scripts/run_llm_commander.py                       # offline mock LLM vs a balanced scripted team
  uv run --with anthropic python scripts/run_llm_commander.py --llm claude
  uv run python scripts/run_llm_commander.py --no-commander        # baseline: scripted team, no LLM
  uv run python scripts/run_llm_commander.py --partial --radius 3  # fog of war
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from ctf.environment import CTFEnvironment  # noqa: E402
from ctf.observations import ObservationMode  # noqa: E402
from strategies.llm_commander import (  # noqa: E402
    STYLES, CachedLLM, ClaudeLLM, LLMCommanderStrategy, MockLLM, ScriptedStrategy,
)


def play(args, game_no: int, llm):
    config = load_config(ROOT / args.config)
    env = CTFEnvironment(config)
    env.reset(seed=args.seed + game_no)
    my_team = Team[args.team.upper()]
    other = my_team.opponent
    log = Path(args.log).with_name(f"{Path(args.log).stem}_game{game_no}.jsonl") if args.log else None
    if log and log.exists():
        log.unlink()
    mine = (ScriptedStrategy("baseline") if args.no_commander else
            LLMCommanderStrategy(llm, every_k_turns=args.k, log_path=str(log) if log else None,
                                 max_steps=config.max_episode_steps))
    theirs = ScriptedStrategy(args.opponent)
    mode = ObservationMode.PARTIAL if args.partial else ObservationMode.GLOBAL
    events_log = []
    while not env.done:
        obs = {t: env.observe(mode, team=t, visibility_radius=args.radius) for t in (Team.RED, Team.BLUE)}
        actions = {**mine.act(obs[my_team], my_team), **theirs.act(obs[other], other)}
        _, _, _, info = env.step(actions)
        for e in info["events"]:
            events_log.append(f"turn {env.state.timestep}: {e}")
            # TODO(3): additional metrics. Count flag captures / pickups / tags per team here (from the
            # event types) and return them from play() with the movement counts from the strategy.
    return env, mine, events_log


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", choices=["mock", "claude"], default="mock")
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--team", choices=["red", "blue"], default="red", help="team the commander controls")
    p.add_argument("--opponent", choices=sorted(STYLES), default="balanced")
    p.add_argument("--no-commander", action="store_true", help="baseline: scripted role defaults, no LLM")
    p.add_argument("--partial", action="store_true", help="partial observability (fog of war)")
    p.add_argument("--radius", type=int, default=3, help="visibility radius for --partial")
    p.add_argument("--k", type=int, default=5, help="ask the LLM at least every K turns")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--log", default="logs/llm_commander.jsonl", help="prompt/response log (JSON lines)")
    p.add_argument("--cache", default=None, help="file to cache LLM answers (saves money, repeatable)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    llm = MockLLM() if args.llm == "mock" else ClaudeLLM(args.model)
    if args.cache:
        llm = CachedLLM(llm, args.cache)
    my_team = Team[args.team.upper()]
    wins = 0
    for g in range(args.games):
        env, mine, events = play(args, g, llm)
        s = env.state
        won = s.winner is my_team
        wins += won
        print(f"\n=== Game {g + 1}: commander={my_team.value} ({'baseline' if args.no_commander else llm.name}) "
              f"vs {my_team.opponent.value} ({args.opponent}) | {'partial' if args.partial else 'global'} observability ===")
        if not args.quiet:
            if not args.no_commander:
                for line in mine.timeline:
                    print("  " + line)
            for line in events:
                print("  GAME " + line)
        # TODO(2): write one row per game (result, turns, metrics, LLM stats) to a CSV file
        # instead of only printing it; the experiment runner (scripts/run_experiments.py) will reuse this.
        result = "WIN" if won else ("LOSS" if s.winner else "NO WINNER (turn limit)")
        print(f"Result: {result} after {s.timestep} turns")
        if not args.no_commander:
            st = mine.commander.stats
            n = st["llm_requests"] or 1
            print(f"LLM stats: {st['calls']} decisions, {st['llm_requests']} requests, "
                  f"{st['orders']} orders, {st['illegal_orders']} illegal orders rejected, "
                  f"{st['failed_replies']} unreadable replies, avg {st['latency_s'] / n:.2f}s/request, "
                  f"tokens in/out {st['tokens_in']}/{st['tokens_out']}")
    if args.games > 1:
        print(f"\nCommander won {wins} of {args.games} games.")
    if args.log and not args.no_commander:
        print(f"Prompt/response logs: {Path(args.log).parent}/ (one file per game)")


if __name__ == "__main__":
    main()
