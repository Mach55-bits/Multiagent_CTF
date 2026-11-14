#!/usr/bin/env python3
"""First-run check for the real LLM: tests the API key and model, then lets the model give orders
for the opening position of a game and prints everything it did.

  uv run --with google-genai python scripts/check_llm.py
  uv run --with google-genai python scripts/check_llm.py --model <model name from Google AI Studio>

Needs the GEMINI_API_KEY environment variable (key from https://aistudio.google.com).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from ctf.environment import CTFEnvironment  # noqa: E402
from strategies.llm_commander import LLMCommanderStrategy  # noqa: E402
from strategies.llm_commander.llm import make_llm  # noqa: E402
from strategies.llm_commander.prompt import build_system_prompt  # noqa: E402
from strategies.llm_commander.view import view_to_text  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", choices=["mock", "gemini"], default="gemini")
    p.add_argument("--model", default=None)
    p.add_argument("--rpm", type=float, default=10.0)
    args = p.parse_args()

    try:
        llm = make_llm(args.llm, args.model, args.rpm)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"1) Connection test with model '{llm.name}' ...")
    t0 = time.time()
    try:
        reply = llm.complete("Reply with JSON only.", 'Return exactly {"ok": true}')
    except RuntimeError as e:
        sys.exit(f"   FAILED: {e}")
    print(f"   reply: {reply.strip()[:200]!r}  ({time.time() - t0:.1f}s, usage {getattr(llm, 'last_usage', None)})")

    print("\n2) One real decision on the opening position of a game ...")
    config = load_config(ROOT / "configs/default.yaml")
    env = CTFEnvironment(config)
    env.reset(seed=1)
    strategy = LLMCommanderStrategy(llm, max_steps=config.max_episode_steps, log_path="logs/check_llm.jsonl")
    obs = env.observe()
    strategy.act(obs, Team.RED)
    st = strategy.commander.stats
    for line in strategy.timeline:
        print("   " + line)
    print(f"   requests {st['llm_requests']}, orders accepted {st['orders']}, illegal orders {st['illegal_orders']}, "
          f"unreadable replies {st['failed_replies']}, latency {st['latency_s']:.1f}s, tokens in/out "
          f"{st['tokens_in']}/{st['tokens_out']}")
    print("   full prompt and reply saved in logs/check_llm.jsonl")
    if st["orders"] == 0 and st["illegal_orders"] == 0 and st["failed_replies"] == 0:
        print("   (the model chose to give no orders; the default behaviours apply)")


if __name__ == "__main__":
    main()
