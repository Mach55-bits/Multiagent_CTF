#!/usr/bin/env python3
"""Run a deterministic full Capture the Flag episode (no strategies / no GUI)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from ctf.environment import CTFEnvironment  # noqa: E402
from demo_scenario import (  # noqa: E402
    build_red_capture_script,
    flag_summary,
    format_events,
)


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    env = CTFEnvironment(config)
    state = env.reset()
    agent_ids = config.agent_ids()
    script = build_red_capture_script(agent_ids)

    print("=== Multi-Agent CTF — Prototype 1 Demo (RED flag capture) ===")
    print(f"Grid: {config.width}x{config.height}, seed={config.seed}, tagging={config.tagging_enabled}")
    print(f"RED base {config.red_base}, BLUE base {config.blue_base}")
    print(f"Agents: {', '.join(sorted(state.agents.keys()))}")
    print(f"Start: {flag_summary(state)}")
    print(f"R1 start @ ({state.agents['R1'].x},{state.agents['R1'].y})")
    print()

    last_info: dict = {"events": []}
    for actions in script:
        if env.done:
            break
        state, _rewards, done, last_info = env.step(actions)
        r1 = state.agents["R1"]
        print(
            f"timestep={state.timestep} | R1=({r1.x},{r1.y}) carrying={r1.carrying_flag} "
            f"active={r1.active} | {flag_summary(state)}"
        )
        print(f"  events: {format_events(last_info['events'])}")
        if last_info["rejected_moves"]:
            print(f"  rejected: {last_info['rejected_moves']}")

    print()
    end_events = [e for e in last_info.get("events", []) if e.get("type") == "episode_end"]
    termination_reason = end_events[-1]["reason"] if end_events else "unknown"

    print(f"Episode done={env.done}")
    print(f"winner={state.winner}")
    print(f"termination_reason={termination_reason}")
    print(f"truncation={state.truncation}")

    if state.winner is not Team.RED or termination_reason != "flag_capture":
        print("ERROR: demo did not end with RED winning via flag_capture.")
        sys.exit(1)

    print("Demo complete — RED returned the BLUE flag.")


if __name__ == "__main__":
    main()
