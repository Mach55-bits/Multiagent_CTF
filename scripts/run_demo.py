#!/usr/bin/env python3
"""Run a deterministic full Capture the Flag episode (no AI / no GUI)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctf.actions import Action  # noqa: E402
from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from ctf.environment import CTFEnvironment  # noqa: E402
from ctf.game_state import FlagStatus  # noqa: E402


def _idle_actions(agent_ids: list[str]) -> dict[str, Action]:
    return {aid: Action.STAY for aid in agent_ids}


def _with_overrides(base: dict[str, Action], **overrides: Action) -> dict[str, Action]:
    actions = dict(base)
    actions.update(overrides)
    return actions


def _flag_summary(state) -> str:
    parts: list[str] = []
    for team in (Team.RED, Team.BLUE):
        flag = state.flags[team]
        if flag.status is FlagStatus.CARRIED:
            parts.append(f"{team.value} flag CARRIED by {flag.carrier_id} @ ({flag.x},{flag.y})")
        else:
            parts.append(f"{team.value} flag AT_BASE @ ({flag.x},{flag.y})")
    return "; ".join(parts)


def _format_events(events: list[dict]) -> str:
    if not events:
        return "(none)"
    return ", ".join(
        f"{e['type']}"
        + (f"[{e.get('agent_id', '')}]" if e.get("agent_id") else "")
        + (f" reason={e['reason']}" if e.get("reason") else "")
        for e in events
    )


def _build_red_capture_script(agent_ids: list[str]) -> list[dict[str, Action]]:
    """
    R1 routes above the center wall (x=5), captures BLUE at (9,3), returns to RED base (1,3).
    B2 steps off the flag cell on turn 1 so tagging does not stop the pickup.
    """
    idle = _idle_actions(agent_ids)
    outbound_r1 = [Action.UP, Action.UP] + [Action.RIGHT] * 8 + [Action.DOWN] * 3
    # After pickup at (9,3), step LEFT before moving up — avoids defenders at (9,2).
    return_home_r1 = (
        [Action.LEFT] + [Action.UP] * 3 + [Action.LEFT] * 7 + [Action.DOWN] * 3
    )

    script: list[dict[str, Action]] = []
    script.append(_with_overrides(idle, R1=Action.UP, B2=Action.UP))
    for move in outbound_r1[1:]:
        script.append(_with_overrides(idle, R1=move))
    for move in return_home_r1:
        script.append(_with_overrides(idle, R1=move))
    return script


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    env = CTFEnvironment(config)
    state = env.reset()
    agent_ids = config.agent_ids()
    script = _build_red_capture_script(agent_ids)

    print("=== Multi-Agent CTF — Prototype 1 Demo (RED flag capture) ===")
    print(f"Grid: {config.width}x{config.height}, seed={config.seed}, tagging={config.tagging_enabled}")
    print(f"RED base {config.red_base}, BLUE base {config.blue_base}")
    print(f"Agents: {', '.join(sorted(state.agents.keys()))}")
    print(f"Start: {_flag_summary(state)}")
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
            f"active={r1.active} | {_flag_summary(state)}"
        )
        print(f"  events: {_format_events(last_info['events'])}")
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
