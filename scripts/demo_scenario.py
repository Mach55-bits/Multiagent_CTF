"""Shared deterministic RED capture demo (text + visualization)."""

from __future__ import annotations

from ctf.actions import Action
from ctf.agent import Team
from ctf.game_state import FlagStatus


def idle_actions(agent_ids: list[str]) -> dict[str, Action]:
    return {aid: Action.STAY for aid in agent_ids}


def with_overrides(base: dict[str, Action], **overrides: Action) -> dict[str, Action]:
    actions = dict(base)
    actions.update(overrides)
    return actions


def build_red_capture_script(agent_ids: list[str]) -> list[dict[str, Action]]:
    # R1 goes over the wall at x=5; B2 moves off the flag on step 1 (tagging enabled).
    idle = idle_actions(agent_ids)
    outbound_r1 = [Action.UP, Action.UP] + [Action.RIGHT] * 8 + [Action.DOWN] * 3
    return_home_r1 = (
        [Action.LEFT] + [Action.UP] * 3 + [Action.LEFT] * 7 + [Action.DOWN] * 3
    )

    script: list[dict[str, Action]] = []
    script.append(with_overrides(idle, R1=Action.UP, B2=Action.UP))
    for move in outbound_r1[1:]:
        script.append(with_overrides(idle, R1=move))
    for move in return_home_r1:
        script.append(with_overrides(idle, R1=move))
    return script


def flag_summary(state) -> str:
    parts: list[str] = []
    for team in (Team.RED, Team.BLUE):
        flag = state.flags[team]
        if flag.status is FlagStatus.CARRIED:
            parts.append(f"{team.value} flag CARRIED by {flag.carrier_id}")
        else:
            parts.append(f"{team.value} flag AT_BASE")
    return "; ".join(parts)


def format_events(events: list[dict]) -> str:
    if not events:
        return "(none)"
    return ", ".join(format_event_message(e) for e in events)


def format_event_message(event: dict) -> str:
    kind = event.get("type", "")
    if kind == "flag_pickup":
        return f"{event['agent_id']} picked up {event['flag_team']} flag"
    if kind == "tag":
        return (
            f"{event['agent_id']} tagged by {event['by_agent_id']} "
            f"({event['flag_team']} flag returned)"
        )
    if kind == "flag_capture":
        return f"{event['agent_id']} returned {event['flag_team']} flag to base"
    if kind == "episode_end":
        winner = event.get("winner")
        reason = event.get("reason", "")
        if winner:
            return f"Episode end — {winner} wins ({reason})"
        return f"Episode end — no winner ({reason})"
    return str(event)
