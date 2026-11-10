"""Safety layer: checks every order before any agent acts on it."""
from __future__ import annotations

from .commands import COMMANDS, NEEDS_TARGET, ROLE_ALLOWED, View, resolve_target
from .pathing import in_bounds, reachable


def validate(view: View, parsed) -> tuple[list[dict], list[str]]:
    """Returns (clean_orders, error_messages). Clean orders keep the target as given
    (zone name or [x, y]) so dynamic targets such as 'enemy_carrier' stay dynamic."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("orders"), list):
        return [], ['Reply must be a JSON object with an "orders" list.']
    clean, errors, seen_agents = [], [], set()
    for i, o in enumerate(parsed["orders"]):
        tag = f"order {i}"
        if not isinstance(o, dict):
            errors.append(f"{tag}: not an object"); continue
        aid, cmd = o.get("agent_id"), o.get("command")
        agent = view.agent(aid) if isinstance(aid, str) else None
        if agent is None:
            errors.append(f"{tag}: unknown agent '{aid}'"); continue
        if not agent.active:
            errors.append(f"{tag}: {aid} is tagged and out of the game"); continue
        if aid in seen_agents:
            errors.append(f"{tag}: {aid} already has an order in this reply"); continue
        if cmd not in COMMANDS:
            errors.append(f"{tag}: unknown command '{cmd}' (allowed: {', '.join(COMMANDS)})"); continue
        if cmd not in ROLE_ALLOWED[agent.role]:
            errors.append(f"{tag}: {aid} is a {agent.role} and cannot use '{cmd}'"); continue
        target = o.get("target")
        if cmd in NEEDS_TARGET:
            cell = resolve_target(view, target)
            if cell is None:
                errors.append(f"{tag}: invalid or currently unavailable target {target!r}"); continue
            if not in_bounds(view.width, view.height, cell):
                errors.append(f"{tag}: target {cell} is outside the map"); continue
            if cell in view.obstacles:
                errors.append(f"{tag}: target {cell} is a wall"); continue
            if not reachable(view.width, view.height, view.obstacles, agent.pos, cell):
                errors.append(f"{tag}: target {cell} cannot be reached by {aid}"); continue
        else:
            target = None
        seen_agents.add(aid)
        clean.append({"agent_id": aid, "command": cmd, "target": target,
                      "rationale": str(o.get("rationale", ""))[:200]})
    return clean, errors
