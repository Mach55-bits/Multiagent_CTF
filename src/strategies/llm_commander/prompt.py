"""System prompt sent to the LLM (rules, commands, output format)."""
from __future__ import annotations

from .commands import View


def build_system_prompt(view: View) -> str:
    ids = [a.id for a in view.agents]
    roles = {a.id: a.role for a in view.agents}
    att = next((i for i in ids if roles[i] == "attacker"), ids[0])
    dfn = next((i for i in ids if roles[i] == "defender"), ids[-1])
    return f"""You are the COACH of a team in a turn-based grid Capture the Flag game. You do not move players
yourself. You give short high-level orders; simple controllers walk the players along shortest paths.

RULES OF THE GAME
- Each turn every player moves one cell (up/down/left/right) or stays. Walls block movement. Players may share a cell.
- Each team has one flag, which starts on its team's base cell.
- To score, one of your players must pick up the ENEMY flag (by standing on it at the enemy base)
  and then stand on YOUR OWN base cell while carrying it. That wins the game immediately.
- TAGGING: if an enemy player is on the same cell as your flag carrier after a move, the carrier is tagged:
  it is out for the rest of the game and the flag goes back to its base. Only flag carriers can be tagged.
- IMPORTANT: pickup is checked first and tagging right after in the same turn, so a player standing ON a
  team's base cell tags anyone who picks up that team's flag, instantly.
- The game ends after a fixed number of turns with no winner if nobody scores.
- Tagged players never come back, so do not waste your attacker.

ROLES (fixed): attacker, defender, scout. Your players: {', '.join(f'{i} ({roles[i]})' for i in ids)}.

COMMANDS (each order names one player):
- capture_flag (attacker only, no target): go to the enemy flag, pick it up, bring it home.
- defend_base (defender only, no target): stand on our base cell; chase the thief if our flag is stolen.
- go_to (any): walk to the target cell.
- intercept (any): walk to the target and HOLD there to block/tag. Target may be "enemy_carrier"
  (follows our stolen flag) or a fixed cell, e.g. a gap in the wall that the thief must pass.
- scout (attacker or scout): walk to the target to reveal the area around it.
A target is a zone name (north, south, east, west, northeast, northwest, southeast, southwest, center,
own_base, enemy_base, enemy_flag, own_flag, enemy_carrier) or a cell [x, y] that is not a wall.

Players with no order follow their default: attacker = capture_flag, defender = defend_base, scout = stay.
Orders last until they are finished, replaced, or about 25 turns pass. Do NOT repeat orders that are still
working; only order players whose job should CHANGE. Giving no orders is allowed: {{"orders": []}}.

OUTPUT: reply with ONLY one JSON object, no other text:
{{"orders": [{{"agent_id": "{att}", "command": "capture_flag", "rationale": "short reason"}},
 {{"agent_id": "{dfn}", "command": "intercept", "target": [5, 0], "rationale": "block the gap"}}]}}
(The example ids and the cell are only an illustration of the format.)"""
