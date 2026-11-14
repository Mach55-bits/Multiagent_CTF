"""Observation -> View (what the commander knows) -> text for the LLM."""
from __future__ import annotations

from ctf.agent import Team
from ctf.game_state import FlagStatus
from ctf.observations import ObservationMode

from .commands import AgentView, FlagView, View, zone_table


def _flag_view(flag) -> FlagView:
    return FlagView((flag.x, flag.y), flag.status is FlagStatus.CARRIED, flag.carrier_id)


def build_view(obs, team: Team, roles: dict[str, str], enemy_memory: dict, observed, max_steps,
               observers: list | None = None) -> View:
    """Uses ONLY the observation. enemy_memory persists last-seen enemy positions (for fog of war)."""
    own_base = obs.red_base if team is Team.RED else obs.blue_base
    enemy_base = obs.blue_base if team is Team.RED else obs.red_base
    agents = [
        AgentView(a.agent_id, roles.get(a.agent_id, "attacker"), (a.x, a.y), a.carrying_flag, a.active)
        for a in sorted(obs.agents.values(), key=lambda a: a.agent_id) if a.team is team
    ]
    radius = getattr(obs, "visibility_radius", None)

    def seen_by_observers(x: int, y: int) -> bool:
        """With observers given (scout sharing off), only their own eyes count."""
        if observers is None or radius is None:
            return True
        return any(abs(ox - x) + abs(oy - y) <= radius for ox, oy in observers)

    for a in obs.agents.values():
        if a.team is team.opponent:
            if a.active and seen_by_observers(a.x, a.y):
                enemy_memory[a.agent_id] = ((a.x, a.y), obs.timestep)
            elif not a.active:
                enemy_memory.pop(a.agent_id, None)
    enemies = [(aid, pos, t) for aid, (pos, t) in sorted(enemy_memory.items())]
    enemy_flag = obs.flags.get(team.opponent)
    if enemy_flag is not None and not enemy_flag.carrier_id and not seen_by_observers(enemy_flag.x, enemy_flag.y):
        enemy_flag = None
    return View(
        team=team, timestep=obs.timestep, max_steps=max_steps,
        width=obs.width, height=obs.height, obstacles={tuple(o) for o in obs.obstacles},
        observed=observed, own_base=tuple(own_base), enemy_base=tuple(enemy_base),
        own_flag=_flag_view(obs.flags[team]),
        enemy_flag=_flag_view(enemy_flag) if enemy_flag else None,
        agents=agents, enemies=enemies,
    )


def render_map(view: View) -> str:
    grid = []
    for y in range(view.height):
        row = []
        for x in range(view.width):
            c = (x, y)
            if c in view.obstacles:
                row.append("#")
            elif view.observed is not None and c not in view.observed:
                row.append("?")
            else:
                row.append(".")
        grid.append(row)
    grid[view.own_base[1]][view.own_base[0]] = "H"
    grid[view.enemy_base[1]][view.enemy_base[0]] = "X"
    for _, (x, y), _t in view.enemies:
        grid[y][x] = "e"
    for a in view.agents:
        if a.active:
            grid[a.pos[1]][a.pos[0]] = a.role[0].upper()
    header = "     " + " ".join(str(x % 10) for x in range(view.width))
    lines = [header] + [f"y={y:<2} " + " ".join(r) for y, r in enumerate(grid)]
    return "\n".join(lines)


def view_to_text(view: View, include_map: bool = True) -> str:
    out = []
    limit = f" of {view.max_steps}" if view.max_steps else ""
    out.append(f"You command team {view.team.value}. Turn {view.timestep}{limit}.")
    out.append(f"Map {view.width}x{view.height}; coordinates are (x, y), x = column, y = row, (0,0) top-left, north = smaller y.")
    out.append("Legend: # wall, . open, ? not yet explored, H our base (our flag's home), X enemy base (enemy flag's home), "
               "e enemy, A/D/S our attacker/defender/scout.")
    if include_map:
        out.append(render_map(view))
    out.append("Our agents:")
    for a in view.agents:
        state = "" if a.active else "  [TAGGED - out of the game]"
        carry = "  CARRYING THE ENEMY FLAG" if a.carrying else ""
        out.append(f"- {a.id} ({a.role}) at {a.pos}{carry}{state}")
    f = view.own_flag
    out.append(f"Our flag: {'STOLEN, carried by ' + str(f.carrier_id) + ' now at ' + str(f.pos) if f.carried else 'safe at our base ' + str(f.pos)}.")
    ef = view.enemy_flag
    if ef is None:
        out.append(f"Enemy flag: not currently visible (it starts at the enemy base {view.enemy_base}).")
    elif ef.carried:
        out.append(f"Enemy flag: carried by our agent {ef.carrier_id} at {ef.pos}.")
    else:
        out.append(f"Enemy flag: at the enemy base {ef.pos}.")
    if view.enemies:
        out.append("Enemies (last known): " + "; ".join(
            f"{i} at {p}" + ("" if t == view.timestep else f" (seen turn {t})") for i, p, t in view.enemies))
    else:
        out.append("Enemies: none visible.")
    if f.carried:
        out.append(f"ALERT: our flag is stolen! Carrier at {f.pos}. It must be tagged before it reaches the enemy base {view.enemy_base}.")
    if view.order_status:
        out.append("Your previous orders: " + "; ".join(
            f"{s.agent_id}:{s.command}->{s.target}={s.status}" for s in view.order_status))
    if view.events:
        out.append("Events since your last look: " + ", ".join(view.events))
    out.append("Named zones: " + ", ".join(f"{k}={v}" for k, v in zone_table(view).items())
               + ", enemy_flag, own_flag, enemy_carrier (only when our flag is stolen)")
    return "\n".join(out)
