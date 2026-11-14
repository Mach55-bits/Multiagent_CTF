"""Low-level controller: turns orders into the game's UP/DOWN/LEFT/RIGHT/STAY actions.

This simple version walks shortest paths (BFS). classical_adapter.py provides a drop-in replacement
that uses Team 1's A* with the influence map and Team 1's role logic for agents without an order.

rl_adapter.py provides a controller that lets Team 2's RL policy play the agents without an order.
"""
from __future__ import annotations

from ctf.actions import Action

from .commands import DEFAULT_ORDER_LIFETIME, ROLE_DEFAULT, OrderStatus, View, resolve_target
from .pathing import bfs_next_action

FINISHED = ("done", "failed", "blocked", "expired")


class Controller:
    def __init__(self, defaults: dict[str, str] | None = None) -> None:
        self.orders: dict[str, dict] = {}   # agent_id -> order
        self.status: dict[str, str] = {}    # agent_id -> status
        self.defaults = defaults or {}      # agent_id -> default command (overrides role default)

    def reset(self) -> None:
        self.orders.clear(); self.status.clear()

    def attach(self, observation, team, roles) -> None:
        """Called every turn before actions(); the base controller does not need the raw observation."""

    # ---- orders -------------------------------------------------------
    def set_orders(self, orders: list[dict], timestep: int) -> None:
        for o in orders:
            self.orders[o["agent_id"]] = {**o, "issued": timestep}
            self.status[o["agent_id"]] = "active"

    def report(self) -> list[OrderStatus]:
        return [OrderStatus(a, o["command"], o.get("target"), self.status.get(a, "active"))
                for a, o in self.orders.items()]

    def purge_finished(self) -> None:
        for a in [a for a, s in self.status.items() if s in FINISHED]:
            self.orders.pop(a, None); self.status.pop(a, None)

    def _command_for(self, agent) -> dict:
        if agent.id in self.orders:
            return self.orders[agent.id]
        cmd = self.defaults.get(agent.id, ROLE_DEFAULT.get(agent.role, "hold"))
        return {"agent_id": agent.id, "command": cmd, "target": None, "default": True}

    # ---- goal logic -----------------------------------------------------
    def _goal(self, view: View, agent, order: dict):
        """Returns (goal_cell or None, finished_status or None). Reflexes live here."""
        cmd, target = order["command"], order.get("target")
        if agent.carrying:                       # reflex: a carrier always runs home
            return view.own_base, None
        if cmd == "capture_flag":
            ef = view.enemy_flag
            if ef is not None and ef.carried:     # a teammate already has it
                return None, "done"
            return view.enemy_base, None
        if cmd == "defend_base":            
            if view.own_flag.carried:             # reflex: chase the thief
                return view.own_flag.pos, None
            return view.own_base, None
        if cmd == "hold":
            return agent.pos, None
        cell = resolve_target(view, target)
        if cmd == "intercept":
            if isinstance(target, str) and target.lower() == "enemy_carrier":
                return (cell, None) if cell is not None else (None, "done")  # thief gone/tagged
            return (cell, None) if cell is not None else (None, "failed")
        if cmd in ("go_to", "scout"):
            if cell is None:
                return None, "failed"
            return cell, ("done" if agent.pos == cell else None)
        return None, "failed"

    def update(self, view: View) -> None:
        """Refresh statuses (arrived / expired / blocked / agent tagged)."""
        for aid, order in list(self.orders.items()):
            agent = view.agent(aid)
            if agent is None or not agent.active:
                self.orders.pop(aid, None); self.status.pop(aid, None); continue
            if view.timestep - order["issued"] >= DEFAULT_ORDER_LIFETIME:
                self.status[aid] = "expired"; continue
            goal, finished = self._goal(view, agent, order)
            if finished:
                self.status[aid] = finished
            elif goal is not None and bfs_next_action(view.width, view.height, view.obstacles, agent.pos, goal) is None:
                self.status[aid] = "blocked"

    def actions(self, view: View) -> dict[str, Action]:
        result: dict[str, Action] = {}
        for agent in view.agents:
            if not agent.active:
                continue
            order = self._command_for(agent)
            goal, _ = self._goal(view, agent, order)
            move = None if goal is None else bfs_next_action(
                view.width, view.height, view.obstacles, agent.pos, goal)
            result[agent.id] = move if move is not None else Action.STAY
        return result
