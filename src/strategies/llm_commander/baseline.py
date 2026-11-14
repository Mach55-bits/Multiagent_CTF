"""Scripted teams with NO LLM: the 'no commander' baseline, and simple opponents for testing.
(Placeholders until Team 1's Classical AI exists.)"""
from __future__ import annotations

from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import TeamObservation

from strategies.base import BaseStrategy

from .commands import assign_roles
from .controller import Controller
from .view import build_view

# Simple scripted teams on this package's own controller. "baseline" (role defaults) is the lowest
# baseline for comparisons; opponents for the experiments come from opponents.py.
# Note: "balanced" and "defensive" keep a defender on its base cell (see TODO(8) in controller.py).
# agent number (1,2,3) -> default command
STYLES = {
    "baseline":   {},                                                        # role defaults
    "aggressive": {1: "capture_flag", 2: "capture_flag", 3: "capture_flag"},
    "balanced":   {1: "capture_flag", 2: "defend_base", 3: "capture_flag"},
    "defensive":  {1: "defend_base", 2: "defend_base", 3: "capture_flag"},
}


class ScriptedStrategy(BaseStrategy):
    def __init__(self, style: str = "baseline") -> None:
        if style not in STYLES:
            raise ValueError(f"style must be one of {sorted(STYLES)}")
        self.style, self.controller, self.roles, self.last_t, self.mem = style, Controller(), {}, -1, {}

    def reset(self) -> None:
        self.controller.reset()
        self.roles, self.mem, self.last_t = {}, {}, -1

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        if observation.timestep <= self.last_t:
            self.reset()
        self.last_t = observation.timestep
        if not self.roles:
            ids = [a.agent_id for a in observation.agents.values() if a.team is team]
            self.roles = assign_roles(ids)
            plan = STYLES[self.style]
            self.controller.defaults = {aid: plan[int("".join(c for c in aid if c.isdigit()))]
                                        for aid in ids if int("".join(c for c in aid if c.isdigit())) in plan}
        view = build_view(observation, team, self.roles, self.mem, None, None)
        return self.controller.actions(view)
