"""LLMCommanderStrategy: plugs the commander into the group's BaseStrategy interface."""
from __future__ import annotations

from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import ObservationMode, TeamObservation

from strategies.base import BaseStrategy

from .commander import Commander
from .commands import assign_roles
from .controller import Controller
from .view import build_view


class LLMCommanderStrategy(BaseStrategy):
    """High-level orders from an LLM, executed by a simple low-level controller.

    Works with global observations now and partial ones later (it only uses what the observation shows).
    Create ONE instance per team and per game.
    """

    def __init__(self, llm, every_k_turns: int = 5, max_retries: int = 2,
                 log_path: str | None = None, max_steps: int | None = None) -> None:
        self.commander = Commander(llm, every_k_turns, max_retries, log_path)
        # TODO(5): mixed teams. To let the LLM command Team 1's behaviour-tree agents or Team 2's
        # learned policies, replace this Controller with an adapter that has the same methods
        # (update, report, set_orders, purge_finished, actions); see controller.py.
        self.controller = Controller()
        self.max_steps = max_steps
        self._reset_memory()

    # ---- internal bookkeeping ------------------------------------------
    def _reset_memory(self) -> None:
        self.controller.reset()
        self.roles: dict[str, str] = {}
        self.enemy_memory: dict = {}
        self.shared_map = None
        self.prev_view = None
        self.last_t = -1
        self.timeline: list[str] = []      # human-readable record of orders/events for printing

    def _observed_cells(self, obs: TeamObservation, team: Team):
        if obs.mode is not ObservationMode.PARTIAL:
            return None                      # global: the whole map is visible
        from coordination.shared_map import SharedTeamMap, update_shared_map_from_partial_team

        if self.shared_map is None:
            self.shared_map = SharedTeamMap(obs.width, obs.height, team)
        # TODO(4d): scout information sharing on/off. Add a flag (e.g. share_scout_info=True). When it
        # is False, do not merge the scout's view into the team map (use only each agent's own view).
        update_shared_map_from_partial_team(self.shared_map, obs)
        return {(x, y) for y in range(obs.height) for x in range(obs.width) if self.shared_map.is_observed(x, y)}

    @staticmethod
    def _events(prev, cur) -> list[str]:
        if prev is None:
            return []
        ev = []
        if prev.enemy_flag is not None and cur.enemy_flag is not None:
            if cur.enemy_flag.carried and not prev.enemy_flag.carried:
                ev.append(f"we_picked_up_enemy_flag({cur.enemy_flag.carrier_id})")
            if prev.enemy_flag.carried and not cur.enemy_flag.carried:
                ev.append("our_carrier_was_tagged")
        if cur.own_flag.carried and not prev.own_flag.carried:
            ev.append(f"enemy_stole_our_flag({cur.own_flag.carrier_id})")
        if prev.own_flag.carried and not cur.own_flag.carried:
            ev.append("our_flag_recovered")
        for a in cur.agents:
            before = prev.agent(a.id)
            if before and before.active and not a.active:
                ev.append(f"{a.id}_tagged")
        return ev

    # ---- the BaseStrategy method -----------------------------------------
    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        if observation.timestep <= self.last_t:        # new game started
            self._reset_memory()
        self.last_t = observation.timestep
        if not self.roles:
            self.roles = assign_roles([a.agent_id for a in observation.agents.values() if a.team is team])

        observed = self._observed_cells(observation, team)
        view = build_view(observation, team, self.roles, self.enemy_memory, observed, self.max_steps)
        view.events = self._events(self.prev_view, view)

        self.controller.update(view)                    # arrived? expired? blocked?
        view.order_status = self.controller.report()
        orders = self.commander.decide(view)            # may call the LLM
        for ev in view.events:
            self.timeline.append(f"turn {view.timestep}: EVENT {ev}")
        for o in orders:
            self.timeline.append(f"turn {view.timestep}: ORDER {o['agent_id']} {o['command']} "
                                 f"{o['target'] if o['target'] is not None else ''} | {o['rationale']}")
        self.controller.set_orders(orders, view.timestep)
        self.controller.purge_finished()
        actions = self.controller.actions(view)
        # TODO(3): movement efficiency. Count here how many moves (actions other than STAY) each
        # agent makes per game, e.g. in a self.metrics dict, so distance travelled can be reported.
        self.prev_view = view
        return actions
