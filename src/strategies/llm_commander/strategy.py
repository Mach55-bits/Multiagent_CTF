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
    """High-level orders from an LLM, executed by a low-level controller.

    Works with global and partial observations (it only uses what the observation shows).
    Create ONE instance per team and per game.

    controller:        "simple" (BFS walker), "classical" (Team 1's A*, influence map and role logic),
                       "rl" (Team 2's trained policy plays every agent that has no order; needs a checkpoint)
                       or a ready-made Controller object
    use_validator:     False = ablation, orders are executed without legality checks
    include_map:       False = ablation, the prompt has no ASCII map
    share_scout_info:  False = ablation (partial observability only), the scout's observations are
                       not merged into what the commander knows
    """

    def __init__(self, llm, every_k_turns: int = 5, max_retries: int = 2,
                 log_path: str | None = None, max_steps: int | None = None,
                 controller: str = "simple", use_validator: bool = True,
                 include_map: bool = True, share_scout_info: bool = True,
                 rl_checkpoint: str | None = None) -> None:
        if isinstance(controller, str) and controller not in ("simple", "classical", "rl"):
            raise ValueError("controller must be 'simple', 'classical', 'rl' or a Controller object")
        self.commander = Commander(llm, every_k_turns, max_retries, log_path,
                                   use_validator=use_validator, include_map=include_map)
        self.controller_kind = controller if isinstance(controller, str) else "custom"
        if not isinstance(controller, str):
            self.controller = controller                 # any object with the Controller methods
        elif controller == "classical":
            from .classical_adapter import ClassicalControllerAdapter

            self.controller = ClassicalControllerAdapter()
        elif controller == "rl":
            from .rl_adapter import RLControllerAdapter

            self.controller = RLControllerAdapter(rl_checkpoint)
        else:
            self.controller = Controller()
        self.max_steps = max_steps
        self.share_scout_info = share_scout_info
        self.reset()

    @property
    def needs_partial_observation(self) -> bool:
        """The RL policy was trained with fog of war, so it must be given partial observations."""
        return bool(getattr(self.controller, "needs_partial_observation", False))

    # ---- per-game memory ---------------------------------------------------
    def reset(self) -> None:
        self.controller.reset()
        self.roles: dict[str, str] = {}
        self.enemy_memory: dict = {}
        self.shared_map = None
        self.prev_view = None
        self.last_t = -1
        self.timeline: list[str] = []      # human-readable record of orders/events for printing
        self.moves: dict[str, int] = {}    # moves (actions other than STAY) per agent, this game

    def _team_knowledge(self, obs: TeamObservation, team: Team):
        """Returns (observed cells or None, observer positions or None) for the view."""
        if obs.mode is not ObservationMode.PARTIAL:
            return None, None                # global: the whole map is visible
        from coordination.shared_map import SharedTeamMap, build_scout_view_from_partial

        if self.shared_map is None:
            self.shared_map = SharedTeamMap(obs.width, obs.height, team)
        observers = []
        for agent_id, agent in obs.agents.items():
            if agent.team is not team or not agent.active:
                continue
            if not self.share_scout_info and self.roles.get(agent_id) == "scout":
                continue                     # scout sharing off: its observations are not merged
            self.shared_map.update_from_scout(build_scout_view_from_partial(obs, agent_id))
            observers.append((agent.x, agent.y))
        observed = {(x, y) for y in range(obs.height) for x in range(obs.width)
                    if self.shared_map.is_observed(x, y)}
        return observed, (None if self.share_scout_info else observers)

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
            self.reset()
        self.last_t = observation.timestep
        if not self.roles:
            self.roles = assign_roles([a.agent_id for a in observation.agents.values() if a.team is team])

        observed, observers = self._team_knowledge(observation, team)
        view = build_view(observation, team, self.roles, self.enemy_memory, observed, self.max_steps, observers)
        view.events = self._events(self.prev_view, view)

        self.controller.attach(observation, team, self.roles)
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
        for aid, action in actions.items():
            if action is not Action.STAY:
                self.moves[aid] = self.moves.get(aid, 0) + 1
        self.prev_view = view
        return actions
