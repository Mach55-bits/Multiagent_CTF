from collections import defaultdict

from ctf.agent import Team
from ctf.game_state import GameState


class StrategyEvaluator:

    def __init__(self):
        self.total_games = 0
        self.wins = defaultdict(int)
        self.draws = 0

        self.total_timesteps = 0
        self.total_flag_pickups = 0
        self.total_flag_captures = 0
        self.total_tags = 0
        self.total_rejected_moves = 0

        self.total_moves = defaultdict(int)
        self.role_actions = defaultdict(int)

        self.episode_events = defaultdict(int)
        self.episode_moves = defaultdict(int)
        self.episode_role_actions = defaultdict(int)
        self.episode_rejected_moves = 0

    def record_step(
        self,
        actions: dict,
        info: dict,
        role_assignments: dict | None = None,
    ) -> None:

        for agent_id, action in actions.items():

            if action.value != "STAY":
                self.episode_moves[agent_id] += 1

            if role_assignments is not None:
                role = role_assignments.get(agent_id)

                if role is not None:
                    self.episode_role_actions[role] += 1

        events = info.get("events", [])

        for event in events:
            event_type = event.get("type")

            if event_type == "flag_pickup":
                self.episode_events["flag_pickup"] += 1

            elif event_type == "flag_capture":
                self.episode_events["flag_capture"] += 1

            elif event_type == "tag":
                self.episode_events["tag"] += 1

        self.episode_rejected_moves += len(
            info.get("rejected_moves", {})
        )

    def record_episode(self, final_state: GameState) -> None:

        self.total_games += 1

        if final_state.winner is not None:
            self.wins[final_state.winner] += 1
        else:
            self.draws += 1

        self.total_timesteps += final_state.timestep

        self.total_flag_pickups += self.episode_events["flag_pickup"]
        self.total_flag_captures += self.episode_events["flag_capture"]
        self.total_tags += self.episode_events["tag"]

        self.total_rejected_moves += self.episode_rejected_moves

        for agent_id, moves in self.episode_moves.items():
            self.total_moves[agent_id] += moves

        for role, count in self.episode_role_actions.items():
            self.role_actions[role] += count

        self.reset_episode()

    def reset_episode(self) -> None:

        self.episode_events.clear()
        self.episode_moves.clear()
        self.episode_role_actions.clear()
        self.episode_rejected_moves = 0

    def get_results(self) -> dict:

        total_games = max(self.total_games, 1)

        return {
            "total_games": self.total_games,
            "red_wins": self.wins[Team.RED],
            "blue_wins": self.wins[Team.BLUE],
            "draws": self.draws,
            "red_win_rate": self.wins[Team.RED] / total_games * 100,
            "blue_win_rate": self.wins[Team.BLUE] / total_games * 100,
            "average_timesteps": self.total_timesteps / total_games,
            "total_flag_pickups": self.total_flag_pickups,
            "total_flag_captures": self.total_flag_captures,
            "total_tags": self.total_tags,
            "total_rejected_moves": self.total_rejected_moves,
            "agent_moves": dict(self.total_moves),
            "role_actions": dict(self.role_actions),
        }

    def print_results(self) -> None:

        results = self.get_results()

        print("\n========== STRATEGY EVALUATION ==========")

        for key, value in results.items():
            print(f"{key}: {value}")

        print("=========================================\n")
