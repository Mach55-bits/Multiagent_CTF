"""Curriculum controller: decides when training moves to the next phase."""

from __future__ import annotations

from collections import deque
from typing import Any

from rl.config import PhaseConfig


class Curriculum:
    """A phase ends when the rolling win rate over ``window_episodes`` reaches
    ``advance_win_rate`` (window must be full) or after ``max_steps`` in that phase.
    The last phase never ends."""

    def __init__(self, phases: list[PhaseConfig]) -> None:
        self.phases = phases
        self.index = 0
        self.steps_in_phase = 0
        self._window: deque[float] = deque(maxlen=phases[0].window_episodes)

    @property
    def phase(self) -> PhaseConfig:
        return self.phases[self.index]

    @property
    def is_last(self) -> bool:
        return self.index == len(self.phases) - 1

    @property
    def win_rate(self) -> float | None:
        return sum(self._window) / len(self._window) if self._window else None

    def record(self, outcomes: list[str], steps: int) -> bool:
        """Add finished episodes ('win'/'draw'/'loss') and env steps; True if the phase advanced."""
        self._window.extend(1.0 if o == "win" else 0.0 for o in outcomes)
        self.steps_in_phase += steps
        if self.is_last:
            return False
        full = len(self._window) == self._window.maxlen
        reached = full and (self.win_rate or 0.0) >= self.phase.advance_win_rate
        if reached or self.steps_in_phase >= self.phase.max_steps:
            self.index += 1
            self.steps_in_phase = 0
            self._window = deque(maxlen=self.phase.window_episodes)
            return True
        return False

    def state_dict(self) -> dict[str, Any]:
        return {"index": self.index, "steps_in_phase": self.steps_in_phase, "window": list(self._window)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.index = int(state["index"])
        self.steps_in_phase = int(state["steps_in_phase"])
        self._window = deque(state["window"], maxlen=self.phase.window_episodes)
