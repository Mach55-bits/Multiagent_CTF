#!/usr/bin/env python3
"""Animated Pygame visualization of the deterministic RED capture demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pygame  # noqa: E402

from ctf.agent import Team  # noqa: E402
from ctf.config import load_config  # noqa: E402
from ctf.environment import CTFEnvironment  # noqa: E402
from demo_scenario import (  # noqa: E402
    build_red_capture_script,
    format_event_message,
)
from visualization.renderer import CTFRenderer  # noqa: E402

STEP_DELAY_SEC = 0.25
CONFIG_PATH = ROOT / "configs" / "default.yaml"


class DemoSession:
    """Holds environment + script playback state for one episode."""

    def __init__(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.env = CTFEnvironment(self.config)
        self.script: list = []
        self.script_index = 0
        self.last_events: list[dict] = []
        self.status_lines: list[str] = ["SPACE pause | ESC quit | R restart"]
        self.reset()

    def reset(self) -> None:
        self.env.reset()
        self.script = build_red_capture_script(self.config.agent_ids())
        self.script_index = 0
        self.last_events = []
        self.status_lines = [
            "Deterministic demo — R1 captures BLUE flag and returns to RED base",
            "SPACE pause | ESC quit | R restart",
        ]

    @property
    def state(self):
        return self.env.state

    def can_step(self) -> bool:
        return not self.env.done and self.script_index < len(self.script)

    def step_once(self) -> None:
        if not self.can_step():
            return
        actions = self.script[self.script_index]
        _state, _rewards, _done, info = self.env.step(actions)
        self.script_index += 1
        self.last_events = info.get("events", [])
        self._refresh_status()

    def _refresh_status(self) -> None:
        lines: list[str] = []
        step_label = f"Step {self.state.timestep}"
        for event in self.last_events:
            msg = format_event_message(event)
            if event.get("type") in ("flag_pickup", "tag", "flag_capture", "episode_end"):
                lines.append(f"{step_label} — EVENT: {msg}")
        if not lines:
            lines.append("EVENT: (none this step)")
        if self.env.done:
            end = [e for e in self.last_events if e.get("type") == "episode_end"]
            if end:
                reason = end[-1].get("reason", "")
                winner = end[-1].get("winner")
                if winner:
                    lines.append(f"Reason: {reason}")
                else:
                    lines.append(f"Reason: {reason}")
        lines.append("SPACE pause | ESC quit | R restart")
        self.status_lines = lines


def main() -> None:
    pygame.init()
    session = DemoSession()
    renderer = CTFRenderer(session.config)
    renderer.init_fonts()
    screen = pygame.display.set_mode((renderer.layout.width, renderer.layout.height))
    pygame.display.set_caption("Multi-Agent CTF — Prototype 1 Visualization")
    clock = pygame.time.Clock()

    paused = False
    step_accum_ms = 0.0
    running = True

    while running:
        dt_ms = clock.tick(60)
        if not paused and session.can_step():
            step_accum_ms += dt_ms
            if step_accum_ms >= STEP_DELAY_SEC * 1000:
                step_accum_ms = 0.0
                session.step_once()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    session.reset()
                    step_accum_ms = 0.0
                    paused = False

        renderer.draw(screen, session.state, session.status_lines)
        pygame.display.flip()

    pygame.quit()

    if session.env.done and session.state.winner is Team.RED:
        end = [e for e in session.last_events if e.get("type") == "episode_end"]
        if end and end[-1].get("reason") == "flag_capture":
            return
    # Non-interactive check when window closed early — no failure exit for manual ESC


if __name__ == "__main__":
    main()
