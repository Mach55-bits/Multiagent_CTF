"""Pygame renderer for CTF environment state (no game logic)."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from ctf.agent import Agent, Team
from ctf.config import CTFConfig
from ctf.game_state import FlagStatus, GameState


@dataclass
class Layout:
    cell_size: int
    grid_origin_x: int
    grid_origin_y: int
    hud_y: int
    width: int
    height: int


class CTFRenderer:
    """Draws a GameState snapshot; does not mutate or simulate."""

    BG = (245, 245, 240)
    GRID_LINE = (210, 210, 205)
    OBSTACLE = (55, 55, 60)
    RED_BASE = (255, 210, 210)
    BLUE_BASE = (210, 220, 255)
    RED_AGENT = (200, 40, 40)
    BLUE_AGENT = (40, 90, 200)
    INACTIVE = (150, 150, 150)
    RED_FLAG = (180, 20, 20)
    BLUE_FLAG = (20, 60, 180)
    TEXT = (25, 25, 25)
    MUTED = (90, 90, 90)
    WIN_BANNER = (20, 120, 40)

    def __init__(self, config: CTFConfig) -> None:
        self.config = config
        self.layout = self._compute_layout(config.width, config.height)
        self.font: pygame.font.Font
        self.font_sm: pygame.font.Font
        self.font_lg: pygame.font.Font

    def _compute_layout(self, grid_w: int, grid_h: int) -> Layout:
        hud_h = 120
        margin = 24
        max_grid_w = 880
        max_grid_h = 520
        cell = min(max_grid_w // grid_w, max_grid_h // grid_h, 64)
        cell = max(cell, 32)
        grid_px_w = cell * grid_w
        grid_px_h = cell * grid_h
        win_w = grid_px_w + margin * 2
        win_h = grid_px_h + margin + hud_h
        return Layout(
            cell_size=cell,
            grid_origin_x=margin,
            grid_origin_y=margin,
            hud_y=margin + grid_px_h + 8,
            width=win_w,
            height=win_h,
        )

    def init_fonts(self) -> None:
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 16)
        self.font_sm = pygame.font.SysFont("consolas,dejavusansmono,monospace", 13)
        self.font_lg = pygame.font.SysFont("consolas,dejavusansmono,monospace", 22, bold=True)

    def cell_rect(self, x: int, y: int) -> pygame.Rect:
        cs = self.layout.cell_size
        return pygame.Rect(
            self.layout.grid_origin_x + x * cs,
            self.layout.grid_origin_y + y * cs,
            cs,
            cs,
        )

    def draw(self, surface: pygame.Surface, state: GameState, status_lines: list[str]) -> None:
        surface.fill(self.BG)
        self._draw_grid(surface, state)
        self._draw_bases(surface, state)
        self._draw_obstacles(surface, state)
        self._draw_flags(surface, state)
        self._draw_agents(surface, state)
        self._draw_hud(surface, state, status_lines)

    def _draw_grid(self, surface: pygame.Surface, state: GameState) -> None:
        for y in range(state.grid.height):
            for x in range(state.grid.width):
                rect = self.cell_rect(x, y)
                pygame.draw.rect(surface, self.BG, rect)
                pygame.draw.rect(surface, self.GRID_LINE, rect, 1)

    def _draw_bases(self, surface: pygame.Surface, state: GameState) -> None:
        pygame.draw.rect(surface, self.RED_BASE, self.cell_rect(*state.red_base))
        pygame.draw.rect(surface, self.BLUE_BASE, self.cell_rect(*state.blue_base))

    def _draw_obstacles(self, surface: pygame.Surface, state: GameState) -> None:
        for x, y in state.grid.obstacles:
            pygame.draw.rect(surface, self.OBSTACLE, self.cell_rect(x, y))

    def _draw_flags(self, surface: pygame.Surface, state: GameState) -> None:
        for team, flag in state.flags.items():
            if flag.status is FlagStatus.CARRIED:
                continue
            color = self.RED_FLAG if team is Team.RED else self.BLUE_FLAG
            self._draw_flag_marker(surface, flag.x, flag.y, color, small=False)

    def _draw_flag_marker(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        color: tuple[int, int, int],
        *,
        small: bool,
    ) -> None:
        rect = self.cell_rect(x, y)
        cx, cy = rect.centerx, rect.centery
        pole_h = rect.height // (5 if small else 4)
        pygame.draw.line(surface, color, (cx - 4, cy + pole_h // 2), (cx - 4, cy - pole_h), 3)
        points = [(cx - 2, cy - pole_h), (cx + 10, cy - pole_h + 6), (cx - 2, cy - pole_h + 12)]
        pygame.draw.polygon(surface, color, points)

    def _draw_agents(self, surface: pygame.Surface, state: GameState) -> None:
        for aid in sorted(state.agents.keys()):
            agent = state.agents[aid]
            self._draw_agent(surface, agent)

    def _draw_agent(self, surface: pygame.Surface, agent: Agent) -> None:
        rect = self.cell_rect(agent.x, agent.y)
        cx, cy = rect.centerx, rect.centery
        radius = max(8, self.layout.cell_size // 4)
        if agent.active:
            color = self.RED_AGENT if agent.team is Team.RED else self.BLUE_AGENT
            pygame.draw.circle(surface, color, (cx, cy), radius)
        else:
            pygame.draw.circle(surface, self.INACTIVE, (cx, cy), radius, 3)

        label = self.font_sm.render(agent.agent_id, True, self.TEXT)
        surface.blit(label, (cx - label.get_width() // 2, cy - radius - label.get_height() - 2))

        if agent.carrying_flag:
            enemy = agent.team.opponent
            fcolor = self.BLUE_FLAG if enemy is Team.BLUE else self.RED_FLAG
            self._draw_flag_marker(surface, agent.x, agent.y, fcolor, small=True)
            carry_text = self.font_sm.render(f"+ {enemy.value} FLAG", True, fcolor)
            surface.blit(carry_text, (cx - carry_text.get_width() // 2, cy + radius + 2))

    def _draw_hud(self, surface: pygame.Surface, state: GameState, status_lines: list[str]) -> None:
        y = self.layout.hud_y
        header = f"Timestep: {state.timestep}"
        if state.terminated and state.winner:
            header += f"  |  {state.winner.value} WINS"
        elif state.terminated and state.truncation:
            header += "  |  Episode truncated (max steps)"
        surf = self.font.render(header, True, self.TEXT)
        surface.blit(surf, (self.layout.grid_origin_x, y))
        y += 22

        for line in status_lines[:4]:
            surf = self.font_sm.render(line, True, self.MUTED)
            surface.blit(surf, (self.layout.grid_origin_x, y))
            y += 18

        if state.terminated and state.winner:
            banner = self.font_lg.render(f"{state.winner.value} WINS", True, self.WIN_BANNER)
            bx = self.layout.width // 2 - banner.get_width() // 2
            by = self.layout.grid_origin_y + 8
            surface.blit(banner, (bx, by))
