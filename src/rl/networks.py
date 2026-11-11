"""CNN actor-critic with role-conditioned policy heads (parameter sharing across agents).

Actor:  grid channels --conv--> flatten --fc--> concat(role/scalars) --fc--> role head
Critic: privileged global-state channels, same encoder shape, one value per agent.
Invalid actions (walls, edges) are masked out of the policy distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ctf.config import CTFConfig
from rl.canonical import NUM_ACTIONS
from rl.config import ModelConfig
from rl.features import ACTOR_CHANNELS, ACTOR_VECTOR, CRITIC_CHANNELS, CRITIC_VECTOR, ROLE_ORDER


@dataclass(frozen=True)
class NetworkSpec:
    actor_channels: int
    critic_channels: int
    height: int
    width: int
    actor_vec_dim: int
    critic_vec_dim: int
    num_actions: int
    num_roles: int

    @classmethod
    def from_env(cls, env_config: CTFConfig) -> "NetworkSpec":
        return cls(
            actor_channels=len(ACTOR_CHANNELS),
            critic_channels=len(CRITIC_CHANNELS),
            height=env_config.height,
            width=env_config.width,
            actor_vec_dim=len(ACTOR_VECTOR),
            critic_vec_dim=len(CRITIC_VECTOR),
            num_actions=NUM_ACTIONS,
            num_roles=len(ROLE_ORDER),
        )


def _init(module: nn.Module, gain: float) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class ConvEncoder(nn.Module):
    def __init__(
        self, in_channels: int, height: int, width: int, vec_dim: int, cfg: ModelConfig
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for out_channels in cfg.conv_channels:
            layers += [nn.Conv2d(channels, out_channels, cfg.kernel_size, padding=cfg.kernel_size // 2), nn.ReLU()]
            channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.fc_grid = nn.Linear(channels * height * width, cfg.hidden_size)
        self.fc_mix = nn.Linear(cfg.hidden_size + vec_dim, cfg.hidden_size)
        self.apply(lambda m: _init(m, gain=float(np.sqrt(2.0))))

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc_grid(self.conv(grid).flatten(1)))
        return torch.relu(self.fc_mix(torch.cat([x, vec], dim=1)))


class ActorCritic(nn.Module):
    def __init__(self, spec: NetworkSpec, cfg: ModelConfig) -> None:
        super().__init__()
        self.spec = spec
        self.actor_encoder = ConvEncoder(spec.actor_channels, spec.height, spec.width, spec.actor_vec_dim, cfg)
        self.role_heads = nn.Linear(cfg.hidden_size, spec.num_roles * spec.num_actions)
        self.critic_encoder = ConvEncoder(spec.critic_channels, spec.height, spec.width, spec.critic_vec_dim, cfg)
        self.value_head = nn.Linear(cfg.hidden_size, 1)
        _init(self.role_heads, gain=0.01)
        _init(self.value_head, gain=1.0)

    # ----------------------------------------------------------------- policy

    def policy(
        self,
        grid: torch.Tensor,
        vec: torch.Tensor,
        mask: torch.Tensor,
        role_index: torch.Tensor,
    ) -> Categorical:
        hidden = self.actor_encoder(grid, vec)
        all_logits = self.role_heads(hidden).view(-1, self.spec.num_roles, self.spec.num_actions)
        logits = all_logits[torch.arange(all_logits.shape[0], device=grid.device), role_index]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(
        self,
        grid: torch.Tensor,
        vec: torch.Tensor,
        mask: torch.Tensor,
        role_index: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.policy(grid, vec, mask, role_index)
        actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions)

    def evaluate_actions(
        self,
        grid: torch.Tensor,
        vec: torch.Tensor,
        mask: torch.Tensor,
        role_index: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.policy(grid, vec, mask, role_index)
        return dist.log_prob(actions), dist.entropy()

    # ----------------------------------------------------------------- critic

    def value(self, critic_grid: torch.Tensor, critic_vec: torch.Tensor) -> torch.Tensor:
        """Value in the (possibly normalized) scale the critic is trained on."""
        return self.value_head(self.critic_encoder(critic_grid, critic_vec)).squeeze(-1)
