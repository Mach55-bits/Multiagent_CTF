"""Rollout storage and generalized advantage estimation (GAE)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rl.networks import NetworkSpec


@dataclass
class Batch:
    """Flattened training samples (N = T * E * A), all torch tensors on one device."""

    grid: torch.Tensor
    vec: torch.Tensor
    mask: torch.Tensor
    role_index: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor  # raw (unnormalized) scale
    old_values: torch.Tensor  # raw scale
    alive: torch.Tensor  # float 0/1
    critic_grid: torch.Tensor
    critic_vec: torch.Tensor

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class RolloutBuffer:
    def __init__(self, steps: int, envs: int, agents: int, spec: NetworkSpec) -> None:
        T, E, A = steps, envs, agents
        self.T, self.E, self.A = T, E, A
        h, w = spec.height, spec.width
        f32 = np.float32
        self.grid = np.zeros((T, E, A, spec.actor_channels, h, w), f32)
        self.vec = np.zeros((T, E, A, spec.actor_vec_dim), f32)
        self.mask = np.zeros((T, E, A, spec.num_actions), bool)
        self.alive = np.zeros((T, E, A), f32)
        self.role_index = np.zeros((T, E, A), np.int64)
        self.critic_grid = np.zeros((T, E, A, spec.critic_channels, h, w), f32)
        self.critic_vec = np.zeros((T, E, A, spec.critic_vec_dim), f32)
        self.actions = np.zeros((T, E, A), np.int64)
        self.log_probs = np.zeros((T, E, A), f32)
        self.values = np.zeros((T + 1, E, A), np.float64)  # raw scale; [T] is the bootstrap
        self.rewards = np.zeros((T, E, A), np.float64)
        self.dones = np.zeros((T, E), np.float64)  # episode ended after this step
        self.advantages = np.zeros((T, E, A), np.float64)
        self.returns = np.zeros((T, E, A), np.float64)

    def compute_gae(self, gamma: float, lam: float) -> None:
        """Fill advantages/returns. ``values[T]`` must hold the bootstrap value."""
        last = np.zeros((self.E, self.A))
        for t in reversed(range(self.T)):
            nonterminal = (1.0 - self.dones[t])[:, None]
            delta = self.rewards[t] + gamma * self.values[t + 1] * nonterminal - self.values[t]
            last = delta + gamma * lam * nonterminal * last
            self.advantages[t] = last
        self.returns = self.advantages + self.values[: self.T]

    def to_batch(self, device: torch.device) -> Batch:
        def flat(x: np.ndarray) -> np.ndarray:
            return x.reshape(self.T * self.E * self.A, *x.shape[3:])

        def t(x: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
            out = torch.from_numpy(np.ascontiguousarray(flat(x)))
            return out.to(device=device, dtype=dtype) if dtype else out.to(device)

        return Batch(
            grid=t(self.grid),
            vec=t(self.vec),
            mask=t(self.mask),
            role_index=t(self.role_index),
            actions=t(self.actions),
            old_log_probs=t(self.log_probs),
            advantages=t(self.advantages, torch.float32),
            returns=t(self.returns, torch.float32),
            old_values=t(self.values[: self.T], torch.float32),
            alive=t(self.alive),
            critic_grid=t(self.critic_grid),
            critic_vec=t(self.critic_vec),
        )
