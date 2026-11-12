"""Rollout collection shared by the trainer and the benchmark."""

from __future__ import annotations

import numpy as np
import torch

from rl.buffer import RolloutBuffer
from rl.mappo import ValueNorm
from rl.networks import ActorCritic
from rl.vec_env import VecStep


def _flat(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x.reshape(-1, *x.shape[2:]))).to(device)


def collect(
    vec,
    obs: VecStep,
    model: ActorCritic,
    value_norm: ValueNorm,
    buffer: RolloutBuffer,
    device: torch.device,
) -> tuple[VecStep, list[dict]]:
    """Fill ``buffer`` with ``buffer.T`` steps from every env.

    Returns the observation to continue from and the summaries of episodes that ended.
    ``buffer.values[T]`` receives the bootstrap value of that final observation.
    """
    envs, agents = buffer.E, buffer.A
    finished: list[dict] = []
    model.eval()
    for t in range(buffer.T):
        grid, vec_in = _flat(obs.grid, device), _flat(obs.vec, device)
        mask, roles = _flat(obs.mask, device), _flat(obs.role_index, device)
        c_grid, c_vec = _flat(obs.critic_grid, device), _flat(obs.critic_vec, device)
        with torch.no_grad():
            actions, log_probs = model.act(grid, vec_in, mask, roles)
            values = value_norm.denormalize(model.value(c_grid, c_vec))
        actions_np = actions.view(envs, agents).cpu().numpy()

        buffer.grid[t], buffer.vec[t], buffer.mask[t] = obs.grid, obs.vec, obs.mask
        buffer.alive[t], buffer.role_index[t] = obs.alive, obs.role_index
        buffer.critic_grid[t], buffer.critic_vec[t] = obs.critic_grid, obs.critic_vec
        buffer.actions[t] = actions_np
        buffer.log_probs[t] = log_probs.view(envs, agents).cpu().numpy()
        buffer.values[t] = values.view(envs, agents).double().cpu().numpy()

        obs = vec.step(actions_np)
        buffer.rewards[t], buffer.dones[t] = obs.rewards, obs.dones
        finished.extend(obs.infos)

    with torch.no_grad():
        last = value_norm.denormalize(model.value(_flat(obs.critic_grid, device), _flat(obs.critic_vec, device)))
    buffer.values[buffer.T] = last.view(envs, agents).double().cpu().numpy()
    return obs, finished
