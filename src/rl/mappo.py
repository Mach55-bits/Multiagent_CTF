"""MAPPO update: PPO with a shared actor, a centralized critic and optional value normalization."""

from __future__ import annotations

import torch
from torch import nn

from rl.buffer import Batch
from rl.config import PPOConfig
from rl.networks import ActorCritic


class ValueNorm(nn.Module):
    """Running mean/std of return targets (the critic predicts normalized returns)."""

    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        n = float(x.numel())
        batch_mean, batch_var = x.mean(), x.var(unbiased=False)
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total)
        self.count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8) if self.enabled else x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var + 1e-8) + self.mean if self.enabled else x


class MAPPO:
    def __init__(self, model: ActorCritic, cfg: PPOConfig, value_norm: ValueNorm) -> None:
        self.model = model
        self.cfg = cfg
        self.value_norm = value_norm
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)

    def update(
        self,
        batch: Batch,
        learning_rate: float,
        entropy_coef: float,
        generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        cfg = self.cfg
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        self.value_norm.update(batch.returns)
        targets = self.value_norm.normalize(batch.returns)

        advantages = batch.advantages
        if cfg.normalize_advantages:
            alive = batch.alive > 0
            if alive.any():
                mean, std = advantages[alive].mean(), advantages[alive].std(unbiased=False)
                advantages = (advantages - mean) / (std + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        updates = 0
        n = len(batch)
        sizes = [n // cfg.num_minibatches + (1 if i < n % cfg.num_minibatches else 0) for i in range(cfg.num_minibatches)]
        for _ in range(cfg.epochs):
            order = torch.randperm(n, generator=generator, device=batch.actions.device)
            start = 0
            for size in sizes:
                idx = order[start : start + size]
                start += size
                if idx.numel() == 0:
                    continue
                m = self._minibatch_loss(batch, idx, advantages, targets, entropy_coef)
                self.optimizer.zero_grad()
                m["loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                for key in stats:
                    stats[key] += float(m[key])
                updates += 1
        stats = {k: v / max(updates, 1) for k, v in stats.items()}
        stats["explained_variance"] = self._explained_variance(batch)
        return stats

    def _minibatch_loss(self, b: Batch, idx, advantages, targets, entropy_coef) -> dict:
        cfg = self.cfg
        new_log_probs, entropy = self.model.evaluate_actions(
            b.grid[idx], b.vec[idx], b.mask[idx], b.role_index[idx], b.actions[idx]
        )
        values = self.model.value(b.critic_grid[idx], b.critic_vec[idx])
        alive = b.alive[idx]
        denom = alive.sum().clamp(min=1.0)

        log_ratio = new_log_probs - b.old_log_probs[idx]
        ratio = log_ratio.exp()
        adv = advantages[idx]
        surrogate = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv)
        policy_loss = -(surrogate * alive).sum() / denom
        entropy_mean = (entropy * alive).sum() / denom
        # The critic learns from every sample: eliminated agents still share the team outcome.
        value_loss = 0.5 * ((values - targets[idx]) ** 2).mean()
        loss = policy_loss + cfg.value_coef * value_loss - entropy_coef * entropy_mean

        with torch.no_grad():
            approx_kl = (((ratio - 1) - log_ratio) * alive).sum() / denom
            clipped = ((ratio - 1).abs() > cfg.clip_epsilon).float()
            clip_fraction = (clipped * alive).sum() / denom
        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy_mean.detach(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    @torch.no_grad()
    def _explained_variance(self, batch: Batch) -> float:
        var = batch.returns.var(unbiased=False)
        if var < 1e-12:
            return 0.0
        return float(1.0 - (batch.returns - batch.old_values).var(unbiased=False) / var)
