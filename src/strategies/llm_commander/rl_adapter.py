"""Integration with Team 2's reinforcement-learning agent (branch rl_agent).

RLPolicyStrategy   runs a trained RL policy as a normal BaseStrategy (opponent, or Approach 2 baseline).
RLControllerAdapter mixed team: agents with an LLM order follow it (shortest path); all other agents
                    are played by the RL policy. The policy has no input for orders, so the LLM can only
                    take over individual agents, not steer the policy's own decisions.

The policy was trained under fog of war, so it always needs a PartialTeamObservation. Strategies that
need one set `needs_partial_observation = True`; runner.play_game then supplies it, also in games
that are otherwise played with global observability.

Requires the rl_agent branch, NumPy and PyTorch (uv sync --extra dev --extra rl)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import ObservationMode, TeamObservation
from strategies.base import BaseStrategy

from .controller import Controller

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = "runs/mappo_default/checkpoints/latest.pt"   # where `python -m rl.train` writes
UNTRAINED = "untrained"        # random weights: only for testing the plumbing, plays badly


def resolve_checkpoint(path: str | None) -> str:
    """Checkpoint path from the argument, the RL_CHECKPOINT environment variable or the default location.
    Raises RuntimeError with instructions if the file does not exist ('untrained' is always accepted)."""
    import os

    path = path or os.environ.get("RL_CHECKPOINT") or DEFAULT_CHECKPOINT
    if path == UNTRAINED:
        return path
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    if not p.exists():
        raise RuntimeError(
            f"RL checkpoint not found: {path}\nTrain the RL agent first (python -m rl.train --config "
            f"configs/rl/default.yaml), or pass an existing file with --rl-checkpoint. "
            f"'--rl-checkpoint untrained' uses random weights and only tests that the plumbing works.")
    return str(p)


def _torch_infer(model, device, deterministic: bool):
    import torch

    def infer(features) -> np.ndarray:
        def tensor(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)

        actions, _ = model.act(tensor(features.grid), tensor(features.vec), tensor(features.mask),
                               tensor(features.role_index), deterministic=deterministic)
        return actions.cpu().numpy()

    return infer


class RLPolicyStrategy(BaseStrategy):
    needs_partial_observation = True

    def __init__(self, checkpoint: str | None = None, *, deterministic: bool = True, device: str = "cpu",
                 seed: int = 0, infer=None, env_config=None, rl_config=None) -> None:
        """checkpoint: path to a .pt file written by rl.train, or "untrained".
        infer / env_config / rl_config: supply all three to plug in a stand-in network (used by the tests)."""
        if infer is None:
            infer, env_config, rl_config = self._load(checkpoint or DEFAULT_CHECKPOINT, deterministic, device, seed)
        self._infer, self.env_config, self.rl_config = infer, env_config, rl_config
        self._builders: dict = {}
        self._last_t: dict = {}

    @staticmethod
    def _load(checkpoint: str, deterministic: bool, device: str, seed: int):
        import torch

        if checkpoint == UNTRAINED:
            from rl.config import load_base_env_config, load_rl_config
            from rl.networks import ActorCritic, NetworkSpec

            rl_config = load_rl_config(str(REPO_ROOT / "configs/rl/default.yaml"), {})
            env_config = load_base_env_config(rl_config, root=REPO_ROOT)
            torch.manual_seed(seed)
            model = ActorCritic(NetworkSpec.from_env(env_config), rl_config.model).to(device).eval()
        else:
            from rl.checkpoint import CheckpointError, load_policy

            path = Path(checkpoint)
            if not path.is_absolute() and not path.exists():
                path = REPO_ROOT / path
            try:
                model, meta = load_policy(path, device)
            except CheckpointError as e:
                raise RuntimeError(
                    f"{e}\nTrain the RL agent first (python -m rl.train --config configs/rl/default.yaml) or "
                    f"pass the path of an existing checkpoint with --rl-checkpoint "
                    f"(use 'untrained' only to test the plumbing).") from e
            rl_config, env_config = meta["config"], meta["env_config"]
        return _torch_infer(model, device, deterministic), env_config, rl_config

    def reset(self) -> None:
        self._builders.clear()
        self._last_t.clear()

    def act(self, observation: TeamObservation, team: Team) -> dict[str, Action]:
        from rl.features import FeatureBuilder

        if observation.mode is not ObservationMode.PARTIAL:
            raise ValueError("The RL policy needs a partial team observation (it was trained with fog of war). "
                             "Run it through strategies.llm_commander.runner.play_game, which provides one.")
        builder = self._builders.get(team)
        if builder is None or observation.timestep <= self._last_t.get(team, -1):     # first call or new game
            builder = FeatureBuilder(self.env_config, team, list(self.rl_config.roles.slots), self.rl_config.features)
            self._builders[team] = builder
        self._last_t[team] = observation.timestep
        if (observation.width, observation.height) != (builder.width, builder.height):
            raise ValueError(f"Checkpoint was trained on a {builder.width}x{builder.height} map but the game "
                             f"is {observation.width}x{observation.height}.")
        features = builder.actor_features(observation)
        chosen = self._infer(features)
        return {aid: builder.canon.to_env_action(int(chosen[slot]))
                for slot, aid in enumerate(builder.agent_ids) if features.alive[slot]}


class RLControllerAdapter(Controller):
    """Mixed team: LLM orders for the agents that have one, the RL policy for everybody else."""

    needs_partial_observation = True

    def __init__(self, checkpoint: str | None = None, policy: RLPolicyStrategy | None = None) -> None:
        super().__init__()
        self.policy = policy or RLPolicyStrategy(checkpoint)
        self.observation = None
        self.team = None

    def reset(self) -> None:
        super().reset()
        self.policy.reset()

    def attach(self, observation, team, roles) -> None:
        self.observation, self.team = observation, team

    def actions(self, view) -> dict[str, Action]:
        walking = super().actions(view)                          # shortest-path moves for ordered agents
        learned = self.policy.act(self.observation, self.team)   # the RL policy's move for every agent
        return {a.id: (walking[a.id] if a.id in self.orders else learned.get(a.id, Action.STAY))
                for a in view.agents if a.active}
