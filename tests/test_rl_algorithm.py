import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataclasses import replace  # noqa: E402

from rl.buffer import RolloutBuffer  # noqa: E402
from rl.config import load_base_env_config  # noqa: E402
from rl.mappo import MAPPO, ValueNorm  # noqa: E402
from rl.networks import ActorCritic, NetworkSpec  # noqa: E402
from rl_helpers import ROOT, rl_config  # noqa: E402

CFG = rl_config()
SPEC = NetworkSpec.from_env(load_base_env_config(CFG, ROOT))
DEVICE = torch.device("cpu")


def _model(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(SPEC, CFG.model)


def _inputs(n: int, mask=None, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.rand(n, SPEC.actor_channels, SPEC.height, SPEC.width, generator=g)
    vec = torch.rand(n, SPEC.actor_vec_dim, generator=g)
    mask = torch.ones(n, SPEC.num_actions, dtype=torch.bool) if mask is None else mask
    roles = torch.randint(0, SPEC.num_roles, (n,), generator=g)
    return grid, vec, mask, roles


def test_forward_shapes_and_parameter_budget() -> None:
    m = _model()
    grid, vec, mask, roles = _inputs(7)
    actions, logp = m.act(grid, vec, mask, roles)
    assert actions.shape == (7,) and logp.shape == (7,)
    cgrid = torch.rand(7, SPEC.critic_channels, SPEC.height, SPEC.width)
    cvec = torch.rand(7, SPEC.critic_vec_dim)
    assert m.value(cgrid, cvec).shape == (7,)
    assert sum(p.numel() for p in m.parameters()) < 2_000_000


def test_masked_actions_are_never_sampled_and_have_zero_probability() -> None:
    m = _model()
    n = 2000
    grid, vec, _, roles = _inputs(n)
    mask = torch.rand(n, SPEC.num_actions) > 0.5
    mask[:, -1] = True  # STAY always legal
    actions, _ = m.act(grid, vec, mask, roles)
    assert mask.gather(1, actions[:, None]).all()
    probs = m.policy(grid, vec, mask, roles).probs
    assert (probs[~mask] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(n), atol=1e-5)


def test_deterministic_action_is_the_most_likely_legal_action() -> None:
    m = _model()
    grid, vec, mask, roles = _inputs(50)
    mask[:, 0] = False
    actions, _ = m.act(grid, vec, mask, roles, deterministic=True)
    assert (actions != 0).all()
    assert torch.equal(actions, m.policy(grid, vec, mask, roles).probs.argmax(-1))


def test_role_selects_a_different_policy_head() -> None:
    m = _model()
    with torch.no_grad():
        m.role_heads.weight.normal_(0, 1.0)  # make heads clearly different
    grid, vec, mask, _ = _inputs(1)
    a = m.policy(grid, vec, mask, torch.tensor([0])).probs
    b = m.policy(grid, vec, mask, torch.tensor([1])).probs
    assert not torch.allclose(a, b)


def test_gae_matches_a_hand_computed_example() -> None:
    buf = RolloutBuffer(3, 1, 1, SPEC)
    buf.rewards[:, 0, 0] = [1.0, 0.0, 2.0]
    buf.values[:, 0, 0] = [0.5, 0.4, 0.3, 0.2]
    buf.dones[:, 0] = [0.0, 0.0, 0.0]
    gamma, lam = 0.9, 0.8
    buf.compute_gae(gamma, lam)
    d2 = 2.0 + gamma * 0.2 - 0.3
    d1 = 0.0 + gamma * 0.3 - 0.4
    d0 = 1.0 + gamma * 0.4 - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    np.testing.assert_allclose(buf.advantages[:, 0, 0], [a0, a1, a2])
    np.testing.assert_allclose(buf.returns[:, 0, 0], np.array([a0, a1, a2]) + [0.5, 0.4, 0.3])


def test_gae_does_not_bootstrap_across_episode_ends() -> None:
    buf = RolloutBuffer(3, 1, 1, SPEC)
    buf.rewards[:, 0, 0] = [1.0, 5.0, 2.0]
    buf.values[:, 0, 0] = [0.0, 0.0, 0.0, 100.0]  # huge bootstrap must be ignored after done
    buf.dones[:, 0] = [0.0, 1.0, 1.0]
    buf.compute_gae(1.0, 1.0)
    np.testing.assert_allclose(buf.advantages[:, 0, 0], [6.0, 5.0, 2.0])


def test_value_norm_round_trip_and_statistics() -> None:
    vn = ValueNorm(enabled=True)
    data = torch.randn(5000) * 3 + 10
    for chunk in data.split(500):
        vn.update(chunk)
    assert vn.mean.item() == pytest.approx(data.mean().item(), abs=1e-3)
    assert vn.var.item() == pytest.approx(data.var(unbiased=False).item(), rel=1e-3)
    x = torch.randn(10)
    assert torch.allclose(vn.denormalize(vn.normalize(x)), x, atol=1e-4)
    off = ValueNorm(enabled=False)
    assert torch.equal(off.normalize(x), x)


def _bandit_batch(model: ActorCritic, vn: ValueNorm, n: int, good_action: int, seed: int):
    """One-state bandit: reward 1 for `good_action`, else 0; every step is an episode."""
    T, E, A = n, 1, 1
    buf = RolloutBuffer(T, E, A, SPEC)
    rng = np.random.default_rng(seed)
    grid, vec, mask, roles = _inputs(1, seed=1)
    cgrid = torch.rand(1, SPEC.critic_channels, SPEC.height, SPEC.width, generator=torch.Generator().manual_seed(2))
    cvec = torch.rand(1, SPEC.critic_vec_dim, generator=torch.Generator().manual_seed(3))
    for t in range(T):
        actions, logp = model.act(grid, vec, mask, roles)
        value = vn.denormalize(model.value(cgrid, cvec)).item()
        buf.grid[t, 0, 0], buf.vec[t, 0, 0], buf.mask[t, 0, 0] = grid[0].numpy(), vec[0].numpy(), True
        buf.role_index[t, 0, 0] = int(roles[0])
        buf.critic_grid[t, 0, 0], buf.critic_vec[t, 0, 0] = cgrid[0].numpy(), cvec[0].numpy()
        buf.alive[t, 0, 0] = 1.0
        buf.actions[t, 0, 0] = int(actions[0])
        buf.log_probs[t, 0, 0] = float(logp[0])
        buf.values[t, 0, 0] = value
        buf.rewards[t, 0, 0] = 1.0 if int(actions[0]) == good_action else 0.0
        buf.dones[t, 0] = 1.0
    buf.values[T] = 0.0
    buf.compute_gae(0.99, 0.95)
    return buf, (grid, vec, mask, roles)


def test_mappo_learns_a_one_state_bandit() -> None:
    cfg = replace(CFG.ppo, epochs=4, num_minibatches=2)
    model = _model(0)
    vn = ValueNorm(cfg.value_normalization)
    algo = MAPPO(model, cfg, vn)
    good = 2
    for it in range(40):
        buf, (grid, vec, mask, roles) = _bandit_batch(model, vn, 128, good, seed=it)
        algo.update(buf.to_batch(DEVICE), learning_rate=1e-3, entropy_coef=0.0)
    prob = model.policy(grid, vec, mask, roles).probs[0, good].item()
    assert prob > 0.9, f"policy did not learn the rewarded action (p={prob:.3f})"


def test_update_reports_finite_stats_and_value_loss_falls_on_a_fixed_batch() -> None:
    cfg = replace(CFG.ppo, epochs=1, num_minibatches=1)
    model = _model(1)
    vn = ValueNorm(True)
    algo = MAPPO(model, cfg, vn)
    buf, _ = _bandit_batch(model, vn, 64, 1, seed=0)
    batch = buf.to_batch(DEVICE)
    losses = []
    for _ in range(30):
        stats = algo.update(batch, learning_rate=1e-3, entropy_coef=0.0)
        assert all(np.isfinite(v) for v in stats.values())
        losses.append(stats["value_loss"])
    assert losses[-1] < losses[0]
    assert set(stats) >= {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"}


def test_eliminated_agents_get_no_policy_gradient_but_train_the_critic() -> None:
    model = _model(2)
    vn = ValueNorm(False)
    algo = MAPPO(model, replace(CFG.ppo, epochs=1, num_minibatches=1), vn)
    buf, _ = _bandit_batch(model, vn, 32, 0, seed=0)
    buf.alive[:] = 0.0  # everyone eliminated
    before_actor = [p.detach().clone() for p in model.actor_encoder.parameters()]
    before_critic = [p.detach().clone() for p in model.critic_encoder.parameters()]
    stats = algo.update(buf.to_batch(DEVICE), learning_rate=1e-2, entropy_coef=0.5)
    assert stats["policy_loss"] == 0.0 and stats["entropy"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before_actor, model.actor_encoder.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_critic, model.critic_encoder.parameters()))


def test_benchmark_reports_positive_timings(tmp_path) -> None:
    from rl.benchmark import measure
    from rl_helpers import DEFAULT_RL_YAML

    overrides = {
        "model.conv_channels": [4], "model.hidden_size": 16, "rollout.num_envs": 2,
        "rollout.rollout_length": 4, "rollout.num_workers": 0, "run.torch_threads": 1,
        "env.config_path": str(ROOT / "configs" / "default.yaml"),
    }
    result = measure(str(DEFAULT_RL_YAML), overrides, workers=0, updates=1)
    assert result["rollout_s"] > 0 and result["update_s"] > 0 and result["steps_per_sec"] > 0
