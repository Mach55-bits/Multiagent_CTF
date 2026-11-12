import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from rl.checkpoint import (  # noqa: E402
    CheckpointError,
    CheckpointManager,
    load_checkpoint,
    load_policy,
    restore_training_state,
    save_checkpoint,
)
from rl.config import load_base_env_config  # noqa: E402
from rl.trainer import Trainer  # noqa: E402
from rl_helpers import ROOT, tiny_config  # noqa: E402


def _trainer(tmp_path, **kw) -> Trainer:
    cfg = tiny_config(str(tmp_path), **kw)
    return Trainer(cfg, load_base_env_config(cfg, ROOT), tmp_path / "run")


def _logits(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    spec = model.spec
    grid = torch.rand(3, spec.actor_channels, spec.height, spec.width, generator=g)
    vec = torch.rand(3, spec.actor_vec_dim, generator=g)
    mask = torch.ones(3, spec.num_actions, dtype=torch.bool)
    return model.policy(grid, vec, mask, torch.tensor([0, 1, 2])).logits


def test_round_trip_restores_everything(tmp_path) -> None:
    a = _trainer(tmp_path)
    a.counters.update(total_steps=123, update=7)
    a.curriculum.record(["win", "loss"], 20)
    a.value_norm.update(torch.randn(100) * 5 + 3)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, **a._state())
    assert not list(tmp_path.glob("*.tmp"))

    b = _trainer(tmp_path / "other")
    counters = restore_training_state(
        load_checkpoint(path), model=b.model, optimizer=b.algo.optimizer,
        value_norm=b.value_norm, curriculum=b.curriculum,
    )
    assert counters == {"total_steps": 123, "update": 7}
    assert torch.equal(_logits(a.model), _logits(b.model))
    assert torch.equal(a.value_norm.mean, b.value_norm.mean) and torch.equal(a.value_norm.var, b.value_norm.var)
    assert b.curriculum.state_dict() == a.curriculum.state_dict()


def test_observation_layout_mismatch_is_refused(tmp_path) -> None:
    t = _trainer(tmp_path)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, **t._state())
    payload = torch.load(path, weights_only=True)
    payload["obs_spec"]["actor_channels"] = payload["obs_spec"]["actor_channels"][:-1]
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="observation layout"):
        load_checkpoint(path)
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")


def test_model_size_mismatch_is_reported_clearly(tmp_path) -> None:
    t = _trainer(tmp_path)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, **t._state())
    bigger = _trainer(tmp_path / "big")
    bigger_cfg = tiny_config(str(tmp_path / "big2"))
    from dataclasses import replace
    bigger_cfg = replace(bigger_cfg, model=replace(bigger_cfg.model, hidden_size=32))
    other = Trainer(bigger_cfg, load_base_env_config(bigger_cfg, ROOT), tmp_path / "run3")
    with pytest.raises(CheckpointError, match="do not fit"):
        restore_training_state(load_checkpoint(path), model=other.model, optimizer=other.algo.optimizer,
                               value_norm=other.value_norm, curriculum=other.curriculum)
    del bigger


def test_load_policy_rebuilds_the_network_for_inference(tmp_path) -> None:
    t = _trainer(tmp_path)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, **t._state())
    model, meta = load_policy(path)
    assert not model.training
    assert torch.equal(_logits(model), _logits(t.model))
    assert meta["env_config"].width == 11 and meta["config"].model.hidden_size == 16
    assert isinstance(meta["git_commit"], str)


def test_manager_prunes_periodic_checkpoints_but_keeps_phase_snapshots(tmp_path) -> None:
    t = _trainer(tmp_path)
    manager = CheckpointManager(tmp_path / "run2", keep=2)
    manager.save_phase_start(1, "next", **t._state())
    for step in (100, 200, 300, 400):
        manager.save_periodic(step, **t._state())
    names = sorted(p.name for p in manager.dir.glob("*.pt"))
    assert names == ["latest.pt", "phase_1_next_start.pt", "step_0000000300.pt", "step_0000000400.pt"]
    assert load_checkpoint(manager.latest)["counters"] == t.counters
    keep_all = CheckpointManager(tmp_path / "run3", keep=0)
    for step in (1, 2, 3):
        keep_all.save_periodic(step, **t._state())
    assert len(list(keep_all.dir.glob("step_*.pt"))) == 3
    assert np.isfinite(float(t.value_norm.var))
