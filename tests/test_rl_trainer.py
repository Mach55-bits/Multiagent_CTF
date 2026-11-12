import math

import pytest
import yaml

torch = pytest.importorskip("torch")

from dataclasses import replace  # noqa: E402

from rl.checkpoint import load_checkpoint  # noqa: E402
from rl.config import config_to_plain, load_base_env_config  # noqa: E402
from rl.metrics import read_metrics  # noqa: E402
from rl.train import main  # noqa: E402
from rl.trainer import Trainer  # noqa: E402
from rl_helpers import ROOT, tiny_config  # noqa: E402


def _run(cfg, run_dir, resume=None):
    trainer = Trainer(cfg, load_base_env_config(cfg, ROOT), run_dir, resume)
    return trainer, trainer.run()


def test_training_runs_logs_and_checkpoints(tmp_path) -> None:
    cfg = tiny_config(str(tmp_path))  # 3 updates of 2 envs x 8 steps
    trainer, counters = _run(cfg, tmp_path / "run")
    assert counters == {"total_steps": 48, "update": 3}
    rows = [r for r in read_metrics(tmp_path / "run" / "metrics.jsonl") if "event" not in r]
    assert [r["step"] for r in rows] == [16, 32, 48]
    for key in ("train/policy_loss", "train/value_loss", "train/entropy", "perf/steps_per_sec",
                "schedule/learning_rate", "curriculum/phase"):
        assert key in rows[0]
        assert all(r[key] is None or math.isfinite(r[key]) for r in rows)
    assert rows[0]["schedule/learning_rate"] > rows[-1]["schedule/learning_rate"]  # linear decay
    assert (tmp_path / "run" / "checkpoints" / "latest.pt").exists()
    assert load_checkpoint(tmp_path / "run" / "checkpoints" / "latest.pt")["counters"]["total_steps"] == 48
    assert trainer.counters["update"] == 3


def test_resume_continues_the_same_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    first, _ = _run(tiny_config(str(tmp_path), total_steps=32), run_dir)
    latest = run_dir / "checkpoints" / "latest.pt"
    saved = load_checkpoint(latest)
    longer = tiny_config(str(tmp_path), total_steps=64)
    second = Trainer(longer, load_base_env_config(longer, ROOT), run_dir, latest)
    assert second.counters == saved["counters"] == {"total_steps": 32, "update": 2}
    for p, q in zip(first.model.parameters(), second.model.parameters()):
        assert torch.equal(p, q)  # weights restored before any new training
    assert second.run() == {"total_steps": 64, "update": 4}
    steps = [r["step"] for r in read_metrics(run_dir / "metrics.jsonl") if "event" not in r]
    assert steps == [16, 32, 48, 64]


def test_phase_advance_is_logged_and_snapshotted(tmp_path) -> None:
    cfg = tiny_config(str(tmp_path), phases=2, first_phase_max_steps=16)
    trainer, _ = _run(cfg, tmp_path / "run")
    assert trainer.curriculum.index == 1
    events = [r for r in read_metrics(tmp_path / "run" / "metrics.jsonl") if r.get("event") == "phase_advance"]
    assert len(events) == 1 and events[0]["phase_index"] == 1
    assert list((tmp_path / "run" / "checkpoints").glob("phase_1_*_start.pt"))


def test_training_with_worker_processes(tmp_path) -> None:
    cfg = tiny_config(str(tmp_path), num_envs=2, num_workers=2, total_steps=32)
    _, counters = _run(cfg, tmp_path / "run")
    assert counters["total_steps"] == 32


def test_cli_runs_refuses_to_overwrite_and_resumes(tmp_path, capsys, monkeypatch) -> None:
    cfg = tiny_config(str(tmp_path), total_steps=32)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config_to_plain(cfg)))
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(path)]) == 0
    run_dir = tmp_path / "t"
    assert (run_dir / "config.yaml").exists() and (run_dir / "env_config.yaml").exists()
    assert (run_dir / "run_info.json").exists()
    assert main(["--config", str(path)]) == 2  # same run name again
    assert "already has results" in capsys.readouterr().err
    assert main(["--config", str(path), "--set", "rollout.total_steps=48", "--resume", "latest"]) == 0
    assert load_checkpoint(run_dir / "checkpoints" / "latest.pt")["counters"]["total_steps"] == 48
    assert main(["--config", str(path), "--set", "ppo.nonsense=1"]) == 2
    assert main(["--config", str(path), "--resume", "nope.pt", "--set", "run.name=other"]) == 2
