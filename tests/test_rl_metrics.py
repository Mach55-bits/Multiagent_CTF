import json

import numpy as np
import pytest

from rl.metrics import MetricsLogger, read_metrics, summarize_episodes


def _info(outcome, opponent="stay", team="RED", ret=(1.0, 2.0, 3.0)):
    return {
        "outcome": outcome, "length": 10, "learner_team": team, "opponent": opponent,
        "pickups": 1, "tags_inflicted": 0, "tags_suffered": 1, "phase": "p", "fixed_spawns": False,
        "return": list(ret), "components": {"outcome": [1.0, 1.0, 1.0], "step": [-0.1, -0.1, -0.1]},
    }


def test_summary_aggregates_rates_returns_and_breakdowns() -> None:
    infos = [_info("win"), _info("loss", opponent="random", team="BLUE"), _info("draw"), _info("win")]
    s = summarize_episodes(infos, ["attacker", "defender", "scout"])
    assert s["episodes"] == 4 and s["win_rate"] == 0.5 and s["loss_rate"] == 0.25 and s["draw_rate"] == 0.25
    assert s["return/slot0_attacker"] == 1.0 and s["return/slot2_scout"] == 3.0
    assert s["win_rate/opp_stay"] == pytest.approx(2 / 3) and s["win_rate/opp_random"] == 0.0
    assert s["win_rate/side_red"] == pytest.approx(2 / 3) and s["win_rate/side_blue"] == 0.0
    assert s["reward/outcome"] == pytest.approx(3.0) and s["reward/step"] == pytest.approx(-0.3)
    assert summarize_episodes([], ["a"]) == {}


def test_logger_writes_jsonl_events_and_cleans_non_finite_values(tmp_path) -> None:
    logger = MetricsLogger(tmp_path, tensorboard=False)
    logger.log(100, {"a": 1.5, "bad": float("nan"), "np": np.float32(2.0)})
    logger.event(200, "phase_advance", phase="x")
    logger.close()
    rows = read_metrics(tmp_path / "metrics.jsonl")
    assert rows[0]["step"] == 100 and rows[0]["a"] == 1.5 and rows[0]["bad"] is None and rows[0]["np"] == 2.0
    assert rows[1]["event"] == "phase_advance" and rows[1]["phase"] == "x"


def test_logger_appends_across_runs_and_reader_ignores_a_torn_last_line(tmp_path) -> None:
    for step in (1, 2):
        logger = MetricsLogger(tmp_path, tensorboard=False)
        logger.log(step, {"v": step})
        logger.close()
    with (tmp_path / "metrics.jsonl").open("a") as f:
        f.write('{"step": 3, "v"')  # half-written line (training still running)
    assert [r["step"] for r in read_metrics(tmp_path / "metrics.jsonl")] == [1, 2]
    json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
