"""Episode statistics and the run's metric log (``metrics.jsonl`` + optional TensorBoard)."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def summarize_episodes(infos: list[dict], role_names: list[str]) -> dict[str, float]:
    """Aggregate finished-episode summaries (from CTFTrainEnv) into scalar metrics."""
    if not infos:
        return {}
    n = len(infos)
    out: dict[str, float] = {"episodes": float(n)}
    for outcome in ("win", "draw", "loss"):
        out[f"{outcome}_rate"] = sum(i["outcome"] == outcome for i in infos) / n
    out["episode_length"] = float(np.mean([i["length"] for i in infos]))
    for key in ("pickups", "tags_inflicted", "tags_suffered"):
        out[key] = float(np.mean([i[key] for i in infos]))
    returns = np.array([i["return"] for i in infos])
    for slot, role in enumerate(role_names):
        out[f"return/slot{slot}_{role}"] = float(returns[:, slot].mean())
    for name in infos[0]["components"]:
        out[f"reward/{name}"] = float(np.mean([sum(i["components"][name]) for i in infos]))

    by_opponent: dict[str, list[float]] = defaultdict(list)
    by_side: dict[str, list[float]] = defaultdict(list)
    for info in infos:
        won = 1.0 if info["outcome"] == "win" else 0.0
        by_opponent[info["opponent"]].append(won)
        by_side[info["learner_team"].lower()].append(won)
    for kind, wins in by_opponent.items():
        out[f"win_rate/opp_{kind}"] = sum(wins) / len(wins)
    for side, wins in by_side.items():
        out[f"win_rate/side_{side}"] = sum(wins) / len(wins)
    return out


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MetricsLogger:
    """Appends one JSON object per line; safe to tail while training runs."""

    def __init__(self, run_dir: Path, tensorboard: bool) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "metrics.jsonl"
        self._file = self.path.open("a", encoding="utf-8")
        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "run.tensorboard is true but TensorBoard is not installed; "
                    "install the rl extra (uv sync --extra rl) or set run.tensorboard: false"
                ) from exc
            self._tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def log(self, step: int, scalars: dict[str, float]) -> None:
        record = {"step": int(step), "time": time.time(), **{k: _clean(v) for k, v in scalars.items()}}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self._tb is not None:
            for key, value in scalars.items():
                value = _clean(value)
                if value is not None:
                    self._tb.add_scalar(key, value, step)

    def event(self, step: int, name: str, **fields: Any) -> None:
        record = {"step": int(step), "time": time.time(), "event": name, **fields}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        if self._tb is not None:
            self._tb.close()


def read_metrics(path: Path) -> list[dict]:
    """Read a metrics.jsonl file (ignores a trailing partially written line)."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return records
