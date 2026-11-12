"""Training entry point.

    python -m rl.train --config configs/rl/default.yaml
    python -m rl.train --set ppo.learning_rate=0.0001 --set run.name=lr_test
    python -m rl.train --resume latest
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import yaml

from ctf.config import config_to_dict
from rl.checkpoint import git_commit
from rl.config import config_to_plain, load_base_env_config, load_rl_config
from rl.schema import ConfigError, parse_override
from rl.trainer import Trainer


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the MAPPO Capture-the-Flag agent.")
    p.add_argument("--config", default="configs/rl/default.yaml", help="RL config YAML")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="override a config value by dotted path (repeatable)")
    p.add_argument("--resume", default=None, metavar="latest|PATH",
                   help="continue from a checkpoint ('latest' = this run's latest.pt)")
    p.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        overrides = dict(parse_override(text) for text in args.overrides)
        cfg = load_rl_config(args.config, overrides)
        env_config = load_base_env_config(cfg, root=Path.cwd())
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(yaml.safe_dump(config_to_plain(cfg), sort_keys=False))
        return 0

    run_dir = Path(cfg.run.output_dir) / cfg.run.name
    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = run_dir / "checkpoints" / "latest.pt" if args.resume == "latest" else Path(args.resume)
        if not resume_path.exists():
            print(f"cannot resume: {resume_path} does not exist", file=sys.stderr)
            return 2
    elif (run_dir / "metrics.jsonl").exists():
        print(
            f"run directory {run_dir} already has results. Pick another run.name "
            f"(--set run.name=...) or continue it with --resume latest.",
            file=sys.stderr,
        )
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if resume_path is None else f".resume_{int(time.time())}"
    (run_dir / f"config{suffix}.yaml").write_text(
        yaml.safe_dump(config_to_plain(cfg), sort_keys=False), encoding="utf-8"
    )
    (run_dir / f"env_config{suffix}.yaml").write_text(
        yaml.safe_dump(config_to_dict(env_config), sort_keys=False), encoding="utf-8"
    )
    (run_dir / f"run_info{suffix}.json").write_text(
        json.dumps({
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "argv": sys.argv,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2),
        encoding="utf-8",
    )

    print(f"run: {run_dir}  device: {cfg.run.device}  envs: {cfg.rollout.num_envs}  "
          f"workers: {cfg.rollout.num_workers}  total steps: {cfg.rollout.total_steps:,}", flush=True)
    trainer = Trainer(cfg, env_config, run_dir, resume_path)
    counters = trainer.run()
    print(f"finished at step {counters['total_steps']:,} (update {counters['update']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
