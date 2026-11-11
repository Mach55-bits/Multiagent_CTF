# Multi-Agent CTF (Prototype 1)

Capture-the-Flag grid environment for multi-agent coordination research, with optional fog of war and seeded spawn randomization. Core package: game rules and API only (no built-in AI). `step()` is deterministic; randomness enters only through seeded spawn modes. Optional Pygame demo for watching a scripted episode.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11+).

```bash
uv sync --extra dev          # environment + tests
uv sync --extra dev --extra rl   # also the RL dependencies (torch, tensorboard, ...)
```

`ctf`, `coordination` and `strategies` are installed as packages, so no `PYTHONPATH` tweaking is needed.

## Tests

```bash
uv run pytest
```

## Text demo

Scripted episode (RED captures BLUE flag and wins):

```bash
uv run python scripts/run_demo.py
```

## Visualization

```bash
uv run python visualization/demo.py
```

SPACE = pause/resume, ESC = quit, R = restart.

## RL agent (work in progress)

`src/rl/` holds the MAPPO agent (CNN actor-critic, role-conditioned heads, centralized critic).
Everything tunable lives in `configs/rl/default.yaml` (every key is required; override from the
command line with dotted paths). The RL tests need the extras: `uv sync --extra dev --extra rl`.

## Configuration

All parameters have a single default in `src/ctf/config.py`; YAML files override them and are validated strictly (unknown keys and wrong types are errors). See `configs/default.yaml` for every option with comments.

## Documentation

Full Prototype 1 spec: [docs/environment.md](docs/environment.md)

## Layout

- `src/ctf/` — environment
- `src/coordination/` — shared team map (scout line-of-sight sharing)
- `src/strategies/` — strategy interface (`BaseStrategy`), `Role`, dummy strategy
- `configs/default.yaml` — default map and rules
- `scripts/demo_scenario.py` — shared action script for demos
- `visualization/` — Pygame layer
- `tests/`
