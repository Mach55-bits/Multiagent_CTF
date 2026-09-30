# Multi-Agent CTF (Prototype 1)

Deterministic Capture-the-Flag grid environment for multi-agent coordination research. Core package: game rules and API only (no built-in AI). Optional Pygame demo for watching a scripted episode.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11+).

```bash
uv sync --extra dev
```

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

## Documentation

Full Prototype 1 spec: [docs/environment.md](docs/environment.md)

## Layout

- `src/ctf/` — environment
- `configs/default.yaml` — default map and rules
- `scripts/demo_scenario.py` — shared action script for demos
- `visualization/` — Pygame layer
- `tests/`
