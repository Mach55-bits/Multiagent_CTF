# Multi-Agent CTF (Prototype 1)

Lightweight, deterministic **Capture the Flag** environment for multi-agent coordination research. Prototype 1 implements only the game mechanics—no AI, RL, pathfinding, or graphics.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python 3.11+)

## Setup

From the project root:

```bash
uv sync --extra dev
```

This installs the `ctf` package and development dependencies (pytest).

## Run tests

```bash
uv run pytest
```

## Run demo

```bash
uv run python scripts/run_demo.py
```

The demo loads `configs/default.yaml`, resets the environment, applies a short scripted move sequence, and prints agent positions and episode status.

## Project layout

- `src/ctf/` — environment implementation
- `configs/default.yaml` — default grid, spawns, and rules
- `tests/` — deterministic unit tests
- `docs/environment.md` — rules and API notes

## Quick usage

```python
from ctf import Action, CTFEnvironment, load_config

env = CTFEnvironment(load_config("configs/default.yaml"))
state = env.reset()

while not env.done:
    actions = {aid: Action.STAY for aid in env.config.agent_ids()}
    state, rewards, done, info = env.step(actions)
```

See `docs/environment.md` for full details.
