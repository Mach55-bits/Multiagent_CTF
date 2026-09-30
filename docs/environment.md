# CTF Environment (Prototype 1)

## Purpose

This package provides a **research testbed** for multi-agent Capture the Flag. Prototype 1 is the core simulator: grid world, teams, flags, tagging, and episode lifecycle. Strategies, learning algorithms, and opponents are intentionally out of scope.

Design goals:

- Deterministic, seed-controlled simulation
- Clear state and configuration objects
- Extensible observation modes (only **global** is implemented now)
- Minimal dependencies

## Game rules

### Teams and agents

- Two teams: **RED** and **BLUE**
- Default: **3 agents per team** (configurable)
- Each agent has: `agent_id`, `team`, position `(x, y)`, `active`, `carrying_flag`

Agent IDs default to `R1…Rn` and `B1…Bn`.

### World

- 2D discrete grid (`width` × `height`)
- Cells are traversable or **obstacles** (non-traversable)
- **Bases** are fixed cells for each team (RED base, BLUE base)

### Actions

Each active agent chooses one of:

| Action | Effect |
|--------|--------|
| `UP` | `y - 1` |
| `DOWN` | `y + 1` |
| `LEFT` | `x - 1` |
| `RIGHT` | `x + 1` |
| `STAY` | no movement |

No diagonal moves. **STAY** is always valid for an active agent.

Moves are rejected (agent remains in place) when:

- The target cell is outside the grid
- The target cell is an obstacle

Multiple agents may occupy the same cell; tagging applies only to flag carriers vs opponents.

Inactive agents do not require actions.

### Flags

- Each team has one flag at its base initially (`FlagStatus.AT_BASE`).
- A flag **owned by team T** sits at T’s base when not carried.
- An agent **captures** the opponent’s flag by ending a step on the cell where that flag is at base.
- While carried, the flag position follows the carrier (`FlagStatus.CARRIED`).
- A team **wins** when an active agent **carrying the opponent’s flag** ends a step on **its own base**.

### Tagging

When `tagging_enabled` is true:

- If an opposing **active** agent shares a cell with a **flag carrier**, the carrier is tagged.
- The carrier becomes **inactive** and stops carrying the flag.
- The opponent’s flag returns to **its original base** (`AT_BASE`).
- No respawning in Prototype 1.

When tagging is disabled, carriers are not eliminated this way.

### Episode termination

An episode ends when:

1. **Win:** a team returns the enemy flag to its own base, or
2. **Truncation:** `timestep >= max_episode_steps` with no winner (`truncation=True`, `winner=None`).

After termination, `step()` raises `RuntimeError`.

### Step order (one tick)

Each `step()` applies the following **fixed** sequence. Do not rely on a different order when designing experiments.

1. **Parse actions** — validate agent ids and required active-agent actions.
2. **Propose and apply movement** — simultaneous intent from pre-step positions; invalid moves stay in place.
3. **Sync carried flag positions** — carried flags follow their carrier’s new cell.
4. **Flag pickup** — enemy flag at base, same cell as an active non-carrier.
5. **Tagging** — opposing active agent shares a cell with a flag carrier (if tagging enabled).
6. **Scoring / win** — active carrier on own base while holding the enemy flag.
7. **Increment timestep** — `timestep` increases by one after the above.
8. **Maximum-step truncation** — if `timestep >= max_episode_steps` and the episode has not already ended, set `truncation=True`, `terminated=True`, `winner=None`.

**Ordering implications (intentional Prototype 1 behavior):**

- **Pickup before tagging:** a flag is picked up in phase 4 before tagging runs in phase 5.
- **Tagging before scoring:** a carrier tagged on the pickup step is inactive before the win check, so **they cannot score on that same step**.
- **No respawn:** tagged carriers remain inactive for the rest of the episode.
- **Flag reset on tag:** the carried enemy flag returns to **its original base** (`AT_BASE` at that team’s base coordinates).

**Co-location:** multiple agents may occupy the same cell. Movement does not block on other agents; only obstacles and grid bounds block movement. Tagging is the special case when an opponent shares a cell with a **flag carrier**.

### Step `info` events

`step()` returns `info` with at least:

- `rejected_moves` — map of agent id → action name for blocked moves.
- `events` — list of event dicts for **this step only** (empty if nothing happened).

Event types include:

| `type` | When |
|--------|------|
| `flag_pickup` | Agent picks up enemy flag at base |
| `tag` | Carrier tagged (`by_agent_id` is the tagger) |
| `flag_capture` | Win condition met (return enemy flag to own base) |
| `episode_end` | Episode ends (`reason`: `flag_capture` or `max_steps`; `winner` team name or `null`) |

## State representation

`GameState` holds:

- `grid` (`GridMap`): dimensions and obstacles
- `agents`: map of id → `Agent`
- `flags`: map of `Team` → `Flag`
- `red_base`, `blue_base`
- `timestep`, `winner`, `terminated`, `truncation`, `tagging_enabled`

Use `env.reset()` for a fresh episode. **`env.state` returns a snapshot copy** — mutating it does not change the environment’s internal state. `reset()` and `step()` also return snapshot copies.

## Action space

`Action` enum in `ctf.actions`. Pass a dict mapping **every active agent id** to an `Action` (or string name) in `env.step(actions)`.

Unknown agent IDs → `KeyError`. Missing active agent → `KeyError`.

## Observation model

`ObservationMode.GLOBAL` (default) exposes the full state via `GlobalObservation`:

- Traversability matrix and obstacle list
- All agents and flags
- Bases, timestep, winner, termination flags, tagging setting

Build with:

```python
from ctf.observations import ObservationMode, build_observation

obs = build_observation(env.state, mode=ObservationMode.GLOBAL)
```

Future modes (e.g. partial observability) can extend `ObservationMode` and `build_observation()` without changing core physics.

## Configuration

`CTFConfig` / `load_config("configs/default.yaml")`:

| Field | Meaning |
|-------|---------|
| `width`, `height` | Grid size |
| `agents_per_team` | Agents on each side |
| `tagging_enabled` | Tagging on/off |
| `max_episode_steps` | Step cap |
| `seed` | Default RNG seed |
| `red_base`, `blue_base` | Base coordinates |
| `red_spawns`, `blue_spawns` | Optional per-agent spawn cells (must list exactly `agents_per_team` cells per team) |
| `obstacles` | Blocked cells |

If spawns are omitted, exactly `agents_per_team` cells are chosen deterministically near each base (closest traversable cells by Manhattan distance). Bases and spawns must lie inside the grid and not on obstacles.

Defaults are suitable for local experiments; override in YAML or code.

## Reproducibility

- `CTFEnvironment.reset(seed=…)` reinitializes the episode RNG from that seed.
- With identical config, seed, and action sequences, simulation outcomes match (see tests).

**Prototype 1 dynamics do not consume RNG state during `step`.** The environment holds a dedicated `random.Random` instance (not the global `random` module) for **future** procedural map or spawn generation. All current transitions are driven only by configuration and submitted actions.

## Extension points

- **Observations:** add modes in `observations.py`
- **Rewards:** currently zero; `step()` already returns a per-agent reward dict
- **Policies / MARL / LLM agents:** external controllers calling `step()`
- **Opponents:** not in env; pass actions like any other agent
- **Respawn / combat:** not in Prototype 1; extend `_handle_tagging` and agent lifecycle carefully

## API summary

```python
env = CTFEnvironment(config)
state = env.reset()          # optional seed= in reset()
obs = env.observe()          # GlobalObservation
while not env.done:
    state, rewards, done, info = env.step(actions_dict)
```

`info["rejected_moves"]` maps agent id → action name for rejected movement attempts. `info["events"]` lists structured step events (see above).
