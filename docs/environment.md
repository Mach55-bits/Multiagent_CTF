# Multi-Agent CTF Environment Specification

Prototype 1 as implemented in `src/ctf/`.

## 1. Purpose

Lightweight testbed for multi-agent coordination and adversarial play on a discrete CTF grid. The environment is not the research contribution—strategies and experiments live outside `src/ctf/`.

## 2. Environment Overview

| Aspect | Prototype 1 behavior |
|--------|----------------------|
| Teams | **2** fixed teams: `RED` and `BLUE` |
| Agents | Default **3 per team** (`agents_per_team` configurable) |
| World | **2D discrete grid** (`width` × `height`), no continuous physics |
| Obstacles | Configurable blocked cells |
| Bases | One base cell per team (spawn / scoring reference) |
| Flags | **One flag per team**, initially at that team’s base |
| Parameters | Grid size, spawns, spawn randomization, obstacles, tagging, episode cap, visibility radius, seed — via `CTFConfig` / YAML |

The shipped demo configuration is `configs/default.yaml` (11×7 grid, center wall obstacle column, explicit spawns).

---

## 3. Game Configuration

Configuration is defined by `CTFConfig` and loaded with `load_config(path)` from YAML. Values below list **code defaults**; **`configs/default.yaml`** overrides spawns and obstacles as noted.

### 3.1 Teams

- Teams are the enum `Team.RED` and `Team.BLUE`.
- Exactly two teams are supported in Prototype 1 (not configurable beyond RED/BLUE).

### 3.2 Agents

- Count per team: `agents_per_team` (default **3**, minimum 1).
- Agent IDs are assigned deterministically: `R1…Rn` for RED, `B1…Bn` for BLUE, where `n = agents_per_team`.
- Each agent tracks: `agent_id`, `team`, position `(x, y)`, `active`, `carrying_flag`.

**Default YAML:** 3 agents per team (`R1–R3`, `B1–B3`).

### 3.3 Grid and Obstacles

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `width` | 11 | 11 |
| `height` | 7 | 7 |
| `obstacles` | `[]` | `(5,1)…(5,5)` — vertical wall at column 5 |

Validation: `width >= 1`, `height >= 1`; obstacle cells must lie inside the grid.

### 3.4 Bases and Flags

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `red_base` | `(1, 3)` | `(1, 3)` |
| `blue_base` | `(9, 3)` | `(9, 3)` |

- Each team owns one flag (`Flag.team` = owning team).
- Initial flag state: `FlagStatus.AT_BASE` at that team’s base coordinates.
- Bases must be inside the grid and **not** on obstacles.

There is no separate “dropped flag” state in Prototype 1—only **at base** or **carried**.

### 3.5 Spawn Configuration

Optional explicit lists: `red_spawns`, `blue_spawns`.

- If provided, each list must contain **exactly** `agents_per_team` positions.
- Each spawn must be in bounds and not on an obstacle.

If **not** provided, the environment generates **exactly** `agents_per_team` spawn cells per team: traversable cells sorted by Manhattan distance from that team’s base (then `y`, then `x`), taking the closest cells first. Configuration fails if not enough traversable cells exist.

**Default YAML** (explicit):

- RED: `(1,2)`, `(1,3)`, `(1,4)`
- BLUE: `(9,2)`, `(9,3)`, `(9,4)`

**Spawn randomization** (`spawn_mode`, applied at every `reset()` using the environment's seeded RNG):

| `spawn_mode` | Behavior |
|--------------|----------|
| `fixed` (default) | Use the spawn list above (or the cells nearest the base). Identical every episode. |
| `shuffle` | Same cells, randomly re-assigned among the team's agents. |
| `near_base` | Distinct random cells within `spawn_radius` (Manhattan) of the base, restricted to the team's home zone (strictly closer to own base than to the enemy base). `spawn_allow_base` controls whether the flag cell itself is allowed. Ignores explicit spawn lists. |

Construction fails if `near_base` cannot find `agents_per_team` valid cells.

### 3.6 Tagging Configuration

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `tagging_enabled` | `true` | `true` |

When enabled, an opposing active agent on the same cell as a flag carrier tags the carrier (see §7.4). The value is copied into `GameState.tagging_enabled` at reset.

### 3.7 Episode Length

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `max_episode_steps` | 200 | 200 |
| `seed` | 42 | 42 |

If `timestep` reaches `max_episode_steps` without a win, the episode ends with `truncation=True`, `winner=None`.

### 3.8 Visibility

| Parameter | Default (code) | `configs/default.yaml` |
|-----------|----------------|-------------------------|
| `visibility_radius` | 3 | 3 |

Manhattan radius used by `ObservationMode.PARTIAL` (see §6). Copied into `GameState.visibility_radius` at reset; `observe(..., visibility_radius=r)` can override it per call.

### 3.9 Defaults and strict parsing

Defaults are declared once, in the `CTFConfig` dataclass. YAML files only override them. `load_config` / `config_from_dict` reject unknown keys and wrongly typed values (e.g. `width: "11"`), so a misspelled parameter fails loudly instead of silently using a default. `config_to_dict` returns a YAML-serializable copy for logging the exact configuration of a run.

---

## 4. Action Space

Each **active** agent selects one of five actions (`Action` enum):

| Action | Grid delta |
|--------|------------|
| `UP` | `(0, -1)` |
| `DOWN` | `(0, +1)` |
| `LEFT` | `(-1, 0)` |
| `RIGHT` | `(+1, 0)` |
| `STAY` | `(0, 0)` |

No diagonal movement.

**Movement validation (per agent, from pre-step position):**

- `STAY` is always valid for active agents.
- A move is **rejected** (agent remains in place) if the target cell is out of bounds or an obstacle.
- Rejected moves are recorded in `info["rejected_moves"]` (agent id → action name string).

**Agent–agent collision:** Movement does **not** block on other agents. Multiple agents may occupy the same cell.

**Inactive agents:** Do not require an entry in the action dict; they do not move.

**Action dict requirements for `step()`:**

- Every **active** agent must have an action.
- Unknown agent id → `KeyError`.
- Invalid action string → `ValueError` from `Action(...)`.

---

## 5. State Representation

`GameState` is the full simulator state.

| Field | Meaning |
|-------|---------|
| `grid` | `GridMap`: `width`, `height`, `obstacles` (frozen set) |
| `agents` | `dict[str, Agent]` |
| `flags` | `dict[Team, Flag]` |
| `red_base`, `blue_base` | Base coordinates |
| `timestep` | Integer step counter (incremented once per successful `step()` call) |
| `winner` | `Team` or `None` |
| `terminated` | Episode ended (win or max steps) |
| `truncation` | `True` if ended due to step limit without winner |
| `tagging_enabled` | Snapshot of tagging setting for this episode |

**Agent** (`Agent`): `agent_id`, `team`, `x`, `y`, `active`, `carrying_flag`.

**Flag** (`Flag`): `team` (owner), `status` (`AT_BASE` | `CARRIED`), `x`, `y`, `carrier_id` (when carried).

---

## 6. Observation Model

Two modes are implemented.

**`ObservationMode.GLOBAL`** returns `GlobalObservation`:

- Grid: `width`, `height`, `traversable` (height×width bool matrix), `obstacles`
- `red_base`, `blue_base`
- Full `agents` and `flags` maps
- `timestep`, `winner`, `terminated`, `truncation`, `tagging_enabled`, `visibility_radius`
- `to_dict()` for serialization

**`ObservationMode.PARTIAL`** (requires `team=`) returns `PartialTeamObservation`, the team-level fog-of-war view:

- Friendly agents (including eliminated ones) are always fully visible.
- An enemy agent is visible only if it is active and within `visibility_radius` (Manhattan) of an **active** friendly agent. Eliminated agents see nothing.
- The enemy flag is visible under the same rule; the team's own flag is always visible.
- Because the view is the union over all active teammates, whatever one agent (e.g. a scout) sees, the whole team sees.

---

## 7. Game Mechanics

### 7.1 Movement

All active agents’ moves are proposed from the **same pre-step** positions, then applied together. Inactive agents stay fixed.

### 7.2 Flag Pickup

After movement, an **active** agent that is **not** already carrying a flag may pick up the **enemy** flag if:

- Enemy flag status is `AT_BASE`, and
- Agent shares the flag’s cell.

Effects: `agent.carrying_flag = True`; enemy flag becomes `CARRIED` with `carrier_id` set. A `flag_pickup` event is emitted.

Agents are processed in dict iteration order (insertion order: `R1…Rn`, then `B1…Bn`). Only one pickup per flag per step is possible because after the first pickup the flag is no longer `AT_BASE`.

### 7.3 Flag Carrying

While `CARRIED`, the flag’s `(x, y)` follows the carrier after movement (`_sync_carried_flag_positions`). The agent’s `carrying_flag` indicates it holds the **opponent’s** flag (only enemy pickup is implemented).

### 7.4 Tagging

When `tagging_enabled` is true: for each active carrier, if any **opposing** active agent shares the cell, the carrier is tagged:

- Carrier: `active=False`, `carrying_flag=False`
- A `tag` event is emitted (`agent_id`, `by_agent_id`, `flag_team` = the flag that was being carried — i.e. the **enemy team’s** flag identifier)

Carriers are scanned in agent dict order; the first opposing agent in dict order that shares the cell is recorded as `by_agent_id`.

Teammates on the same cell do **not** cause tagging.

### 7.5 Flag Reset

When a carrier is tagged, the carried enemy flag returns to **`AT_BASE`** at its **original base coordinates** (from config).

### 7.6 Flag Return / Winning

After tagging, an **active** agent with `carrying_flag=True` on **its own base** cell wins:

- `winner` = agent’s team
- `terminated` = `True`
- Events: `flag_capture`, then `episode_end` with `reason="flag_capture"` and `winner` set to the team name string.

“Capture” in the win sense means **returning the enemy flag to your base**, not merely touching the enemy flag.

### 7.7 No Respawn

Tagged (inactive) agents remain inactive for the rest of the episode. No respawn mechanic exists.

---

## 8. Step Execution Order

Each call to `step()` applies this **fixed** sequence:

1. **Parse actions** — validate ids and required active-agent actions.
2. **Propose and apply movement** — simultaneous intent from pre-step positions.
3. **Synchronize carried flag positions** — update coordinates for flags already carried.
4. **Flag pickup** — enemy flag at base, same cell.
5. **Tagging** — if enabled.
6. **Scoring / win check** — carrier on own base.
7. **Timestep increment** — `timestep += 1`.
8. **Maximum-step truncation** — if `timestep >= max_episode_steps` and not already terminated: set `truncation=True`, `terminated=True`, `winner=None`, emit `episode_end` with `reason="max_steps"`.

**Ordering implications:**

- **Pickup → tag → score** on the same step.
- A carrier who is tagged in phase 5 cannot score in phase 6 that same step.
- Example: pickup on the enemy base with a defender on the same cell can produce pickup then tag before any win check.

Co-location of non-carriers is allowed throughout; only tagging special-cases carriers vs opponents.

---

## 9. Configuration and Reproducibility

- **Loading:** `load_config("configs/default.yaml")` or construct `CTFConfig(...)` in code.
- **Validation:** On construction, `CTFConfig` checks grid dimensions, episode length, agent count, base/spawn bounds, bases/spawns not on obstacles, explicit spawn counts, and feasibility of auto-generated spawns.
- **Seed:** the environment owns a private `random.Random`, created from `CTFConfig.seed`. `reset(seed=…)` reseeds it; `reset()` without a seed *continues* the stream, so consecutive episodes differ (when `spawn_mode` is random) while the whole sequence stays reproducible from `config.seed`.
- **Determinism:** `step()` never consumes RNG state. Given the same spawns and actions, transitions are identical. With `spawn_mode: fixed` (the default) every episode starts identically, so repeated matches between deterministic strategies are identical too; use `shuffle` or `near_base` to get varied, seeded episodes.
- The RNG is environment-local; the global `random` module is never used.

---

## 10. Environment API

Main class: `CTFEnvironment(config: CTFConfig)`.

| Member | Behavior |
|--------|----------|
| `reset(*, seed=None)` | New episode; returns **snapshot** `GameState`. Samples spawns from the internal RNG; `seed` reseeds it first. |
| `step(actions)` | One transition; returns `(state, rewards, done, info)`. `state` is a snapshot. Raises if not reset or already terminated. |
| `observe(mode=GLOBAL, *, team=None, visibility_radius=None)` | Builds a `GlobalObservation`, or a `PartialTeamObservation` for `mode=PARTIAL` (needs `team`). Radius defaults to `config.visibility_radius`. |
| `state` | **Snapshot** copy of current state (mutating it does not affect the simulator). |
| `done` | `True` when internal `terminated` is set. |
| `config` | The `CTFConfig` instance. |

**`step(actions)`** expects `dict[str, Action | str]` keyed by agent id.

**Returns:**

- `rewards`: one float per agent id; **always `0.0` in Prototype 1**.
- `done`: same as `state.terminated` after the step.
- `info`: see §11.

Typical loop:

```python
env = CTFEnvironment(load_config("configs/default.yaml"))
state = env.reset()
while not env.done:
    actions = {aid: Action.STAY for aid in env.config.agent_ids()}  # example
    state, rewards, done, info = env.step(actions)
```

External controllers must supply actions for all **active** agents each step.

---

## 11. Event Reporting

`info` on each step contains:

### `rejected_moves`

`dict[str, str]`: agent id → rejected action name (empty dict if none).

### `events`

`list[dict]`: structured records for **this step only**.

| `type` | Fields | When |
|--------|--------|------|
| `flag_pickup` | `agent_id`, `flag_team` | Enemy flag picked up at base |
| `tag` | `agent_id`, `by_agent_id`, `flag_team` | Carrier tagged |
| `flag_capture` | `agent_id`, `flag_team` | Win condition (enemy flag returned to own base) |
| `episode_end` | `winner` (`str` or `null`), `reason` | `"flag_capture"` or `"max_steps"` |

Win steps may emit both `flag_capture` and `episode_end` in the same list.

---

## 12. Visualization

A **separate** Pygame layer lives under `visualization/` (not in `src/ctf/`):

- `visualization/renderer.py` — draws a `GameState` snapshot (grid, bases, obstacles, flags, agents).
- `visualization/demo.py` — stepped animation of the shared demo script in `scripts/demo_scenario.py`.

The visualizer:

- Uses the public API (`reset`, `step`, `state`, `info["events"]`).
- Does **not** implement movement, flags, tagging, or termination rules.
- Does **not** implement AI or pathfinding; it replays a **predefined action sequence** (same scenario as `scripts/run_demo.py`).

Launch: `uv run python visualization/demo.py` (requires `pygame` dependency).

---

## 13. Extensibility

Control loop: `observation → strategy → actions → environment → next state`. Strategy code stays outside `src/ctf/`; rewards and extra observation modes can be layered on without changing current step logic.

---

## 14. Current Scope and Exclusions

**Not implemented in Prototype 1** (may be added later only if the finalized methodology requires):

- AI / hand-crafted strategies (beyond external scripts)
- Reinforcement learning or MARL training loops
- Built-in opponent AI
- A*, Minimax, MCTS, or other planners inside the env
- Networking or multiplayer
- Sophisticated combat beyond tagging
- Dropped flags, respawn, or extra game modes
- Advanced graphics or game UI (basic Pygame demo only)
- Non-zero reward shaping or experiment runners in the core package

---

## 15. Prototype 1 Verification

Documented verification status (implementation / QA, not research results):

| Check | Status |
|-------|--------|
| Automated tests | `uv run pytest` (all passing) |
| Deterministic text demo | `scripts/run_demo.py` — RED wins, `reason=flag_capture` |
| Flag pickup and return | Covered by tests and end-to-end demo |
| Episode termination | Win and max-step truncation tested |
| Visualization | Pygame demo runs shared scenario; environment unchanged |

No performance benchmarks or coordination-strategy research claims are made at this stage.
