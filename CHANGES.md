# RL-prep changes to `main`

Base: `main` (bd1e8bf). `classical_ai` is **not** involved. Result: 84 tests pass (55 original + 29 new), from any working directory.

## Apply

```bash
git checkout -b rl_agent main
git rm -r -q --cached $(git ls-files '*.pyc')   # .pyc files are committed on main; .gitignore now covers them
git apply rl_prep.patch                          # or unzip rl_prep_files.zip over the repo
uv sync --extra dev
uv run pytest
git add -A && git commit -m "RL prep: config-driven env, seeded spawns, strict config"
```

## What changed and why

| File | Change | Why |
|---|---|---|
| `src/ctf/config.py` | Defaults declared once in `CTFConfig`; YAML only overrides. Strict parsing (unknown keys and wrong types raise). New fields `visibility_radius`, `spawn_mode`, `spawn_radius`, `spawn_allow_base`. New `sample_spawns()`, `config_to_dict()` (lossless round trip, for logging a run's exact config). | The old `config_from_dict` duplicated every default and silently ignored typos. |
| `src/ctf/environment.py` | `reset(seed)` now actually drives randomness: spawns are sampled from the env's private RNG. `reset()` without a seed continues the stream (reproducible from `config.seed`). `observe()` defaults to `config.visibility_radius`. | The old seed did nothing, so every episode was identical. |
| `src/ctf/game_state.py` | `initial(config, red_spawns=, blue_spawns=)`; carries `visibility_radius`; `copy()` via `dataclasses.replace`. | Needed for per-episode spawns. |
| `src/ctf/observations.py` | Radius comes from state/config instead of a module constant. `GlobalObservation` also carries the radius. Eliminated agents no longer provide vision. `manhattan_cells` cached. Duplicate `to_dict` code merged. | Removes a hardcoded 3, fixes a rule bug, speeds up fog-of-war. |
| `src/coordination/shared_map.py` | Eliminated agents no longer update the shared map. | Same fix, shared-map side. |
| `src/ctf/agent.py` | `with_*` methods use `dataclasses.replace`. | Less code, same API. |
| `src/strategies/base.py` | Optional `reset()` hook (no-op by default). | A strategy instance can be reused across episodes without leaking memory. |
| `src/strategies/roles.py` (new) | `Role` enum (attacker / defender / scout). | One shared definition for every paradigm. |
| `configs/default.yaml` | Every option listed and commented. | Config is self-documenting. |
| `pyproject.toml` | `rl` extra (numpy, torch, tensorboard, matplotlib). Packages `ctf`, `coordination`, `strategies` are installed properly, so no `PYTHONPATH` or `sys.path` hacks. | Reproducible setup. |
| `.gitignore` (new) | Python caches, `runs/`, `checkpoints/`, `outputs/`. | Keeps experiment output out of git. |
| `README.md`, `docs/environment.md` | Updated: fog of war, spawn modes, seeding, visibility, strict config. | The docs said "GLOBAL only", "30 tests" and "reset reseeds". |
| `tests/` | `test_config_strict.py`, `test_spawn_modes.py`, `test_visibility_config.py`. | Covers every change above. |

Measured: step + 2 observations + 2 shared-map updates went from ~5.7k to ~7.7k steps/s (about +34%).

## Behavior changes you should know about

1. **Eliminated agents no longer see.** Before, a dead agent still revealed enemies and updated the shared map. This is a rule change, so it must reach `main` before final evaluation so every approach plays by the same rules.
2. **`reset()` without a seed continues the RNG stream** instead of reseeding. With `spawn_mode: fixed` (the default) nothing changes.
3. **Default behavior is otherwise identical:** `spawn_mode: fixed`, radius 3, same spawns, same rules.

## Compatibility with `classical_ai` (not modified)

I ran the teammate's unmodified `classical_ai` code on top of these changes. It imports and runs. `DEFAULT_PARTIAL_VISIBILITY_RADIUS` is kept as an alias, so their `from ctf.observations import ...` still works.

- Global view: classical vs classical is unchanged (BLUE wins at step 29).
- Partial view: it was a 200-step draw and is now a RED win at step 43, because of change 1 above. Their partial-view baselines should be re-run after merging.

Notes for whoever owns `classical_ai` (not touched here):

- `ClassicalStrategy` has no reset, so its shared map leaks between episodes if an instance is reused. This produced a misleading 18-2 win split in my first measurements.
- Default roles are computed from *active* agents, so roles shift when an agent dies.
- Literals that could move to config: `visibility_radius=3` in the scout, `algorithm="astar"` in the roles, and the influence-map defaults.
- `decide_action` rebuilds the behavior tree on every call.

## Policy for the RL code (next step)

No literals in code. Every hyperparameter, reward weight, path, seed and curriculum threshold lives in a YAML file validated the same strict way, and each run saves its resolved config next to its checkpoints.
