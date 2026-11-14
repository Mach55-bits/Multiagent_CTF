# LLM Commander (Approach 3)

An LLM acts as a coach. Every few turns it reads what the team knows and gives short high-level orders
("attacker, capture the flag", "defender, block the gap at [5, 0]"). A low-level controller turns those orders
into the game's UP/DOWN/LEFT/RIGHT/STAY moves. Illegal orders are rejected before they reach the players.

## Setup
Work on a branch that contains `main`, `classical_ai` and `rl_agent` (the commander uses their opponents,
pathfinding and evaluator):

    git checkout llm_approach
    git merge origin/main origin/classical_ai origin/rl_agent
    uv sync --extra dev --extra rl        # the rl extra (NumPy, PyTorch) is needed for the RL agent and the RL tests

## Run
    uv run python scripts/run_llm_commander.py                              # offline mock LLM vs 'balanced'
    uv run python scripts/run_llm_commander.py --condition classical        # Team 1's classical AI, no LLM
    uv run python scripts/run_llm_commander.py --condition llm_classical    # LLM orders on Team 1's agents
    uv run python scripts/run_llm_commander.py --partial --radius 3         # fog of war
    uv run pytest                                                           # all tests (RL tests need the rl extra)
    uv run python scripts/run_llm_commander.py --condition llm_rl --rl-checkpoint untrained   # LLM + RL agent (random weights, plumbing test)

### With Gemini
1. Create a key in Google AI Studio (aistudio.google.com) and set it:
   Windows PowerShell `$env:GEMINI_API_KEY="your_key"`, Mac/Linux `export GEMINI_API_KEY=your_key`.
2. Check key and model, and see one real decision:
       uv run --with google-genai python scripts/check_llm.py
   If the model name is rejected, pass one from AI Studio: `--model <name>` (or set GEMINI_MODEL).
3. Play a game:
       uv run --with google-genai python scripts/run_llm_commander.py --llm gemini --opponent classical --cache logs/cache.json
Choosing the model: `--model <name>` or the GEMINI_MODEL environment variable. Use a normal text model
(for example `gemini-3.8-flash`); "Live" models (`gemini-3.8-live`) are for real-time voice sessions and
are not suitable for this program. `GEMINI_THINKING_LEVEL` (low / medium / high) lowers the model's
reasoning effort, which makes requests faster; settings a model rejects (such as temperature) are dropped
automatically.
Free tiers limit requests per minute and per day. `--rpm` (default 10) spaces requests out, `--cache` stores
answers so repeated games cost nothing, and `--k` (ask every K turns) controls how many requests a game needs
(about 200/K + events).

## Experiments
    uv run python scripts/run_experiments.py --dry-run                      # plan + estimated LLM requests
    uv run python scripts/run_experiments.py --llm mock --games 50          # all-mock run to test the pipeline
    uv run --with google-genai python scripts/run_experiments.py --llm gemini --games 10 \
        --conditions llm,classical,random_commander --opponents balanced,classical --cache logs/cache.json
Every pairing (condition x opponent x observability) is played with the same seeds and with sides swapped.
Games are appended to `results/games.csv` as they finish (an interrupted run resumes when repeated);
`results/summary.csv` has win rate with a 95% confidence interval, turns, flag pickups, tags, moves,
illegal-order rate, LLM time and tokens. Spawns are randomised (`--spawn near_base`) so games differ; with
`--spawn fixed` every game starts identically.

| Condition | Meaning |
|---|---|
| `llm` | LLM orders + simple controller (main condition) |
| `llm_classical` | LLM orders executed by Team 1's A* / role logic (mixed team) |
| `llm_rl` | LLM orders for the agents it picks, Team 2's trained RL policy plays the rest (mixed team) |
| `llm_no_validator` | ablation: orders executed without the safety check |
| `llm_no_map` | ablation: prompt without the ASCII map |
| `llm_no_scout_share` | ablation: scout observations not shared (partial observability only) |
| `random_commander` | control: random but legal orders |
| `classical` | Team 1's classical AI alone (Approach 1, no commander) |
| `rl` | Team 2's trained RL policy alone (Approach 2) |
| `scripted` | role-default team on the simple controller |
Add `@K` for the trigger period, e.g. `llm@3,llm@10`.

Opponents: `aggressive`, `balanced`, `defensive`, `camper` (RL branch scripted teams), `random`, `stay`,
`classical`, `rl` (trained RL policy), `baseline`.

## The RL agent (Team 2)
The RL conditions and the `rl` opponent need a trained checkpoint. Train one with
`uv run --extra rl python -m rl.train --config configs/rl/default.yaml`; it is written to
`runs/mappo_default/checkpoints/latest.pt` (the default; override with `--rl-checkpoint` or RL_CHECKPOINT).
`--rl-checkpoint untrained` uses random weights and only checks that the plumbing works.
Check the integration with `uv run --extra rl python scripts/check_rl.py`.
How the mixed team works: the RL policy has no input for orders, so in `llm_rl` the LLM can only take over
individual agents (those with an order walk to their target); every other agent is played by the policy.
The policy was trained with fog of war, so RL-based teams always receive partial observations, also in
games that are otherwise global.

## Report material
    uv run python scripts/make_report.py                          # results/report/tables.md (+ win_rates.png with matplotlib)
    uv run python scripts/make_report.py --demo llm,classical     # seeds where the LLM won and classical did not, and the reverse
Measured stalemate / opponent-strength data for the group: docs/stalemate_findings.md.

## Files (src/strategies/llm_commander/)
- strategy.py            LLMCommanderStrategy (BaseStrategy implementation) and its options
- commander.py           when to ask the LLM, parsing, validation, retries, logging, statistics
- prompt.py              the text sent to the LLM
- view.py                observation -> what the commander knows -> text
- commands.py            command vocabulary, roles, named zones
- validator.py           safety layer
- controller.py          simple controller (orders -> moves, plus reflexes)
- classical_adapter.py   controller using Team 1's A*, influence map and role logic
- rl_adapter.py          Team 2's RL policy as a strategy and as the executor of a mixed team
- llm.py                 MockLLM, GeminiLLM, RandomOrdersLLM, CachedLLM
- conditions.py          named experimental conditions
- opponents.py           opponent factory (RL scripted teams, classical)
- runner.py              plays one game and collects metrics (uses Team 1's StrategyEvaluator)
- reporting.py           summary statistics, Markdown tables, demo/failure seed finder
- baseline.py            scripted team without an LLM

## Design choices (agree these with the group)
- Roles are not part of the game: first agent = attacker, second = defender, third = scout.
- Commands: capture_flag, defend_base, go_to, intercept, scout. Agents with no order follow role defaults.
- The commander only uses what the observation shows, so global and partial modes both work.
- The game waits while the LLM thinks (synchronous calls); decision time is recorded.
- Every prompt, raw answer, validation result, latency and token count is logged in logs/*.jsonl.

## Still to do (search for "TODO(" in the code)
1. Run with the real LLM and refine the prompt: prompt.py, llm.py, scripts/check_llm.py
5. Run the RL integration with a real trained checkpoint (scripts/check_rl.py, then the `rl` / `llm_rl` experiments): rl_adapter.py
6. Group-agreed definitions of aggressive / defensive / balanced opponents: opponents.py
7. Report text, demo scenario and demo video (tables, chart and demo seeds come from scripts/make_report.py)
8. Group decision on the base-camping stalemate (a defender on its base cell tags every pickup): controller.py, docs/stalemate_findings.md
