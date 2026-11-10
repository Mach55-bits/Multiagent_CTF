# LLM Commander (Approach 3)

An LLM acts as a coach. Every few turns it reads what the team knows, and gives short high-level orders
("attacker, capture the flag", "defender, block the gap at [5, 0]"). A simple low-level controller turns
those orders into the game's UP/DOWN/LEFT/RIGHT/STAY moves. Illegal orders are rejected before they reach the players.

## Run it (from the repository root)
    uv sync --extra dev
    uv run python scripts/run_llm_commander.py                  # offline mock LLM, balanced scripted opponent
    uv run python scripts/run_llm_commander.py --no-commander   # baseline without any LLM
    uv run python scripts/run_llm_commander.py --partial        # fog of war
    uv run pytest                                               # all tests

With the real LLM (needs an API key in the ANTHROPIC_API_KEY environment variable):
    uv run --with anthropic python scripts/run_llm_commander.py --llm claude --cache logs/cache.json

Useful options: --opponent aggressive|balanced|defensive  --team red|blue  --games N  --k 5
                --radius 3  --model NAME  --log PATH  --cache PATH  --quiet

## Files (all under src/strategies/llm_commander/)
- strategy.py    LLMCommanderStrategy: the BaseStrategy implementation (the plug into the shared environment)
- commander.py   when to ask the LLM, parsing, validation, retries, logging, statistics
- prompt.py      the instructions the LLM receives (the text sent to the LLM)
- view.py        observation -> what the commander knows -> text for the LLM (uses only the observation)
- commands.py    command vocabulary, roles, named zones
- validator.py   safety layer: rejects illegal orders
- controller.py  orders -> game actions, plus reflexes (carrier runs home, defender chases a thief)
- pathing.py     shortest-path moves (Team 1's A* can replace this)
- llm.py         MockLLM (offline), ClaudeLLM (real), CachedLLM (saves money, repeatable runs)
- baseline.py    scripted teams without an LLM: the "no commander" baseline and test opponents

## Design choices (agree these with the group)
- Roles are not part of the game, so: first agent = attacker, second = defender, third = scout.
- Commands: capture_flag, defend_base, go_to, intercept, scout. Players with no order follow role defaults.
- The commander only sees the observation it is given, so global and partial modes both work.
- The game waits while the LLM thinks (synchronous calls). Decision time is recorded.
- Every prompt, raw answer, validation result, latency and token count is logged in logs/*.jsonl.

## Still to do (search for "TODO(" in the code)
1. Run with the real LLM and refine the prompt: prompt.py, llm.py (ClaudeLLM)
2. Experiment runner (50+ games per pairing, sides swapped, results to CSV): scripts/run_experiments.py, scripts/run_llm_commander.py
3. Extra metrics (flag captures, distance travelled, completion time): scripts/run_llm_commander.py, strategy.py
4. Comparison conditions: (a) random-orders commander: llm.py; (b) validator on/off: commander.py;
   (c) prompt with/without the map: view.py; (d) scout information sharing on/off: strategy.py
5. Mixed teams (LLM commanding Team 1 / Team 2 agents): controller.py, strategy.py
6. Agreed opponent definitions (aggressive / defensive / balanced): baseline.py
7. Report, demo scenario and demo video: scripts/run_experiments.py (demo scenario), docs
8. Decision on the base-camping stalemate: controller.py, baseline.py
