"""Named experimental conditions for the team the commander controls.

LLM conditions (need an LLM backend):
  llm                 LLM orders + simple controller (the main condition)
  llm_classical       LLM orders executed by Team 1's A* / role logic (mixed team)
  llm_rl              LLM orders for the agents it picks, Team 2's trained RL policy plays the rest (mixed team;
                      partial observations, because the policy was trained with fog of war)
  llm_no_validator    ablation: orders are executed without the safety check
  llm_no_map          ablation: the prompt contains no ASCII map
  llm_no_scout_share  ablation: scout observations are not shared (partial observability only)
  random_commander    control: random but legal orders instead of the LLM
No-LLM baselines:
  classical           Team 1's classical AI alone (Approach 1 without a commander)
  rl                  Team 2's trained RL policy alone (Approach 2)
  scripted            this package's role-default team on the simple controller
Add "@K" to set the trigger period for one condition, e.g. "llm@3" (ask at least every 3 turns).
"""
from __future__ import annotations

from strategies.base import BaseStrategy

from .llm import RandomOrdersLLM
from .strategy import LLMCommanderStrategy

LLM_CONDITIONS = ("llm", "llm_classical", "llm_rl", "llm_no_validator", "llm_no_map", "llm_no_scout_share",
                  "random_commander")
BASELINE_CONDITIONS = ("classical", "rl", "scripted")
RL_CONDITIONS = ("llm_rl", "rl")
ALL_CONDITIONS = LLM_CONDITIONS + BASELINE_CONDITIONS


def parse_condition(spec: str) -> tuple[str, int | None]:
    name, _, k = spec.partition("@")
    if name not in ALL_CONDITIONS:
        raise ValueError(f"unknown condition '{name}'; choose from {', '.join(ALL_CONDITIONS)}")
    return name, (int(k) if k else None)


def uses_llm(spec: str) -> bool:
    return parse_condition(spec)[0] in LLM_CONDITIONS


def needs_rl(spec: str) -> bool:
    return parse_condition(spec)[0] in RL_CONDITIONS


def build_condition(spec: str, *, llm, max_steps: int, seed: int = 0, default_k: int = 5,
                    log_path: str | None = None, rl_checkpoint: str | None = None) -> BaseStrategy:
    name, k = parse_condition(spec)
    k = k or default_k
    if name == "classical":
        from classical_ai.classical_strategy import ClassicalStrategy

        return ClassicalStrategy()
    if name == "rl":
        from .rl_adapter import RLPolicyStrategy

        return RLPolicyStrategy(rl_checkpoint)
    if name == "scripted":
        from .baseline import ScriptedStrategy

        return ScriptedStrategy("baseline")
    options = {"llm": {}, "llm_classical": {"controller": "classical"},
               "llm_rl": {"controller": "rl", "rl_checkpoint": rl_checkpoint},
               "llm_no_validator": {"use_validator": False}, "llm_no_map": {"include_map": False},
               "llm_no_scout_share": {"share_scout_info": False}, "random_commander": {}}[name]
    backend = RandomOrdersLLM(seed=seed) if name == "random_commander" else llm
    return LLMCommanderStrategy(backend, every_k_turns=k, log_path=log_path, max_steps=max_steps, **options)
