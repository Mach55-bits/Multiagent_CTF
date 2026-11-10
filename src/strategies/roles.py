"""Agent roles shared by every control paradigm (classical, RL, LLM)."""

from enum import Enum


class Role(str, Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"
    SCOUT = "scout"
