"""Skip the RL tests when the optional RL dependencies are not installed."""

import importlib.util

collect_ignore_glob: list[str] = []
if importlib.util.find_spec("numpy") is None:
    collect_ignore_glob.append("test_rl_*.py")
