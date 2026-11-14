"""The Commander: decides WHEN to ask the LLM, asks, parses, validates, retries, logs."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .commands import View
from .prompt import build_system_prompt
from .validator import validate
from .view import view_to_text


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        data = json.loads(text)
        return {"orders": data} if isinstance(data, list) else data     # a bare list of orders is accepted
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)       # tolerate chatter around the JSON
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def raw_orders(parsed) -> list[dict]:
    """Structure-only extraction used when the validator is switched off (no legality checks)."""
    out = []
    items = parsed.get("orders") if isinstance(parsed, dict) else None
    for o in items if isinstance(items, list) else []:
        if isinstance(o, dict) and isinstance(o.get("agent_id"), str) and isinstance(o.get("command"), str):
            out.append({"agent_id": o["agent_id"], "command": o["command"], "target": o.get("target"),
                        "rationale": str(o.get("rationale", ""))[:200]})
    return out


class Commander:
    def __init__(self, llm, every_k_turns: int = 5, max_retries: int = 2, log_path: str | None = None,
                 label: str = "", use_validator: bool = True, include_map: bool = True) -> None:
        self.llm, self.k, self.max_retries = llm, every_k_turns, max_retries
        self.use_validator = use_validator      # False = ablation: orders are executed unchecked
        self.include_map = include_map          # False = ablation: prompt without the ASCII map
        self.log_path = Path(log_path) if log_path else None
        self.label = label
        self.stats = {"calls": 0, "llm_requests": 0, "illegal_orders": 0, "orders": 0,
                      "failed_replies": 0, "latency_s": 0.0, "tokens_in": 0, "tokens_out": 0}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def should_call(self, view: View) -> bool:
        """Trigger policy (hybrid): first turn, every K turns, or something important happened."""
        if not any(a.active for a in view.agents):
            return False
        if view.timestep == 0 or view.timestep % self.k == 0:
            return True
        if view.events:
            return True
        return any(s.status in ("done", "failed", "blocked", "expired") for s in view.order_status)

    def decide(self, view: View) -> list[dict]:
        if not self.should_call(view):
            return []
        self.stats["calls"] += 1
        system, user = build_system_prompt(view), view_to_text(view, include_map=self.include_map)
        if getattr(self.llm, "wants_view", False):
            self.llm.current_view = view          # used by the random-orders control condition
        prompt, attempts, orders = user, [], []
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            raw = self.llm.complete(system, prompt)
            latency = time.time() - t0
            usage = getattr(self.llm, "last_usage", None) or {}
            self.stats["llm_requests"] += 1
            self.stats["latency_s"] += latency
            self.stats["tokens_in"] += usage.get("in", 0)
            self.stats["tokens_out"] += usage.get("out", 0)
            parsed = parse_json(raw)
            if parsed is None:
                clean, errors = [], ["Reply was not valid JSON."]
                self.stats["failed_replies"] += 1
            else:
                clean, errors = validate(view, parsed)
                self.stats["illegal_orders"] += len(errors)      # counted in both modes
                if not self.use_validator:
                    clean, errors = raw_orders(parsed), []       # ablation: no checking, no retry
            attempts.append({"attempt": attempt, "raw": raw, "errors": errors,
                             "latency_s": round(latency, 3), "usage": usage})
            orders = clean                          # legal orders survive even if others were rejected
            if not errors:
                break
            prompt = (user + "\n\nYour previous reply had problems:\n- " + "\n- ".join(errors)
                      + "\nReply again with the corrected JSON object only.")
        self.stats["orders"] += len(orders)
        self._log({"label": self.label, "turn": view.timestep, "model": self.llm.name,
                   "system": system, "prompt": user, "attempts": attempts, "final_orders": orders})
        return orders

    def _log(self, record: dict) -> None:
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
