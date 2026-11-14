"""LLM backends: MockLLM (offline stand-in), GeminiLLM (Gemini API), RandomOrdersLLM (control), CachedLLM (cache)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path


class MockLLM:
    """Offline stand-in that needs no API key. It applies a few fixed rules to the prompt text.
    Used to test the pipeline end to end; not a result of the study."""

    name = "mock"
    last_usage = None

    def complete(self, system: str, user: str) -> str:
        agents = re.findall(r"^- (\w+) \((\w+)\) at \((\d+), (\d+)\)(.*)$", user, flags=re.M)
        stolen = "ALERT: our flag is stolen" in user
        center = re.search(r"center=\((\d+), (\d+)\)", user)
        center = (int(center.group(1)), int(center.group(2))) if center else None
        orders = []
        for aid, role, x, y, rest in agents:
            if "TAGGED" in rest:
                continue
            if stolen and "CARRYING" not in rest:
                orders.append({"agent_id": aid, "command": "intercept", "target": "enemy_carrier",
                               "rationale": "our flag was stolen, chase the thief"})
            elif role == "scout" and center and (int(x), int(y)) != center and "ALERT" not in user:
                orders.append({"agent_id": aid, "command": "scout", "target": "center",
                               "rationale": "look at the middle of the map"})
        return json.dumps({"orders": orders})


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"   # override with --model or the GEMINI_MODEL environment variable


class GeminiLLM:
    """LLM backend using the Gemini API (google-genai SDK).

    Requires the 'google-genai' package and a key from Google AI Studio in the GEMINI_API_KEY
    environment variable. Requests are spaced out to stay under the per-minute quota (rpm) and
    transient errors (rate limit, server busy) are retried with a pause.

    TODO(1): not yet run against the real API. On the first real run (scripts/check_llm.py) check the
    model name, per-request latency, token usage and the free-tier daily quota.
    """

    def __init__(self, model: str | None = None, temperature: float | None = 0.0,
                 max_output_tokens: int = 4096, rpm: float | None = 10.0, max_retries: int = 5,
                 thinking_level: str | None = None, client=None, sleep=time.sleep, clock=time.monotonic):
        self.name = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.temperature, self.max_output_tokens = temperature, max_output_tokens
        # Optional reasoning effort ("low", "medium", "high") for models that support it; lower = faster.
        self.thinking_level = thinking_level or os.environ.get("GEMINI_THINKING_LEVEL") or None
        self.interval = 60.0 / rpm if rpm else 0.0
        self.max_retries, self._sleep, self._clock = max_retries, sleep, clock
        self._last_call = None
        self.last_usage = None
        if client is None:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set. Create a key in Google AI Studio and export it.")
            from google import genai          # pip install google-genai

            client = genai.Client(api_key=key)
        self.client = client

    @staticmethod
    def _retry_delay(message: str) -> float | None:
        m = re.search(r"retry(?:Delay)?\D{0,20}(\d+(?:\.\d+)?)\s*s", message, flags=re.I)
        return float(m.group(1)) if m else None

    def _throttle(self) -> None:
        if self.interval and self._last_call is not None:
            wait = self.interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    def complete(self, system: str, user: str) -> str:
        config = {"system_instruction": system, "max_output_tokens": self.max_output_tokens,
                  "response_mime_type": "application/json"}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.thinking_level:
            config["thinking_config"] = {"thinking_level": self.thinking_level}
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.client.models.generate_content(model=self.name, contents=user, config=config)
                break
            except Exception as e:                                   # SDK raises several error types
                msg = str(e)
                # Some models reject individual settings: drop the setting and try again.
                if "temperature" in msg.lower() and "temperature" in config:
                    config.pop("temperature"); self.temperature = None
                    continue
                if "thinking" in msg.lower() and "thinking_config" in config:
                    config.pop("thinking_config"); self.thinking_level = None
                    continue
                transient = any(code in msg for code in ("429", "RESOURCE_EXHAUSTED", "500", "503",
                                                          "UNAVAILABLE", "INTERNAL", "DEADLINE"))
                if not transient or attempt == self.max_retries:
                    raise RuntimeError(
                        f"Gemini request failed ({type(e).__name__}): {msg[:300]}\n"
                        f"Check the API key and the model name '{self.name}' (use --model to change it).") from e
                self._sleep(min(90.0, self._retry_delay(msg) or 5.0 * 2 ** attempt))
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            out = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
            self.last_usage = {"in": getattr(usage, "prompt_token_count", 0) or 0, "out": out}
        else:
            self.last_usage = None
        try:
            return response.text or ""
        except Exception:
            return ""


class RandomOrdersLLM:
    """Control condition: random but legal orders, no reasoning. Used to check whether the real
    LLM's orders add value beyond the controller's default behaviour."""

    name = "random"
    wants_view = True          # the commander hands over the structured view before each call
    last_usage = None

    def __init__(self, seed: int = 0, order_probability: float = 0.5):
        self.rng = random.Random(seed)
        self.p = order_probability
        self.current_view = None

    def complete(self, system: str, user: str) -> str:
        from .commands import NEEDS_TARGET, ROLE_ALLOWED
        from .pathing import reachable

        v = self.current_view
        orders = []
        for a in v.agents:
            if not a.active or self.rng.random() > self.p:
                continue
            cmd = self.rng.choice(sorted(ROLE_ALLOWED[a.role]))
            order = {"agent_id": a.id, "command": cmd, "rationale": "random"}
            if cmd in NEEDS_TARGET:
                if cmd == "intercept" and v.own_flag.carried and self.rng.random() < 0.5:
                    order["target"] = "enemy_carrier"
                else:
                    cells = [(x, y) for y in range(v.height) for x in range(v.width)
                             if (x, y) not in v.obstacles and reachable(v.width, v.height, v.obstacles, a.pos, (x, y))]
                    order["target"] = list(self.rng.choice(cells))
            orders.append(order)
        return json.dumps({"orders": orders})


class CachedLLM:
    """Wraps another LLM and stores its answers in a file, keyed by the exact prompt.
    Re-running the same game then makes no new API calls and gives identical results."""

    def __init__(self, llm, path: str):
        self.llm, self.path = llm, Path(path)
        self.name = llm.name
        self.last_usage = None
        self.hits = 0
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}

    def complete(self, system: str, user: str) -> str:
        key = hashlib.sha256(f"{self.llm.name}\n{system}\n{user}".encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            self.last_usage = {"in": 0, "out": 0, "cached": True}
            return self.cache[key]
        text = self.llm.complete(system, user)
        self.last_usage = getattr(self.llm, "last_usage", None)
        self.cache[key] = text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache))
        return text



def make_llm(kind: str = "mock", model: str | None = None, rpm: float | None = 10.0, cache: str | None = None):
    """Build the LLM backend from command-line style options ('mock' or 'gemini')."""
    if kind == "mock":
        llm = MockLLM()
    elif kind == "gemini":
        llm = GeminiLLM(model=model, rpm=rpm)
    else:
        raise ValueError("llm must be 'mock' or 'gemini'")
    return CachedLLM(llm, cache) if cache else llm
