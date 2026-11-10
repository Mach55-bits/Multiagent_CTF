"""LLM backends: MockLLM (offline stand-in), ClaudeLLM (API client), CachedLLM (answer cache)."""
from __future__ import annotations

import hashlib
import json
import re
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


class ClaudeLLM:
    """LLM backend using the Anthropic API.
    Requires the 'anthropic' package and the ANTHROPIC_API_KEY environment variable.

    TODO(1): not yet run against the real API. On the first real run check the model name,
    temperature / max_tokens settings, per-request latency and token cost.
    """

    def __init__(self, model: str = "claude-sonnet-5-5", temperature: float | None = 0.0, max_tokens: int = 700):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.name, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.last_usage = None

    def complete(self, system: str, user: str) -> str:
        kwargs = dict(model=self.name, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            r = self.client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as e:
            if "temperature" in str(e).lower():       # some models reject this setting
                kwargs.pop("temperature", None)
                self.temperature = None
                r = self.client.messages.create(**kwargs)
            else:
                raise
        self.last_usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


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


# TODO(4a): control condition "random commander". Add a RandomOrdersLLM here with the same
# complete(system, user) method that returns random but legal orders (random agent, command and
# target, in the same JSON format). Comparing it with the real LLM shows whether the LLM's
# orders add value beyond the controller's default behaviour.
