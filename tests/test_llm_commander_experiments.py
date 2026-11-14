import csv
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import load_config
from ctf.environment import CTFEnvironment
from ctf.observations import ObservationMode
from strategies.llm_commander import GeminiLLM, LLMCommanderStrategy, MockLLM, RandomOrdersLLM
from strategies.llm_commander.commander import Commander
from strategies.llm_commander.commands import assign_roles
from strategies.llm_commander.validator import validate
from strategies.llm_commander.view import build_view, view_to_text

ROOT = Path(__file__).resolve().parents[1]


def make_env(**overrides):
    cfg = load_config(ROOT / "configs/default.yaml")
    env = CTFEnvironment(dataclasses.replace(cfg, **overrides) if overrides else cfg)
    env.reset(seed=1)
    return env


class ScriptedLLM:
    name = "scripted"
    last_usage = None

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, system, user):
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


# ---------------------------------------------------------------- Gemini backend (fake client)
class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = type("U", (), {"prompt_token_count": 11, "candidates_token_count": 5,
                                             "thoughts_token_count": 2})()


class FakeClient:
    def __init__(self, behaviours):
        self.behaviours, self.calls, self.configs = list(behaviours), 0, []
        self.models = self

    def generate_content(self, model, contents, config):
        self.configs.append(dict(config))
        b = self.behaviours[min(self.calls, len(self.behaviours) - 1)]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        return FakeResponse(b)


def test_gemini_returns_text_usage_and_requests_json():
    client = FakeClient(['{"orders": []}'])
    llm = GeminiLLM(model="m", client=client, rpm=None)
    assert llm.complete("sys", "user") == '{"orders": []}'
    assert llm.last_usage == {"in": 11, "out": 7}
    assert client.configs[0]["response_mime_type"] == "application/json"
    assert client.configs[0]["system_instruction"] == "sys"


def test_gemini_retries_rate_limit_then_succeeds():
    pauses = []
    client = FakeClient([RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 7s"), "ok"])
    llm = GeminiLLM(client=client, rpm=None, sleep=pauses.append)
    assert llm.complete("s", "u") == "ok"
    assert client.calls == 2 and pauses == [7.0]


def test_gemini_drops_a_setting_the_model_rejects():
    client = FakeClient([RuntimeError("400 INVALID_ARGUMENT: temperature is not supported for this model"), "ok"])
    llm = GeminiLLM(model="m", client=client, rpm=None, sleep=lambda s: None)
    assert llm.complete("s", "u") == "ok"
    assert "temperature" in client.configs[0] and "temperature" not in client.configs[1]


def test_gemini_thinking_level_is_sent_when_set(monkeypatch):
    client = FakeClient(["ok"])
    GeminiLLM(model="m", client=client, rpm=None, thinking_level="low").complete("s", "u")
    assert client.configs[0]["thinking_config"] == {"thinking_level": "low"}
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "medium")
    client2 = FakeClient(["ok"])
    GeminiLLM(model="m", client=client2, rpm=None).complete("s", "u")
    assert client2.configs[0]["thinking_config"] == {"thinking_level": "medium"}


def test_gemini_non_transient_error_has_helpful_message():
    llm = GeminiLLM(model="bad-model", client=FakeClient([RuntimeError("404 model not found")]), rpm=None)
    with pytest.raises(RuntimeError, match="bad-model"):
        llm.complete("s", "u")


def test_gemini_spaces_out_requests():
    now, pauses = [0.0], []
    llm = GeminiLLM(client=FakeClient(["a", "b"]), rpm=6, sleep=pauses.append, clock=lambda: now[0])  # 10 s apart
    llm.complete("s", "u")
    now[0] = 3.0
    llm.complete("s", "u")
    assert pauses == [7.0]


def test_gemini_needs_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiLLM()


# ---------------------------------------------------------------- control condition and ablations
def test_random_commander_only_produces_legal_orders():
    env = make_env()
    roles = assign_roles(["R1", "R2", "R3"])
    view = build_view(env.observe(), Team.RED, roles, {}, None, 200)
    llm = RandomOrdersLLM(seed=3, order_probability=1.0)
    for _ in range(25):
        llm.current_view = view
        clean, errors = validate(view, json.loads(llm.complete("", "")))
        assert errors == [] and len(clean) == 3


def test_validator_off_executes_bad_orders_without_crashing():
    bad = json.dumps({"orders": [
        {"agent_id": "R2", "command": "capture_flag"},                 # wrong role
        {"agent_id": "R1", "command": "go_to", "target": [5, 2]},      # a wall
        {"agent_id": "R3", "command": "fly", "target": "north"},       # unknown command
        {"agent_id": "Z9", "command": "go_to", "target": "north"},     # unknown agent
    ]})
    env = make_env()
    strat = LLMCommanderStrategy(ScriptedLLM(bad), use_validator=False)
    for _ in range(5):
        actions = strat.act(env.observe(), Team.RED)
        env.step({**actions, **{a: Action.STAY for a in ("B1", "B2", "B3")}})
    assert strat.commander.stats["illegal_orders"] >= 4        # still counted
    assert strat.commander.stats["failed_replies"] == 0


def test_validator_on_rejects_the_same_orders():
    env = make_env()
    llm = ScriptedLLM(json.dumps({"orders": [{"agent_id": "R2", "command": "capture_flag"}]}))
    cmd = Commander(llm, max_retries=0)
    view = build_view(env.observe(), Team.RED, assign_roles(["R1", "R2", "R3"]), {}, None, 200)
    assert cmd.decide(view) == []


def test_prompt_without_map_omits_the_grid():
    view = build_view(make_env().observe(), Team.RED, assign_roles(["R1", "R2", "R3"]), {}, None, 200)
    assert "y=0" in view_to_text(view)
    assert "y=0" not in view_to_text(view, include_map=False)
    assert "Our agents:" in view_to_text(view, include_map=False)


def _scout_only_sees_enemy_env():
    """Partial observability, radius 1: enemy B1 is adjacent to our scout R3 and far from R1/R2."""
    env = make_env()
    st = env._state          # env.state returns a copy; the test edits the live state
    place = {"R1": (0, 0), "R2": (1, 0), "R3": (9, 5), "B1": (9, 6), "B2": (10, 0), "B3": (10, 1)}
    for aid, (x, y) in place.items():
        st.agents[aid] = dataclasses.replace(st.agents[aid], x=x, y=y)
    return env


def test_scout_information_sharing_switch():
    for share, expected in ((True, True), (False, False)):
        env = _scout_only_sees_enemy_env()
        strat = LLMCommanderStrategy(MockLLM(), share_scout_info=share)
        strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED, visibility_radius=1), Team.RED)
        seen = {e[0] for e in strat.prev_view.enemies}
        assert ("B1" in seen) is expected


def test_reset_clears_per_game_memory():
    env = make_env()
    strat = LLMCommanderStrategy(MockLLM())
    strat.act(env.observe(), Team.RED)
    assert strat.roles and strat.timeline is not None
    strat.reset()
    assert strat.roles == {} and strat.last_t == -1 and strat.enemy_memory == {}


# ---------------------------------------------------------------- other branches (skipped if not merged)
def test_all_opponents_can_be_built_and_play():
    pytest.importorskip("rl.opponents")
    pytest.importorskip("classical_ai.classical_strategy")
    from strategies.llm_commander.opponents import ALL_OPPONENTS, make_opponent

    cfg = load_config(ROOT / "configs/default.yaml")
    for name in [n for n in ALL_OPPONENTS if n != "rl"]:     # 'rl' needs a checkpoint: see the RL tests
        env = make_env()
        opp = make_opponent(name, cfg, seed=1)
        for _ in range(10):
            actions = opp.act(env.observe(), Team.BLUE)
            assert set(actions) <= {"B1", "B2", "B3"}
            env.step({**actions, "R1": Action.STAY, "R2": Action.STAY, "R3": Action.STAY})


@pytest.mark.parametrize("mode", [ObservationMode.GLOBAL, ObservationMode.PARTIAL])
def test_classical_controller_runs_full_games(mode):
    pytest.importorskip("classical_ai.coordinator")
    from classical_ai.classical_strategy import ClassicalStrategy

    env = make_env()
    mine = LLMCommanderStrategy(MockLLM(), controller="classical")
    theirs = ClassicalStrategy()
    for _ in range(80):
        if env.done:
            break
        obs = {t: env.observe(mode, team=t, visibility_radius=3) for t in Team}
        env.step({**mine.act(obs[Team.RED], Team.RED), **theirs.act(obs[Team.BLUE], Team.BLUE)})
    assert env.state.timestep > 0


def test_classical_controller_follows_the_llm_order():
    pytest.importorskip("classical_ai.coordinator")
    order = json.dumps({"orders": [{"agent_id": "R3", "command": "go_to", "target": [10, 0], "rationale": "x"}]})
    env = make_env()
    mine = LLMCommanderStrategy(ScriptedLLM(order, '{"orders": []}'), controller="classical", every_k_turns=100)
    start = (env.state.agents["R3"].x, env.state.agents["R3"].y)
    for _ in range(6):
        env.step({**mine.act(env.observe(), Team.RED), "B1": Action.STAY, "B2": Action.STAY, "B3": Action.STAY})
    end = (env.state.agents["R3"].x, env.state.agents["R3"].y)
    assert abs(end[0] - 10) + abs(end[1] - 0) < abs(start[0] - 10) + abs(start[1] - 0)


# ---------------------------------------------------------------- experiment runner
def test_experiment_runner_writes_csv_and_resumes(tmp_path):
    pytest.importorskip("rl.opponents")
    pytest.importorskip("classical_ai.evaluation")
    out, summ = tmp_path / "g.csv", tmp_path / "s.csv"
    cmd = [sys.executable, str(ROOT / "scripts/run_experiments.py"), "--games", "2", "--conditions", "llm,classical",
           "--opponents", "balanced", "--out", str(out), "--summary", str(summ), "--log-dir", str(tmp_path / "logs")]
    first = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert first.returncode == 0, first.stderr
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4 and {r["side"] for r in rows} == {"red", "blue"}
    assert {r["result"] for r in rows} <= {"win", "loss", "draw"}
    assert summ.exists()
    again = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert again.returncode == 0 and len(list(csv.DictReader(out.open()))) == 4      # nothing replayed
    assert "[" not in again.stdout.split("=== Summary")[0].split("Plan:")[-1].split("estimated")[-1]


def test_experiment_runner_dry_run_prints_plan():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_experiments.py"), "--dry-run", "--games", "10",
                        "--conditions", "llm", "--opponents", "balanced"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "estimated LLM requests" in r.stdout


# ---------------------------------------------------------------- reporting and parsing
def test_wilson_interval_and_summary():
    from strategies.llm_commander.reporting import summarise, wilson

    lo, hi = wilson(0, 30)
    assert lo == 0.0 and 0.1 < hi < 0.12
    rows = [{"condition": "llm", "opponent": "x", "observability": "global", "result": r, "turns": "10",
             "my_flag_pickups": "1", "my_agents_tagged": "0", "my_moves": "5", "orders": "2",
             "illegal_orders": "1", "llm_requests": "2", "llm_seconds": "4", "tokens_in": "10", "tokens_out": "5"}
            for r in ("win", "win", "loss", "draw")]
    s = summarise(rows)[0]
    assert s["games"] == 4 and s["wins"] == 2 and s["draws"] == 1 and s["win_rate"] == 0.5
    assert s["illegal_order_rate"] == 0.333 and s["avg_llm_seconds_per_request"] == 2.0


def test_demo_candidates_pair_games_by_seed():
    from strategies.llm_commander.reporting import demo_candidates

    def row(cond, seed, result):
        return {"condition": cond, "opponent": "x", "observability": "global", "side": "red", "seed": seed,
                "result": result, "turns": "20"}
    rows = [row("llm", "1", "win"), row("base", "1", "draw"), row("llm", "2", "draw"), row("base", "2", "win"),
            row("llm", "3", "win"), row("base", "3", "win")]
    better, worse = demo_candidates(rows, "llm", "base")
    assert [d["seed"] for d in better] == ["1"] and [d["seed"] for d in worse] == ["2"]


def test_parse_accepts_bare_list_of_orders():
    from strategies.llm_commander.commander import parse_json

    assert parse_json('[{"agent_id": "R1", "command": "capture_flag"}]') == {
        "orders": [{"agent_id": "R1", "command": "capture_flag"}]}


def test_report_script_runs(tmp_path):
    pytest.importorskip("rl.opponents")
    pytest.importorskip("classical_ai.evaluation")
    out = tmp_path / "g.csv"
    run = subprocess.run([sys.executable, str(ROOT / "scripts/run_experiments.py"), "--games", "2", "--conditions",
                          "llm,classical", "--opponents", "balanced", "--out", str(out), "--summary",
                          str(tmp_path / "s.csv"), "--log-dir", str(tmp_path / "l")], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr
    rep = subprocess.run([sys.executable, str(ROOT / "scripts/make_report.py"), "--games", str(out), "--out",
                          str(tmp_path / "rep"), "--demo", "llm,classical"], capture_output=True, text=True, cwd=ROOT)
    assert rep.returncode == 0, rep.stderr
    assert (tmp_path / "rep" / "tables.md").exists() and (tmp_path / "rep" / "demo_candidates.md").exists()


# ---------------------------------------------------------------- RL agent integration (stand-in network, no PyTorch needed)
def _rl_parts():
    pytest.importorskip("numpy")
    pytest.importorskip("rl.features")
    import numpy as np
    from rl.config import load_base_env_config, load_rl_config

    rl_cfg = load_rl_config(str(ROOT / "configs/rl/default.yaml"), {})
    env_cfg = load_base_env_config(rl_cfg, root=ROOT)
    return np, rl_cfg, env_cfg


def _fake_infer(np, fixed=None, seed=0):
    rng = np.random.default_rng(seed)

    def infer(features):
        return np.array([fixed if fixed is not None else int(rng.choice(np.flatnonzero(features.mask[s])))
                         for s in range(len(features.alive))])
    return infer


def _rl_policy(fixed=None):
    from strategies.llm_commander.rl_adapter import RLPolicyStrategy

    np, rl_cfg, env_cfg = _rl_parts()
    return RLPolicyStrategy(infer=_fake_infer(np, fixed), env_config=env_cfg, rl_config=rl_cfg)


def test_rl_policy_returns_actions_for_active_agents_and_mirrors_blue():
    from rl.canonical import ACTION_INDEX

    policy = _rl_policy(fixed=ACTION_INDEX[Action.LEFT])
    env = make_env()
    red = policy.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    blue = policy.act(env.observe(ObservationMode.PARTIAL, team=Team.BLUE), Team.BLUE)
    assert set(red) == {"R1", "R2", "R3"} and set(blue) == {"B1", "B2", "B3"}
    assert set(red.values()) == {Action.LEFT}
    assert set(blue.values()) == {Action.RIGHT}          # BLUE sees the board mirrored


def test_rl_policy_skips_tagged_agents_and_rejects_global_observations():
    policy = _rl_policy()
    env = make_env()
    env._state.agents["R2"] = dataclasses.replace(env._state.agents["R2"], active=False)
    assert set(policy.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)) == {"R1", "R3"}
    with pytest.raises(ValueError, match="partial"):
        policy.act(env.observe(ObservationMode.GLOBAL), Team.RED)


def test_mixed_rl_team_obeys_orders_and_lets_the_policy_play_the_rest():
    from rl.canonical import ACTION_INDEX
    from strategies.llm_commander.rl_adapter import RLControllerAdapter

    order = json.dumps({"orders": [{"agent_id": "R1", "command": "go_to", "target": [1, 6]}]})   # straight down
    adapter = RLControllerAdapter(policy=_rl_policy(fixed=ACTION_INDEX[Action.UP]))
    strat = LLMCommanderStrategy(ScriptedLLM(order, '{"orders": []}'), controller=adapter, every_k_turns=100)
    assert strat.needs_partial_observation
    env = make_env()
    actions = strat.act(env.observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert actions["R1"] is Action.DOWN                                   # LLM order, walked
    assert actions["R2"] is Action.UP and actions["R3"] is Action.UP      # RL policy


def test_runner_gives_rl_strategies_partial_observations_in_global_games():
    pytest.importorskip("classical_ai.evaluation")
    from classical_ai.classical_strategy import ClassicalStrategy
    from strategies.llm_commander.rl_adapter import RLControllerAdapter
    from strategies.llm_commander.runner import play_game

    cfg = load_config(ROOT / "configs/default.yaml")
    for mine in (_rl_policy(), LLMCommanderStrategy(MockLLM(), controller=RLControllerAdapter(policy=_rl_policy()),
                                                    max_steps=cfg.max_episode_steps)):
        for team in (Team.RED, Team.BLUE):
            res = play_game(cfg, mine, ClassicalStrategy(), team, seed=2)          # global observability
            assert res.result in ("win", "loss", "draw") and res.turns > 0


def test_checkpoint_resolution_messages(tmp_path):
    from strategies.llm_commander.rl_adapter import resolve_checkpoint

    assert resolve_checkpoint("untrained") == "untrained"
    with pytest.raises(RuntimeError, match="rl.train"):
        resolve_checkpoint(str(tmp_path / "missing.pt"))
    existing = tmp_path / "x.pt"
    existing.write_bytes(b"")
    assert resolve_checkpoint(str(existing)) == str(existing)


def test_scripts_explain_a_missing_rl_checkpoint():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_llm_commander.py"), "--condition", "llm_rl",
                        "--rl-checkpoint", "does/not/exist.pt"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "RL checkpoint not found" in r.stderr


def test_untrained_policy_runs_when_torch_is_installed():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from strategies.llm_commander.rl_adapter import RLPolicyStrategy

    policy = RLPolicyStrategy("untrained")
    actions = policy.act(make_env().observe(ObservationMode.PARTIAL, team=Team.RED), Team.RED)
    assert set(actions) == {"R1", "R2", "R3"} and all(isinstance(a, Action) for a in actions.values())
