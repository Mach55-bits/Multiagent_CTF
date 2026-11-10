import json

from ctf.actions import Action
from ctf.agent import Team
from ctf.config import load_config
from ctf.environment import CTFEnvironment
from ctf.observations import ObservationMode
from strategies.base import BaseStrategy
from strategies.llm_commander import LLMCommanderStrategy, MockLLM, ScriptedStrategy
from strategies.llm_commander.commander import Commander, parse_json
from strategies.llm_commander.commands import assign_roles
from strategies.llm_commander.pathing import bfs_next_action
from strategies.llm_commander.validator import validate
from strategies.llm_commander.view import build_view, view_to_text


def make_env():
    env = CTFEnvironment(load_config("configs/default.yaml"))
    env.reset()
    return env


def make_view(env, team=Team.RED):
    obs = env.observe(ObservationMode.GLOBAL)
    roles = assign_roles([a for a in obs.agents if obs.agents[a].team is team])
    return build_view(obs, team, roles, {}, None, 200)


class ScriptedLLM:
    """Returns the given replies one after another (to test retries)."""
    name = "scripted"
    last_usage = None

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, system, user):
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def test_strategy_is_base_strategy_and_acts_for_active_agents_only():
    env = make_env()
    strat = LLMCommanderStrategy(MockLLM())
    assert isinstance(strat, BaseStrategy)
    actions = strat.act(env.observe(), Team.RED)
    assert set(actions) == {"R1", "R2", "R3"}
    assert all(isinstance(a, Action) for a in actions.values())


def test_roles_assigned_in_order():
    assert assign_roles(["R3", "R1", "R2"]) == {"R1": "attacker", "R2": "defender", "R3": "scout"}


def test_bfs_goes_around_the_wall():
    env = make_env()
    v = make_view(env)
    # R1 at (1,2) heading to blue base (9,3): wall at x=5 for y=1..5, so path must go via row 0 or 6
    assert bfs_next_action(v.width, v.height, v.obstacles, (4, 3), (6, 3)) in (Action.UP, Action.DOWN)


def test_validator_rejects_bad_orders():
    v = make_view(make_env())
    bad = {"orders": [
        {"agent_id": "Z9", "command": "go_to", "target": "north"},            # unknown agent
        {"agent_id": "R2", "command": "capture_flag"},                        # defender cannot capture
        {"agent_id": "R1", "command": "fly", "target": "north"},              # unknown command
        {"agent_id": "R1", "command": "go_to", "target": [5, 2]},             # wall
        {"agent_id": "R1", "command": "go_to", "target": [50, 2]},            # off map
        {"agent_id": "R3", "command": "intercept", "target": "enemy_carrier"} # nothing stolen
    ]}
    clean, errors = validate(v, bad)
    assert clean == [] and len(errors) == 6


def test_validator_accepts_good_orders_and_keeps_dynamic_targets():
    v = make_view(make_env())
    good = {"orders": [{"agent_id": "R1", "command": "capture_flag"},
                       {"agent_id": "R3", "command": "scout", "target": "northeast", "rationale": "look"}]}
    clean, errors = validate(v, good)
    assert errors == [] and len(clean) == 2 and clean[1]["target"] == "northeast"


def test_parse_json_handles_code_fences_and_chatter():
    assert parse_json('```json\n{"orders": []}\n```') == {"orders": []}
    assert parse_json('Sure! {"orders": []} hope that helps') == {"orders": []}
    assert parse_json("no json here") is None


def test_commander_retries_with_error_feedback_then_succeeds():
    v = make_view(make_env())
    bad = json.dumps({"orders": [{"agent_id": "R2", "command": "capture_flag"}]})
    good = json.dumps({"orders": [{"agent_id": "R1", "command": "capture_flag"}]})
    llm = ScriptedLLM(bad, good)
    cmd = Commander(llm, every_k_turns=5)
    orders = cmd.decide(v)
    assert llm.calls == 2 and len(orders) == 1
    assert cmd.stats["illegal_orders"] == 1


def test_commander_only_calls_llm_when_triggered():
    env = make_env()
    llm = ScriptedLLM('{"orders": []}')
    cmd = Commander(llm, every_k_turns=5)
    v = make_view(env)
    cmd.decide(v)                       # turn 0 -> call
    v.timestep = 3
    cmd.decide(v)                       # nothing happened -> no call
    assert llm.calls == 1
    v.events = ["enemy_stole_our_flag(B1)"]
    cmd.decide(v)                       # event -> call
    assert llm.calls == 2


def test_view_text_contains_only_known_information_and_flags_theft():
    env = make_env()
    text = view_to_text(make_view(env))
    assert "You command team RED" in text and "Our agents:" in text and "ALERT" not in text


def test_full_game_runs_for_both_observation_modes():
    for mode in (ObservationMode.GLOBAL, ObservationMode.PARTIAL):
        env = make_env()
        red = LLMCommanderStrategy(MockLLM(), max_steps=200)
        blue = ScriptedStrategy("aggressive")
        for _ in range(60):
            if env.done:
                break
            obs = {t: env.observe(mode, team=t, visibility_radius=3) for t in Team}
            env.step({**red.act(obs[Team.RED], Team.RED), **blue.act(obs[Team.BLUE], Team.BLUE)})
        assert env.state.timestep > 0


def test_baseline_beats_aggressive_opponent():
    env = make_env()
    red, blue = ScriptedStrategy("baseline"), ScriptedStrategy("aggressive")
    while not env.done:
        obs = env.observe()
        env.step({**red.act(obs, Team.RED), **blue.act(obs, Team.BLUE)})
    assert env.state.winner is Team.RED
