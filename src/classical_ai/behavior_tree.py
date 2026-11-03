from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import TeamObservation
from ctf.game_state import FlagStatus

from classical_ai.intents import Intent, IntentType
from classical_ai.pathfinding import next_action


Position = tuple[int, int]


class NodeStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class AgentContext:
    observation: TeamObservation
    team: Team
    agent_id: str
    intent: Intent | None = None


class BehaviorNode(ABC):

    @abstractmethod
    def tick(self, context: AgentContext) -> NodeStatus:
        pass


class ConditionNode(BehaviorNode):

    def __init__(self, condition: Callable[[AgentContext], bool]):
        self.condition = condition

    def tick(self, context: AgentContext) -> NodeStatus:
        if self.condition(context):
            return NodeStatus.SUCCESS

        return NodeStatus.FAILURE


class SetIntentNode(BehaviorNode):

    def __init__(
        self,
        intent_selector: Callable[[AgentContext], Intent | None]
    ):
        self.intent_selector = intent_selector

    def tick(self, context: AgentContext) -> NodeStatus:

        intent = self.intent_selector(context)

        if intent is None:
            return NodeStatus.FAILURE

        context.intent = intent

        return NodeStatus.SUCCESS


class SequenceNode(BehaviorNode):

    def __init__(self, children: list[BehaviorNode]):
        self.children = children

    def tick(self, context: AgentContext) -> NodeStatus:

        for child in self.children:

            result = child.tick(context)

            if result == NodeStatus.FAILURE:
                return NodeStatus.FAILURE

        return NodeStatus.SUCCESS


class SelectorNode(BehaviorNode):

    def __init__(self, children: list[BehaviorNode]):
        self.children = children

    def tick(self, context: AgentContext) -> NodeStatus:

        for child in self.children:

            result = child.tick(context)

            if result == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS

        return NodeStatus.FAILURE


class BehaviorTree:

    def __init__(self, root: BehaviorNode):
        self.root = root

    def decide(self, context: AgentContext) -> Intent | None:

        context.intent = None

        self.root.tick(context)

        return context.intent


def create_default_behavior_tree() -> BehaviorTree:

    def get_agent(context):
        return context.observation.agents.get(context.agent_id)

    def is_carrying_flag(context):
        agent = get_agent(context)

        return agent is not None and agent.carrying_flag

    def get_enemy_flag_intent(context):

        enemy_flag = context.observation.flags.get(context.team.opponent)

        if enemy_flag is None:
            return None

        if enemy_flag.status == FlagStatus.CARRIED:
            intent_type = IntentType.INTERCEPT
        else:
            intent_type = IntentType.GO_TO

        return Intent(
            type=intent_type,
            target=(enemy_flag.x, enemy_flag.y)
        )

    def get_own_base_intent(context):

        if context.team is Team.RED:
            target = context.observation.red_base
        else:
            target = context.observation.blue_base

        return Intent(
            type=IntentType.GO_TO,
            target=target
        )

    def get_scout_intent(context):

        if context.team is Team.RED:
            target = context.observation.blue_base
        else:
            target = context.observation.red_base

        return Intent(
            type=IntentType.SCOUT,
            target=target
        )

    carrying_flag = SequenceNode([
        ConditionNode(is_carrying_flag),
        SetIntentNode(get_own_base_intent)
    ])

    pursue_enemy_flag = SetIntentNode(get_enemy_flag_intent)

    scout_enemy_territory = SetIntentNode(get_scout_intent)

    root = SelectorNode([
        carrying_flag,
        pursue_enemy_flag,
        scout_enemy_territory
    ])

    return BehaviorTree(root)


def decide_action(
    agent_id: str,
    observation: TeamObservation,
    team: Team,
    influence_map: list[list[float]] | None = None,
    algorithm: str = "astar",
    behavior_tree: BehaviorTree | None = None,
) -> Action:

    agent = observation.agents.get(agent_id)

    if agent is None or agent.team is not team or not agent.active:
        return Action.STAY

    if behavior_tree is None:
        behavior_tree = create_default_behavior_tree()

    context = AgentContext(
        observation=observation,
        team=team,
        agent_id=agent_id
    )

    intent = behavior_tree.decide(context)

    if intent is None or intent.target is None:
        return Action.STAY

    start = (agent.x, agent.y)

    return next_action(
        start=start,
        goal=intent.target,
        observation=observation,
        algorithm=algorithm,
        influence_map=influence_map
    )
