
from ctf.actions import Action
from ctf.agent import Team
from ctf.observations import TeamObservation

from strategies.base import BaseStrategy

from classical_ai.coordinator import ClassicalCoordinator


class ClassicalStrategy(BaseStrategy):

    def __init__(
        self,
        role_assignments: dict[Team, dict[str, str]] | None = None,
    ):
        self.role_assignments = role_assignments
        self.coordinator = None

    def act(
        self,
        observation: TeamObservation,
        team: Team,
    ) -> dict[str, Action]:

        if self.coordinator is None:
            self.coordinator = ClassicalCoordinator(
                width=observation.width,
                height=observation.height,
                role_assignments=self.role_assignments,
            )

        return self.coordinator.act(
            observation=observation,
            team=team,
        )
