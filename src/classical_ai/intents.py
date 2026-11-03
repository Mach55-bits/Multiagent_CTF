from dataclasses import dataclass
from enum import Enum

class IntentType(str,Enum):
    GO_TO = "GO_TO"
    GUARD = "GUARD"
    INTERCEPT = "INTERCEPT"
    SCOUT = "SCOUT"


@dataclass(frozen=True)
class Intent:
    type: IntentType
    target: tuple[int,int] | None = None