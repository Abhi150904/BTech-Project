"""Public synchronous agent interface; implementation belongs to agent core."""
from typing import Protocol

from runtime_security.contracts import AgentRequest, AgentResult


class Agent(Protocol):
    def invoke(self, request: AgentRequest) -> AgentResult:
        """Run a bounded agent invocation."""
        ...
