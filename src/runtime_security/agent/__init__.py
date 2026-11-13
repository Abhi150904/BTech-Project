"""Public synchronous agent interface; implementation belongs to agent core."""

from typing import Protocol

from runtime_security.agent.core import LangGraphAgent, RegistryExecutor, ToolExecutor
from runtime_security.contracts import AgentRequest, AgentResult

__all__ = ["Agent", "LangGraphAgent", "RegistryExecutor", "ToolExecutor"]


class Agent(Protocol):
    def invoke(self, request: AgentRequest) -> AgentResult:
        """Run a bounded agent invocation."""
        ...
