"""Registry interface; implementations own sandbox configuration and tool execution."""
from collections.abc import Sequence
from typing import Protocol

from langchain_core.tools import BaseTool


class ToolRegistry(Protocol):
    def get_tools(self) -> Sequence[BaseTool]:
        """Return configured LangChain tools for agent binding."""
        ...
