"""Structured in-memory run inspection, without persistence."""

from copy import deepcopy

from runtime_security.contracts import AgentResult


class EventCollector:
    def __init__(self):
        self.events: list[dict] = []

    def collect(self, result: AgentResult) -> None:
        for call in result.tool_calls:
            self.events.append({"kind": "proposal", **call.model_dump()})
        for response in result.tool_responses:
            self.events.append(
                {"kind": "response", "run_id": result.run_id, **response.model_dump()}
            )

    def for_run(self, run_id: str) -> list[dict]:
        return deepcopy([event for event in self.events if event["run_id"] == run_id])
