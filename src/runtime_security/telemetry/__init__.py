"""Structured in-memory run inspection, without persistence."""

from copy import deepcopy

from runtime_security.contracts import AgentResult
from runtime_security.telemetry.events import safe_arguments, safe_summary


class EventCollector:
    def __init__(self):
        self.events: list[dict] = []

    def collect(self, result: AgentResult) -> None:
        for call in result.tool_calls:
            self.events.append(
                {
                    "kind": "proposal",
                    "run_id": call.run_id,
                    "call_id": call.call_id,
                    "tool_name": call.tool_name
                    if call.tool_name
                    in {
                        "read_document",
                        "read_file",
                        "search_files",
                        "mock_http_post",
                    }
                    else "unrecognized_tool",
                    "arguments": safe_arguments(call.tool_name, call.arguments),
                }
            )
        for response in result.tool_responses:
            self.events.append(
                {
                    "kind": "response",
                    "run_id": result.run_id,
                    "call_id": response.call_id,
                    "status": response.status,
                    "output": safe_summary(response.output),
                    "error": "[redacted]" if response.error else None,
                }
            )

    def for_run(self, run_id: str) -> list[dict]:
        return deepcopy([event for event in self.events if event["run_id"] == run_id])
