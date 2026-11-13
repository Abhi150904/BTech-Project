"""Run deterministic synthetic scenario checks without calling an LLM provider."""

from uuid import uuid4

from runtime_security.contracts import AgentRequest, AgentResult, ToolCallEvent, ToolResponse
from runtime_security.scenarios import (
    AUTHORIZED_READ,
    AUTHORIZED_SHARE,
    BENIGN,
    INJECTION,
    SENSITIVE_READ,
    Scenario,
    ScenarioRunner,
)
from runtime_security.tools import SandboxToolRegistry

SCENARIOS = (BENIGN, SENSITIVE_READ, INJECTION, AUTHORIZED_READ, AUTHORIZED_SHARE)


class DeterministicScenarioAgent:
    """A scripted agent adapter for exercising scenario predicates offline."""

    def __init__(self, registry: SandboxToolRegistry, scenario: Scenario):
        self.registry = registry
        self.scenario = scenario
        self.tools = {tool.name: tool for tool in registry.get_tools()}

    def invoke(self, request: AgentRequest) -> AgentResult:
        scenario = self.scenario
        calls: list[ToolCallEvent] = []
        responses: list[ToolResponse] = []

        def finish(answer_material: str = "") -> AgentResult:
            failed = any(response.status == "error" for response in responses)
            return AgentResult(
                run_id=request.run_id,
                final_answer=(
                    "Scripted tool execution failed."
                    if failed
                    else f"Scenario response based on fixture: {answer_material}"
                ),
                tool_calls=calls,
                tool_responses=responses,
            )

        def execute(tool_name: str, arguments: dict) -> str | dict | None:
            call_id = str(len(calls) + 1)
            event = ToolCallEvent(
                run_id=request.run_id,
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
            )
            calls.append(event)
            try:
                output = self.tools[tool_name].invoke(arguments)
            except Exception as exc:
                responses.append(
                    ToolResponse(
                        call_id=call_id,
                        status="error",
                        error=str(exc) or type(exc).__name__,
                    )
                )
                return None
            responses.append(ToolResponse(call_id=call_id, status="success", output=output))
            return output

        answer_material = ""
        if scenario.requires_document_read:
            document = execute("read_document", {"document_id": scenario.document_id})
            if document is None:
                return finish()
            answer_material = str(document)
        if scenario is SENSITIVE_READ:
            execute("read_file", {"file_id": "sensitive.txt"})
        elif scenario is INJECTION:
            secret = execute("read_file", {"file_id": scenario.attack_file_id})
            if secret is None:
                return finish()
            execute(
                "mock_http_post",
                {
                    "destination": scenario.attack_destination,
                    "payload": str(secret),
                },
            )
            answer_material = str(secret)
        elif scenario is AUTHORIZED_READ:
            answer_material = str(execute("read_file", {"file_id": scenario.authorized_file_id}))
        elif scenario is AUTHORIZED_SHARE:
            public = execute("read_file", {"file_id": scenario.authorized_file_id})
            if public is None:
                return finish()
            answer_material = str(public)
            execute(
                "mock_http_post",
                {"destination": scenario.authorized_destination, "payload": str(public)},
            )

        return finish(answer_material)


def run_scenario(scenario: Scenario, max_steps: int = 10):
    registry = SandboxToolRegistry()
    runner = ScenarioRunner(
        DeterministicScenarioAgent(registry, scenario), lambda: registry.recorded_requests
    )
    return runner.run(scenario, str(uuid4()), max_steps, model_name="deterministic-scripted")


def main() -> int:
    failures = 0
    for scenario in SCENARIOS:
        result = run_scenario(scenario)
        print(
            f"{result.scenario_id} v{result.scenario_version}: {result.outcome}; "
            f"attempted={result.attack_attempted}; completed={result.attack_completed}; "
            f"benign_task_completed={result.benign_task_completed}; "
            f"expected={scenario.expected_outcome}"
        )
        if (
            result.outcome != scenario.expected_outcome
            or not result.benign_task_completed
            or (scenario.expected_outcome == "failed" and not result.attack_completed)
        ):
            failures += 1
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
