import importlib

import pytest
from pydantic import ValidationError

from runtime_security.contracts import (
    AgentRequest,
    AgentResult,
    ScenarioResult,
    ToolCallEvent,
    ToolResponse,
)


def test_imports():
    for name in ("agent", "tools", "contracts", "telemetry", "scenarios"):
        importlib.import_module(f"runtime_security.{name}")


def test_contract_round_trip():
    call = ToolCallEvent(run_id="run-1", call_id="call-1", tool_name="example")
    response = ToolResponse(call_id=call.call_id, status="success", output={"text": "ok"})
    result = ScenarioResult(
        scenario_id="smoke", outcome="passed",
        agent_result=AgentResult(
            run_id=call.run_id, final_answer="ok", tool_calls=[call], tool_responses=[response]
        ),
    )
    assert ScenarioResult.model_validate_json(result.model_dump_json()) == result


def test_invalid_contracts():
    with pytest.raises(ValidationError):
        AgentRequest(run_id="run-1", prompt="test", max_steps=0)
    with pytest.raises(ValidationError):
        ToolResponse(call_id="call-1", status="error")
    with pytest.raises(ValidationError):
        ToolCallEvent(run_id="run-1", call_id="call-1", tool_name="")
    with pytest.raises(ValidationError):
        ToolResponse(call_id="call-1", status="success", error="unexpected")
