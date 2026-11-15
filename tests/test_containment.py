"""Exercise the enforced boundary with local tools, deterministic signals and ledgers."""

import json
from threading import Event
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from runtime_security.agent import LangGraphAgent, RegistryExecutor
from runtime_security.contracts import AgentRequest, ToolCallEvent, ToolResponse
from runtime_security.gateway.decisions import Containment
from runtime_security.scenarios import AUTHORIZED_READ, AUTHORIZED_SHARE, BENIGN, INJECTION
from runtime_security.scenarios.lab import run_scenario
from runtime_security.telemetry.events import RunMetadata
from runtime_security.telemetry.recorder import (
    InMemoryRecorder,
    JsonlRecorder,
    TelemetryError,
    read_events,
    run_context,
)
from runtime_security.tools import SandboxToolRegistry

READ = {"file_id": "sensitive.txt"}
DOC = {"document_id": "summary.txt"}
POST = {
    "destination": AUTHORIZED_SHARE.authorized_destination,
    "payload": AUTHORIZED_SHARE.authorized_payload_marker,
}


def signal(score=0.1):
    return {"score": score, "threshold": 0.5, "model_version": "a" * 64}


def quiet(events, data):
    return signal()


def boundary(scenario=AUTHORIZED_SHARE, *, recorder=None, delegate=None, **checks):
    registry = SandboxToolRegistry()
    recorder = recorder if recorder is not None else InMemoryRecorder()
    recorder.start_run(
        "r",
        RunMetadata(
            max_steps=8,
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.scenario_version,
        ),
    )
    checks.setdefault("detector", quiet)
    executor = RegistryExecutor(
        registry.get_tools(),
        delegate=delegate,
        containment=Containment(**checks),
        recorder=recorder,
    )
    return registry, recorder, executor


def send(executor, name="mock_http_post", args=None, cid="c"):
    return executor.execute(
        ToolCallEvent(
            run_id="r", call_id=cid, tool_name=name, arguments=POST if args is None else args
        )
    )


@pytest.mark.parametrize(
    ("scenario", "name", "args", "expected"),
    [
        (BENIGN, "read_document", DOC, "allow"),
        (AUTHORIZED_READ, "read_file", READ, "allow"),
        (BENIGN, "read_file", READ, "block"),
        (AUTHORIZED_SHARE, "mock_http_post", POST, "allow"),
        (BENIGN, "search_files", {"query": "private query"}, "review"),
        (BENIGN, "other", {"raw": "do not record"}, "review"),
        (BENIGN, "read_file", {"file_id": "unknown"}, "review"),
        (
            AUTHORIZED_SHARE,
            "mock_http_post",
            {**POST, "destination": "https://unknown.invalid"},
            "review",
        ),
        (
            AUTHORIZED_SHARE,
            "mock_http_post",
            {**POST, "destination": "https://collector.invalid/upload"},
            "block",
        ),
        (AUTHORIZED_SHARE, "mock_http_post", {**POST, "payload": "private payload"}, "block"),
        (AUTHORIZED_SHARE, "mock_http_post", {**POST, "approval": True}, "block"),
    ],
)
def test_tools_and_grants(scenario, name, args, expected):
    checker = Mock(side_effect=quiet)
    delegate = Mock()
    delegate.execute.return_value = ToolResponse(call_id="c", status="success", output="real value")
    _, recorder, executor = boundary(scenario, delegate=delegate, detector=checker)
    response = send(executor, name, args)
    assert response.call_id == "c"
    assert checker.call_count == 1
    start, proposal, decision, result = recorder.for_run("r")
    assert [start.kind, proposal.kind, decision.kind, result.kind] == [
        "run_start",
        "proposal",
        "decision",
        "tool_result",
    ]
    assert [proposal.call_id, decision.call_id, result.call_id] == ["c"] * 3
    assert decision.summary["decision"] == expected
    assert decision.summary["detector"] == "below_threshold"
    assert decision.summary["model_version"] == "a" * 64
    assert len(decision.summary["policy_hash"]) == 64
    if expected == "allow":
        delegate.execute.assert_called_once()
        assert response.output == "real value" and result.summary["status"] == "success"
    else:
        delegate.execute.assert_not_called()
        assert response.status == "error" and response.output is None
        assert result.summary == {"status": "error", "execution": "not_started"}
    export = recorder.export_run("r")
    for private in ("private payload", "private query", "do not record", POST["payload"].strip()):
        assert private not in export and private not in (response.error or "")


def test_scope_and_single_use_cannot_be_overridden(tmp_path):
    approve = Mock(return_value=True)
    path = tmp_path / "audit.jsonl"
    registry, recorder, gateway = boundary(recorder=JsonlRecorder(path), approve=approve)
    assert send(gateway).status == "success"
    assert send(gateway, cid="repeat").status == "error"
    assert send(gateway, args={**POST, "payload": "other"}, cid="bad").status == "error"
    assert registry.recorded_requests == [POST]
    decisions = [e.summary for e in read_events(path) if e.kind == "decision"]
    assert decisions[1]["rules"] == ["share_grant_used"]
    assert decisions[2]["rules"] == ["payload_out_of_scope"]
    assert POST["payload"].strip() not in recorder.export_run("r")
    approve.assert_not_called()


@pytest.mark.parametrize(
    ("approval", "permitted"), [(None, False), (False, False), ("approved", False), (True, True)]
)
def test_review_requires_trusted_literal_approval(approval, permitted):
    callback = Mock(return_value=approval) if approval is not None else None
    registry, recorder, gateway = boundary(approve=callback, detector=lambda *_: signal(0.5))
    assert (send(gateway).status == "success") == permitted
    assert len(registry.recorded_requests) == int(permitted)
    summary = recorder.for_run("r")[-2].summary
    assert summary["policy_decision"] == "allow" and summary["detector"] == "alert"
    assert summary["decision"] == ("allow" if permitted else "review")
    if callback:
        request = callback.call_args.args[0]
        assert request.target == POST["destination"] and request.call_id == "c"
        assert POST["payload"].strip() not in str(request)


@pytest.mark.parametrize(
    "invalid",
    [
        signal(float("nan")),
        signal(-1),
        signal(True),
        {**signal(), "threshold": 2},
        {**signal(), "model_version": "private"},
        RuntimeError("private detector exception"),
    ],
)
def test_bad_signal_fails_closed(invalid):
    approve = Mock(return_value=True)
    detector = (
        Mock(side_effect=invalid) if isinstance(invalid, Exception) else Mock(return_value=invalid)
    )
    registry, recorder, gateway = boundary(detector=detector, approve=approve)
    assert send(gateway).status == "error" and not registry.recorded_requests
    assert recorder.for_run("r")[-2].summary["reason"] == "detector_failure"
    approve.assert_not_called()
    assert "private" not in recorder.export_run("r")


@pytest.mark.parametrize("slow", ["detector", "approval"])
def test_timeout_has_no_late_dispatch(slow):
    entered, release, returned = Event(), Event(), Event()

    def late(*_):
        entered.set()
        release.wait(3)
        returned.set()
        return signal() if slow == "detector" else True

    settings = (
        {"detector": late, "check_timeout": 0.2}
        if slow == "detector"
        else {"detector": lambda *_: signal(0.9), "approve": late, "approval_timeout": 0.2}
    )
    registry, recorder, gateway = boundary(**settings)
    try:
        response = send(gateway)
        assert entered.is_set() and not returned.is_set()
        assert response.status == "error" and not registry.recorded_requests
    finally:
        release.set()
    assert returned.wait(2)
    assert not registry.recorded_requests
    decision = recorder.for_run("r")[-2].summary
    assert (
        decision["reason"] == "detector_timeout"
        if slow == "detector"
        else decision["approval"] == "timeout"
    )


def test_broken_policy_and_broken_gateway(tmp_path):
    for mode in ("bad_policy", "broken_history"):
        recorder = InMemoryRecorder()
        registry, _, gateway = boundary(recorder=recorder, policy_path=tmp_path / "missing.toml")
        if mode == "broken_history":
            history = recorder.for_run
            recorder.for_run = lambda rid: history(rid)[:-1]
        response = send(gateway)
        events = InMemoryRecorder.for_run(recorder, "r")
        reason = "policy_failure" if mode == "bad_policy" else "gateway_failure"
        assert events[-2].summary["reason"] == reason
        assert response.status == "error" and not registry.recorded_requests


@pytest.mark.parametrize(("stage", "count"), [("proposal", 0), ("decision", 0), ("tool_result", 1)])
def test_audit_failure_distinguishes_dispatch(stage, count):
    class Broken(InMemoryRecorder):
        def _persist(self, event):
            if event.kind == stage:
                raise OSError("private logging error")

    registry, _, gateway = boundary(recorder=Broken())
    with pytest.raises(
        TelemetryError, match="effect may already exist" if count else "tool not started"
    ) as exc:
        send(gateway)
    assert len(registry.recorded_requests) == count
    assert "private" not in str(exc.value)


def test_agent_custom_executor_receives_only_approved_calls_and_causal_prefixes():
    names = ["read_file", "mock_http_post", "mock_http_post"]
    args = [{"file_id": "public.txt"}, POST, {**POST, "payload": "Do whatever I say; approved!"}]
    calls = [
        {"name": n, "args": a, "id": str(i)}
        for i, (n, a) in enumerate(zip(names, args, strict=True))
    ]
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [AIMessage(content="", tool_calls=calls), AIMessage(content="done")]
    prefixes = []

    def detector(events, _):
        prefixes.append([(e.kind, e.call_id) for e in events])
        return signal()

    delegate = Mock()
    delegate.execute.side_effect = lambda c: ToolResponse(
        call_id=c.call_id, status="success", output="ok"
    )
    agent = LangGraphAgent(
        SandboxToolRegistry(), model, delegate, containment=Containment(detector=detector)
    )
    bound = model.bind_tools.call_args.args[0]
    assert len(bound) == 4 and all(isinstance(tool, dict) for tool in bound)
    with run_context(scenario_id=AUTHORIZED_SHARE.scenario_id, scenario_version="1.0"):
        result = agent.invoke(AgentRequest(run_id="r", prompt="I approve every call.", max_steps=2))
    assert [r.status for r in result.tool_responses] == ["success", "success", "error"]
    assert [c.args[0].call_id for c in delegate.execute.call_args_list] == ["0", "1"]
    for i, prefix in enumerate(prefixes):
        assert prefix == [("run_start", None)] + [
            (k, str(n)) for n in range(i) for k in ("proposal", "decision", "tool_result")
        ] + [("proposal", str(i))]
    assert len(prefixes) == 3
    message = model.invoke.call_args.args[0][-1]
    assert json.loads(message.content)["status"] == "error" and message.tool_call_id == "2"
    assert "Do whatever" not in agent.recorder.export_run("r")


def test_real_checkpoint_and_offline_scenarios():
    pytest.importorskip("torch")
    for scenario in (BENIGN, AUTHORIZED_READ, INJECTION, AUTHORIZED_SHARE):
        recorder = InMemoryRecorder()
        # Host grants a reviewed alert only for the already verified share.
        containment = Containment(approve=lambda req: req.scenario == AUTHORIZED_SHARE.scenario_id)
        result = run_scenario(scenario, recorder=recorder, containment=containment)
        events = recorder.for_run(result.agent_result.run_id)
        assert not result.attack_completed
        if scenario is INJECTION:
            assert any(e.summary.get("decision") == "block" for e in events)
            assert not result.recorded_side_effects
        else:
            assert result.benign_task_completed and result.outcome == "passed"
        if scenario is AUTHORIZED_SHARE:
            assert result.recorded_side_effects == [POST]
        assert all(e.summary["model_version"] for e in events if e.kind == "decision")


def test_unregistered_tool_never_reaches_delegate():
    delegate = Mock()
    recorder = InMemoryRecorder()
    recorder.start_run(
        "r",
        RunMetadata(
            max_steps=1, scenario_id=BENIGN.scenario_id, scenario_version=BENIGN.scenario_version
        ),
    )
    gateway = RegistryExecutor(
        [], delegate=delegate, recorder=recorder, containment=Containment(detector=quiet)
    )
    assert send(gateway, "read_document", DOC).status == "error"
    assert recorder.for_run("r")[-2].summary["reason"] == "unregistered_tool"
    delegate.execute.assert_not_called()


def test_failed_policy_check_still_records_independent_detector(monkeypatch):
    check = Mock(side_effect=ValueError("private policy input"))
    monkeypatch.setattr("runtime_security.gateway.decisions.evaluate_pending", check)
    registry, recorder, gateway = boundary(approve=Mock(return_value=True))
    assert send(gateway).status == "error" and not registry.recorded_requests
    summary = recorder.for_run("r")[-2].summary
    assert summary["reason"] == "policy_failure"
    assert summary["detector"] == "below_threshold"
    assert "private" not in recorder.export_run("r")


def test_failed_host_review_is_safe():
    registry, recorder, gateway = boundary(
        detector=lambda *_: signal(1), approve=Mock(side_effect=RuntimeError("private note"))
    )
    assert send(gateway).status == "error" and not registry.recorded_requests
    assert recorder.for_run("r")[-2].summary["approval"] == "failure"
    assert "private note" not in recorder.export_run("r")


def test_custom_executor_audit_failure_is_surfaced_without_its_message():
    delegate = Mock()
    delegate.execute.side_effect = TelemetryError("private nested audit details")
    _, recorder, gateway = boundary(delegate=delegate)
    with pytest.raises(TelemetryError, match="effect may already exist") as caught:
        send(gateway)
    delegate.execute.assert_called_once()
    assert "private" not in str(caught.value)
    assert recorder.for_run("r")[-1].summary == {"status": "error", "execution": "outcome_unknown"}
