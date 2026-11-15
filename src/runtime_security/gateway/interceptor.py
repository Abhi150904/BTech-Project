"""Commit the pre-call decision before dispatching an allowed call to its executor."""

from uuid import uuid4

from runtime_security.contracts import ToolCallEvent, ToolResponse
from runtime_security.policies.engine import POLICY_VERSION
from runtime_security.telemetry.events import resource_metadata, safe_arguments, safe_summary
from runtime_security.telemetry.recorder import TelemetryError

KNOWN_TOOLS = {"read_document", "read_file", "search_files", "mock_http_post"}


def record_rejected_calls(run_id: str, raw_calls: list[dict], recorder, reason: str) -> None:
    """Record rejected provider proposals without retaining raw IDs or arguments."""
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            raw_call = {"args": raw_call}
        raw_name = raw_call.get("name")
        tool_name = (
            raw_name
            if isinstance(raw_name, str) and raw_name in KNOWN_TOOLS
            else "unrecognized_tool"
        )
        fields = {"call_id": f"rejected-{uuid4()}", "tool_name": tool_name}
        arguments = raw_call.get("args")
        recorder.record(
            run_id,
            "proposal",
            **fields,
            arguments=safe_arguments(raw_name if isinstance(raw_name, str) else "", arguments),
            summary=resource_metadata(raw_name if isinstance(raw_name, str) else "", arguments),
        )
        recorder.record(
            run_id,
            "decision",
            **fields,
            summary={
                "mode": "protocol_validation",
                "decision": "block",
                "reason": reason,
                "policy_version": POLICY_VERSION,
                "policy_decision": "not_evaluated",
                "rules": [],
                "detector": "not_evaluated",
            },
        )
        recorder.record(
            run_id,
            "tool_result",
            **fields,
            summary={"status": "error", "execution": "not_started", "error": {"redacted": True}},
        )


def _audit(recorder, run_id, kind, *, dispatched=False, **fields):
    try:
        recorder.record(run_id, kind, **fields)
    except Exception:
        detail = (
            "executor invoked; an effect may already exist" if dispatched else "tool not started"
        )
        raise TelemetryError(f"Audit write failed: {detail}") from None


def _record_proposal(call, recorder):
    name = call.tool_name if call.tool_name in KNOWN_TOOLS else "unrecognized_tool"
    fields = {"call_id": call.call_id, "tool_name": name}
    _audit(
        recorder,
        call.run_id,
        "proposal",
        **fields,
        arguments=safe_arguments(call.tool_name, call.arguments),
        summary=resource_metadata(call.tool_name, call.arguments),
    )
    return fields


def _execute_recorded(call, execute, recorder, fields):
    try:
        response = execute(call)
        if not isinstance(response, ToolResponse) or response.call_id != call.call_id:
            raise ValueError("Executor returned a mismatched call ID")
        if response.status == "error":
            response = ToolResponse(
                call_id=call.call_id, status="error", error="Tool execution failed."
            )
    except TelemetryError:
        _audit(
            recorder,
            call.run_id,
            "tool_result",
            dispatched=True,
            **fields,
            summary={"status": "error", "execution": "outcome_unknown"},
        )
        raise TelemetryError("Executor audit failed; an effect may already exist") from None
    except Exception:
        response = ToolResponse(
            call_id=call.call_id, status="error", error="Tool execution failed."
        )
    _audit(
        recorder,
        call.run_id,
        "tool_result",
        dispatched=True,
        **fields,
        summary={
            "status": response.status,
            "output": safe_summary(response.output),
            "error": {"redacted": True} if response.error else None,
        },
    )
    return response


def enforce_call(
    call: ToolCallEvent, execute, recorder, containment, *, registered: bool
) -> ToolResponse:
    call = call.model_copy(deep=True)
    fields = _record_proposal(call, recorder)
    try:
        events = recorder.for_run(call.run_id)
        pending = events[-1]
        if (
            pending.kind != "proposal"
            or pending.call_id != call.call_id
            or pending.run_id != call.run_id
            or pending.tool_name != fields["tool_name"]
            or pending.arguments != safe_arguments(call.tool_name, call.arguments)
            or any(e.sequence != i or e.run_id != call.run_id for i, e in enumerate(events, 1))
        ):
            raise ValueError("Mismatched proposal prefix")
        decision = containment.decide(events, call, registered=registered)
    except Exception:
        decision = {
            "mode": "containment-v1",
            "policy_version": POLICY_VERSION,
            "approval": "not_requested",
            "decision": "block",
            "reason": "gateway_failure",
            "policy_decision": "unavailable",
            "detector": "unavailable",
            "rules": [],
        }
    _audit(recorder, call.run_id, "decision", **fields, summary=decision)
    if decision["decision"] == "allow":
        return _execute_recorded(call, execute, recorder, fields)
    _audit(
        recorder,
        call.run_id,
        "tool_result",
        **fields,
        summary={"status": "error", "execution": "not_started"},
    )
    return ToolResponse(
        call_id=call.call_id,
        status="error",
        error="Gateway refused this tool call. The tool was not executed.",
    )
