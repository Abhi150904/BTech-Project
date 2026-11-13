"""Record around execution; the supplied executor remains responsible for tools."""

from uuid import uuid4

from runtime_security.contracts import ToolCallEvent, ToolResponse
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
            summary={"mode": "protocol_validation", "decision": "rejected", "reason": reason},
        )
        recorder.record(
            run_id,
            "tool_result",
            **fields,
            summary={"status": "error", "execution": "not_started", "error": {"redacted": True}},
        )


def observe_call(call: ToolCallEvent, execute, recorder) -> ToolResponse:
    name = call.tool_name if call.tool_name in KNOWN_TOOLS else "unrecognized_tool"
    fields = {"call_id": call.call_id, "tool_name": name}
    recorder.record(
        call.run_id,
        "proposal",
        **fields,
        arguments=safe_arguments(call.tool_name, call.arguments),
        summary=resource_metadata(call.tool_name, call.arguments),
    )
    recorder.record(
        call.run_id,
        "decision",
        **fields,
        summary={"mode": "observation", "decision": "not_evaluated"},
    )
    try:
        response = execute(call)
        if response.call_id != call.call_id:
            raise ValueError("Executor returned a mismatched call ID")
    except TelemetryError:
        raise
    except Exception as exc:
        response = ToolResponse(
            call_id=call.call_id, status="error", error=str(exc) or type(exc).__name__
        )
    recorder.record(
        call.run_id,
        "tool_result",
        **fields,
        summary={
            "status": response.status,
            "output": safe_summary(response.output),
            "error": {"redacted": True} if response.error else None,
        },
    )
    return response
