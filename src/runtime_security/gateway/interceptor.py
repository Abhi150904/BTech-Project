"""Record around execution; the supplied executor remains responsible for tools."""

from runtime_security.contracts import ToolCallEvent, ToolResponse
from runtime_security.telemetry.events import resource_metadata, safe_arguments, safe_summary
from runtime_security.telemetry.recorder import TelemetryError


def observe_call(call: ToolCallEvent, execute, recorder) -> ToolResponse:
    name = (
        call.tool_name
        if call.tool_name
        in {
            "read_document",
            "read_file",
            "search_files",
            "mock_http_post",
        }
        else "unrecognized_tool"
    )
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
