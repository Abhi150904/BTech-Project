"""Versioned audit contracts; experiment labels never become tool inputs."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from runtime_security.contracts import Contract, JsonValue


class RunMetadata(Contract):
    max_steps: int = Field(gt=0)
    scenario_id: str | None = None
    scenario_version: str | None = None
    model_name: str | None = None


class TrajectoryEvent(Contract):
    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    timestamp: datetime
    sequence: int = Field(ge=1)
    kind: Literal["run_start", "proposal", "decision", "tool_result", "run_complete"]
    metadata: RunMetadata
    call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    summary: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_event(self):
        if self.timestamp.tzinfo is None:
            raise ValueError("Event timestamps must include timezone")
        if self.kind in {"proposal", "decision", "tool_result"}:
            if not self.call_id or not self.tool_name:
                raise ValueError("Tool events require call ID and tool name")
        elif self.call_id is not None or self.tool_name is not None:
            raise ValueError("Run events cannot reference a tool call")
        return self


def safe_summary(value) -> dict:
    """Never retain raw values, reprs, exception messages, or content hashes."""
    return {"type": type(value).__name__, "redacted": True}


def safe_arguments(tool_name: str, arguments: object) -> dict:
    if not isinstance(arguments, dict):
        return {"malformed": safe_summary(arguments)}
    allowed = {
        "document_id": {"summary.txt", "injection.txt"},
        "file_id": {"public.txt", "sensitive.txt"},
        "destination": {"https://collector.invalid/upload", "https://approved.invalid/report"},
    }
    expected = {
        "read_document": {"document_id"},
        "read_file": {"file_id"},
        "search_files": {"query"},
        "mock_http_post": {"destination", "payload"},
    }.get(tool_name, set())
    result = {}
    for key in expected.intersection(arguments):
        value = arguments[key]
        result[key] = (
            value
            if isinstance(value, str) and value in allowed.get(key, set())
            else safe_summary(value)
        )
    if set(arguments) - expected:
        result["unrecognized_arguments"] = {"redacted": True}
    return result


def resource_metadata(tool_name: str, arguments: object) -> dict:
    """Fixed fixture mapping, independent of model-supplied classifications."""
    if tool_name == "read_file" and isinstance(arguments, dict):
        value = arguments.get("file_id")
        return {
            "resource_sensitivity": (
                "synthetic_sensitive"
                if value == "sensitive.txt"
                else "public"
                if value == "public.txt"
                else "unknown"
            )
        }
    return {"resource_sensitivity": "unknown"}
