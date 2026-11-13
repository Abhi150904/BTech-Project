"""Ordered, inspectable audit recording with optional append-only JSONL."""

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from runtime_security.telemetry.events import RunMetadata, TrajectoryEvent


class TelemetryError(RuntimeError):
    """Instrumentation failed; a tool effect may already have occurred."""


class _Timeline:
    """Validate serial call lifecycles without executing or interpreting tools."""

    def __init__(self):
        self.stage = None
        self.call_id = None
        self.tool_name = None
        self.calls = set()
        self.metadata = None

    def accept(self, event: TrajectoryEvent):
        if self.stage is None:
            if event.kind != "run_start":
                raise ValueError("Timeline must start with run_start")
            self.metadata = event.metadata
        else:
            if event.metadata != self.metadata:
                raise ValueError("Run metadata changed within a timeline")
            expected = {
                "run_start": {"proposal", "run_complete"},
                "proposal": {"decision"},
                "decision": {"tool_result"},
                "tool_result": {"proposal", "run_complete"},
                "run_complete": set(),
            }[self.stage]
            if event.kind not in expected:
                raise ValueError("Missing or out-of-order lifecycle event")
        if event.kind == "proposal":
            if event.call_id in self.calls:
                raise ValueError("Duplicate tool call ID")
            self.calls.add(event.call_id)
            self.call_id, self.tool_name = event.call_id, event.tool_name
        elif event.kind in {"decision", "tool_result"}:
            if (event.call_id, event.tool_name) != (self.call_id, self.tool_name):
                raise ValueError("Tool event does not match pending call")
        self.stage = event.kind


_metadata = ContextVar("scenario_metadata", default={})


@contextmanager
def run_context(**metadata):
    token = _metadata.set(metadata)
    try:
        yield
    finally:
        _metadata.reset(token)


def current_metadata() -> dict:
    return dict(_metadata.get())


def read_events(path: Path) -> list[TrajectoryEvent]:
    events = []
    sequences = {}
    identifiers = set()
    timelines = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            event = TrajectoryEvent.model_validate_json(line)
            if (
                event.event_id in identifiers
                or event.sequence != sequences.get(event.run_id, 0) + 1
            ):
                raise ValueError("Duplicate event ID or discontinuous run sequence")
            identifiers.add(event.event_id)
            timelines.setdefault(event.run_id, _Timeline()).accept(event)
            sequences[event.run_id] = event.sequence
            events.append(event)
    return events


class InMemoryRecorder:
    def __init__(self):
        self._events: list[TrajectoryEvent] = []
        self._runs: dict[str, RunMetadata] = {}
        self._sequences: dict[str, int] = {}
        self._closed: set[str] = set()
        self._lock = threading.RLock()
        self._failed = False
        self._timelines: dict[str, _Timeline] = {}

    def _persist(self, event: TrajectoryEvent) -> None:
        pass

    def start_run(self, run_id: str, metadata: RunMetadata) -> None:
        with self._lock:
            if run_id in self._runs:
                raise TelemetryError("Run ID already recorded; use a fresh run ID")
            self._runs[run_id] = metadata.model_copy(deep=True)
            self.record(run_id, "run_start")

    def record(self, run_id: str, kind: str, **fields) -> None:
        with self._lock:
            if self._failed or run_id not in self._runs or run_id in self._closed:
                raise TelemetryError("Recorder unavailable or run not active")
            try:
                event = TrajectoryEvent(
                    event_id=str(uuid4()),
                    run_id=run_id,
                    timestamp=datetime.now(timezone.utc),
                    sequence=self._sequences.get(run_id, 0) + 1,
                    kind=kind,
                    metadata=self._runs[run_id],
                    **fields,
                )
                self._timelines.setdefault(run_id, _Timeline()).accept(event)
                self._persist(event)
            except Exception as exc:
                self._failed = True
                raise TelemetryError("Audit write failed; tool effects may already exist") from exc
            self._events.append(event)
            self._sequences[run_id] = event.sequence
            if kind == "run_complete":
                self._closed.add(run_id)

    def for_run(self, run_id: str) -> list[TrajectoryEvent]:
        with self._lock:
            return [e.model_copy(deep=True) for e in self._events if e.run_id == run_id]

    def export_run(self, run_id: str) -> str:
        return "".join(e.model_dump_json() + "\n" for e in self.for_run(run_id))


class JsonlRecorder(InMemoryRecorder):
    """One writer per file. Reopening preserves history; old run IDs cannot resume."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            self._events = read_events(self.path)
            for event in self._events:
                self._runs[event.run_id] = event.metadata
                self._sequences[event.run_id] = event.sequence
                self._closed.add(event.run_id)

    def _persist(self, event: TrajectoryEvent) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(event.model_dump_json() + "\n")
            stream.flush()


def completion_status(answer: str) -> str:
    return {
        "Maximum agent steps reached.": "step_limit",
        "The model provider failed to complete this request.": "provider_error",
        "The model returned no answer.": "empty_answer",
        "Scripted tool execution failed.": "tool_error",
    }.get(answer, "completed")
