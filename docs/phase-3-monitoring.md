# Phase 3: structured behavioral trajectories

This implements the guide's Phase 3 instrumentation and the implementation
playbook's JSONL persistence and replay requirements. Phase 2 was merged in
`cb50f19` before this branch was created. No detection, scoring or authorization
policy is added. Existing request/result contracts and scenario outcome
predicates are preserved. New versioned Pydantic types live in
`telemetry/events.py` and inherit the shared strict `Contract` base.

## Event lifecycle

Each run has `run_start`, then zero or more ordered groups of `proposal`,
`decision`, `tool_result`, followed by `run_complete`. Events contain a UUID
`event_id`, caller's `run_id`, UTC timestamp, contiguous per-run `sequence`,
schema version, and copied `RunMetadata` (`max_steps`, optional scenario ID,
scenario version and model name). Tool events preserve `call_id` and a safe tool
name. Sequence is the ordering authority; timestamps are wall-clock observations.

`gateway/interceptor.py` wraps the existing execution callback. Proposals are
recorded before execution, including calls that subsequently fail validation or
refer to unknown tools. The decision event says `observation / not_evaluated`:
it records that this baseline does not evaluate security policy. It is not an
ALLOW verdict. The telemetry recorder never invokes tools. The wrapper delegates
to the existing executor and returns its real result unchanged. Executor errors
and mismatched call IDs become paired error responses as in Phase 2.

Provider tool calls rejected before execution are also recorded as proposal,
protocol-validation decision, and error-result events with `execution: not_started`.
Those events use generated call IDs and sanitized arguments; provider-supplied
call IDs and malformed raw argument strings are not copied into telemetry.

LangGraphAgent uses an in-memory recorder by default, exposed as `agent.recorder`.
The deterministic scenario adapter uses the same wrapper. ScenarioRunner passes
run metadata through a scoped ContextVar which resets on exit; labels are never
added to prompts or tool results. A plain Agent-compatible stub remains supported
but must adopt the boundary hook to produce runtime telemetry. The old
EventCollector remains a sanitized, grouped post-run compatibility view; it is
not a chronological trajectory.

## Redaction and failure policy

Telemetry retains only the fixed allowlisted sandbox document/file IDs and mock
destinations. Payloads, search queries, unknown argument names/values, tool
outputs, and exception messages are omitted or replaced with type/redacted
summaries. Unknown tool names become `unrecognized_tool`. The sensitivity of
`sensitive.txt` comes from a fixed trusted mapping, not text returned by a tool.
Other unmapped resources have `unknown` sensitivity. Raw prompts and final
answers are not logged. Run/call IDs and scenario/model labels are caller-owned
identifiers and must not contain secrets. Recorder.record is a trusted low-level
API; custom integrations must sanitize data before submitting event fields.

Actual AgentResult/ScenarioResult and the mock tool ledger retain original tool
data for execution and assessment. They are not safe general-purpose log exports.
Use the telemetry export/replay interfaces for sanitized inspection.

Recorder validation or persistence failure raises TelemetryError and marks the
recorder unavailable. A failed pre-execution audit write aborts the invocation
before the tool runs. A failed result write occurs *after* a possible side effect;
it aborts without retry or rollback and must not be treated as tool success.
ScenarioRunner surfaces it as an error and retains actual newly recorded effects.
This is instrumentation failure handling, not a security blocking decision.
An interrupted run can end without run_complete; replay preserves that incomplete
prefix. Provider errors and step exhaustion emit explicit completion statuses.
Completion status currently recognizes the existing agent's fixed terminal
messages because AgentResult has no structured termination field.

## Inspect, persist and replay

Offline example (no credentials or network):

```python
from pathlib import Path
from runtime_security.scenarios import INJECTION
from runtime_security.scenarios.lab import run_scenario
from runtime_security.telemetry.recorder import JsonlRecorder

recorder = JsonlRecorder(Path("audit.jsonl"))
result = run_scenario(INJECTION, recorder=recorder)
run_id = result.agent_result.run_id
print(run_id)
events = recorder.for_run(run_id)  # Detached Pydantic snapshots in sequence order
safe_jsonl = recorder.export_run(run_id)
```

```powershell
python -m runtime_security.telemetry.replay audit.jsonl RUN_ID
# After reinstalling the editable package, this entry point is also available:
runtime-security-replay audit.jsonl RUN_ID
# Optional OpenAI-backed observation (requires configured credentials):
python -m runtime_security.cli "Read summary.txt and summarize it" --telemetry audit.jsonl
```

Replay prints only that run's JSON events and never re-executes tools. An unknown
run ID, invalid schema, duplicate event/call ID, metadata drift, sequence gap or
invalid lifecycle produces an error. JSONL uses append mode and flushes/closes
each write. Existing data is validated when reopening. Previously used run IDs
cannot resume or be reused; use a fresh ID. The parent directory must exist.

Use **one recorder/writer per JSONL file**. Threads using the same recorder are
serialized by a lock; independent processes or recorder objects sharing a file
are unsupported. No filesystem locking, fsync durability, automatic repair of
truncated logs, rotation, retention limits or database is provided. In-memory
records and JSONL replay currently load complete history; this is a research-scale
implementation. An incomplete valid prefix is inspectable, while malformed final
lines are rejected for manual investigation. Do not retry a tool solely because
its result event is missing.

Run `.venv/Scripts/python -m pytest -q` and
`.venv/Scripts/python -m ruff check .` for offline verification.
