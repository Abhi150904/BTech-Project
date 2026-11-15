"""Extract one prefix-safe semantic row per Phase 3 event."""

from collections.abc import Iterable
from dataclasses import dataclass

from runtime_security.features.metadata import Grants, Policy
from runtime_security.telemetry.events import TrajectoryEvent
from runtime_security.telemetry.recorder import _Timeline

NA = "not_applicable"
UNKNOWN = "unknown"
# Trusted tool semantics: operation, class, argument, policy catalog.
TOOLS = {
    "read_document": ("read", "document", "document_id", "documents"),
    "read_file": ("read", "file", "file_id", "files"),
    "mock_http_post": ("send", "destination", "destination", "destinations"),
}


@dataclass(frozen=True)
class Action:
    tool: str = NA
    operation: str = NA
    resource_class: str = NA
    sensitivity: str = NA
    destination_trust: str = NA
    authorization: str = NA


def _resolve(event: TrajectoryEvent, policy: Policy, grants: Grants | None) -> Action:
    tool = event.tool_name
    auth = UNKNOWN
    if tool == "search_files":
        if grants is not None:
            auth = "authorized" if grants.search_files else "unauthorized"
        return Action(tool, "search", "document", NA, NA, auth)
    if tool not in TOOLS:
        return Action(UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN)
    operation, resource_class, key, catalog_name = TOOLS[tool]
    target = event.arguments.get(key)
    catalog = getattr(policy, catalog_name)
    classification = UNKNOWN
    if isinstance(target, str) and target in catalog:
        classification = catalog[target]
        if grants is not None:
            auth = "authorized" if target in getattr(grants, tool) else "unauthorized"
    if operation == "send":
        return Action(tool, operation, resource_class, NA, classification, auth)
    return Action(tool, operation, resource_class, classification, NA, auth)


@dataclass(frozen=True)
class Features(Action):
    kind: str = NA
    previous_tool: str = NA
    transition: str = NA
    result_status: str = NA
    mismatch: str = NA
    prior_sensitive_attempts: int = 0
    prior_sensitive_successes: int = 0
    send_after_sensitive_read: bool = False


@dataclass(frozen=True)
class FeatureRecord:
    run_id: str
    event_id: str
    sequence: int
    features: Features


def extract_run(events: Iterable[TrajectoryEvent], policy: Policy) -> list[FeatureRecord]:
    ordered = sorted(events, key=lambda event: event.sequence)
    if not ordered:
        return []
    first = ordered[0]
    versions = policy.tasks.get(first.metadata.scenario_id, {})
    grants = versions.get(first.metadata.scenario_version)
    timeline = _Timeline()
    seen_ids = set()
    attempts = successes = 0
    previous = "start"
    action = Action()
    rows = []
    for index, event in enumerate(ordered, 1):
        if event.run_id != first.run_id or event.sequence != index or event.event_id in seen_ids:
            raise ValueError("Expected one run with contiguous sequences and unique event IDs")
        seen_ids.add(event.event_id)
        timeline.accept(event)
        transition = mismatch = status = NA
        proposal = event.kind == "proposal"
        if proposal:
            action = _resolve(event, policy, grants)
            transition = f"{previous}->{action.tool}"
            mismatch = {"authorized": "no", "unauthorized": "yes"}.get(
                action.authorization, UNKNOWN
            )
        elif event.kind == "tool_result":
            observed = event.summary.get("status")
            status = observed if observed in ("success", "error") else UNKNOWN
        elif event.kind in ("run_start", "run_complete"):
            action = Action()
        values = Features(
            **vars(action),
            kind=event.kind,
            previous_tool=previous,
            transition=transition,
            result_status=status,
            mismatch=mismatch,
            prior_sensitive_attempts=attempts,
            prior_sensitive_successes=successes,
            send_after_sensitive_read=proposal and action.operation == "send" and successes > 0,
        )
        rows.append(FeatureRecord(event.run_id, event.event_id, index, values))
        sensitive = action.operation == "read" and action.sensitivity == "synthetic_sensitive"
        if proposal:
            attempts += int(sensitive)
            previous = action.tool
        if sensitive and status == "success":
            successes += 1
    return rows
