"""Trusted proposal-time policy using Phase 4 features and the Phase 6 rules."""

import tomllib
from pathlib import Path

from runtime_security.features import extract_run
from runtime_security.features.metadata import Policy
from runtime_security.policies.rules import Decision, evaluate_rules
from runtime_security.tools.registry import DocumentInput, FileInput, PostInput, SearchInput

SCHEMAS = {
    "read_document": DocumentInput,
    "read_file": FileInput,
    "mock_http_post": PostInput,
    "search_files": SearchInput,
}

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = ROOT / "configs/policy_metadata.toml"
POLICY_VERSION = "phase8-sandbox-v1"


def evaluate_pending(events, call, policy_data):
    # Import the reviewed scenario grant after agent modules have initialized.
    from runtime_security.scenarios import AUTHORIZED_SHARE

    policy = Policy.model_validate(tomllib.loads(policy_data.decode("utf8")))
    rows = extract_run(events, policy)
    pending = events[-1]
    payload_check = None
    if (
        pending.tool_name == "mock_http_post"
        and pending.metadata.scenario_id == AUTHORIZED_SHARE.scenario_id
        and pending.metadata.scenario_version == AUTHORIZED_SHARE.scenario_version
        and pending.arguments.get("destination") == AUTHORIZED_SHARE.authorized_destination
        and isinstance(call.arguments.get("payload"), str)
    ):
        payload_check = (call.arguments["payload"], AUTHORIZED_SHARE.authorized_payload_marker)
    decision = evaluate_rules([r.features for r in rows], payload_check=payload_check)
    if decision.disposition == "block":
        return decision
    if schema := SCHEMAS.get(call.tool_name):
        try:
            schema.model_validate(call.arguments)
        except ValueError:
            return Decision("block", ("invalid_arguments",), "Arguments outside tool schema")
    if pending.tool_name == "mock_http_post" and any(
        e.kind == "decision"
        and e.tool_name == "mock_http_post"
        and e.summary.get("decision") == "allow"
        for e in events[:-1]
    ):
        # An executor error can still follow a side effect. Consume on permission.
        return Decision("block", ("share_grant_used",), "Single-send grant already consumed")
    return decision
