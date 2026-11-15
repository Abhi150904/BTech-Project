"""Proposal-time policy simulation using reviewed Phase 4 target classifications."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from runtime_security.features.extractor import Features


@dataclass(frozen=True)
class Decision:
    disposition: Literal["allow", "block", "review"]
    rules: tuple[str, ...]
    reason: str


def evaluate_rules(
    prefix: Sequence[Features], *, payload_check: tuple[str, str] | None = None
) -> Decision:
    """Optional (pending payload, trusted permitted payload) from a pre-call observer."""
    if not prefix or prefix[-1].kind != "proposal":
        raise ValueError("Expected a prefix ending at a proposal")
    pending = prefix[-1]
    matched = []
    if pending.operation == "read" and pending.sensitivity == "synthetic_sensitive":
        if pending.authorization == "unauthorized":
            matched.append("sensitive_read_out_of_scope")
    if pending.operation == "send":
        if pending.destination_trust == "unapproved":
            matched.append("unapproved_destination")
        if pending.authorization == "unauthorized":
            matched.append("send_out_of_scope")
        if payload_check is not None and payload_check[0] != payload_check[1]:
            matched.append("payload_out_of_scope")
    if matched:
        return Decision("block", tuple(matched), "Proposal violates a named policy rule")
    if pending.authorization in ("unknown", "not_applicable") or pending.tool == "unknown":
        return Decision("review", ("unknown_scope",), "Tool or task authorization unresolved")
    if pending.operation == "read" and pending.sensitivity == "unknown":
        return Decision("review", ("unknown_sensitivity",), "Read sensitivity unresolved")
    if pending.operation == "send":
        if pending.destination_trust != "approved" or payload_check is None:
            return Decision(
                "review", ("unverified_send",), "Destination or payload scope unverified"
            )
    if pending.authorization != "authorized":
        return Decision("review", ("outside_grant",), "Action outside the known grant")
    return Decision("allow", (), "Proposal within observed grant")
