"""Combine trusted policy, an offline detector and optional host review."""

import math
import re
from concurrent.futures import Future
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from threading import BoundedSemaphore, Thread
from time import monotonic

from runtime_security.policies.engine import DEFAULT_POLICY, POLICY_VERSION, ROOT, evaluate_pending
from runtime_security.policies.rules import Decision

CHECKPOINT = ROOT / "research/phase7-scripted-v1/enriched.json"
CHECKPOINT_VERSION = "b76482ce38952a8921928a7613bf82469c0817d1cf0df4bf4947082472c32e83"


@lru_cache(maxsize=1)
def _checkpoint():
    from runtime_security.detectors.inference import load_checkpoint

    loaded = load_checkpoint(CHECKPOINT)
    if loaded.model_version != CHECKPOINT_VERSION:
        raise ValueError("Unexpected checkpoint")
    return loaded


def checkpoint_score(events, policy_data):
    from runtime_security.detectors.inference import score_prefix_data

    return score_prefix_data(_checkpoint(), events, policy_data)


# Limit outstanding slow checks across runs. Workers can return data only.
_slots = BoundedSemaphore(8)


def _bounded(seconds, fn, *args):
    deadline = monotonic() + seconds
    if not _slots.acquire(blocking=False):
        raise TimeoutError
    future = Future()

    def worker():
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)
        finally:
            _slots.release()

    try:
        Thread(target=worker, daemon=True).start()
    except BaseException:
        _slots.release()
        raise
    value = future.result(timeout=max(0.0, deadline - monotonic()))
    if monotonic() >= deadline:
        raise TimeoutError
    return value


@dataclass(frozen=True)
class ReviewRequest:
    run_id: str
    call_id: str
    sequence: int
    scenario: str | None
    tool: str
    target: str | None
    score: float
    threshold: float
    model_version: str


def _signal(raw):
    score, threshold, version = (raw[k] for k in ("score", "threshold", "model_version"))
    for n in (score, threshold):
        if type(n) not in (float, int) or not math.isfinite(n) or not 0 <= n <= 1:
            raise ValueError("Invalid model signal")
    if not isinstance(version, str) or not re.fullmatch("[0-9a-f]{64}", version):
        raise ValueError("Invalid model version")
    return score, threshold, version


class Containment:
    def __init__(
        self,
        *,
        detector=checkpoint_score,
        approve=None,
        policy_path=DEFAULT_POLICY,
        check_timeout=15.0,
        approval_timeout=5.0,
    ):
        for timeout in (check_timeout, approval_timeout):
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Timeout must be finite and positive")
        self.detector = detector
        self.approve = approve
        self.policy_path = policy_path
        self.check_timeout = check_timeout
        self.approval_timeout = approval_timeout

    def decide(self, events, call, *, registered):
        policy_error = None
        try:
            data = _bounded(self.check_timeout, self.policy_path.read_bytes)
        except Exception as exc:
            return {
                "mode": "containment-v1",
                "policy_version": POLICY_VERSION,
                "policy_decision": "unavailable",
                "rules": [],
                "decision": "block",
                "detector": "unavailable",
                "approval": "not_requested",
                "reason": "policy_timeout" if isinstance(exc, TimeoutError) else "policy_failure",
            }
        try:
            policy = _bounded(self.check_timeout, evaluate_pending, events, call, data)
            if not isinstance(policy, Decision) or policy.disposition not in {
                "allow",
                "block",
                "review",
            }:
                raise ValueError("Invalid policy decision")
        except Exception as exc:
            policy_error = "policy_timeout" if isinstance(exc, TimeoutError) else "policy_failure"
            policy = Decision("block", (), "Policy unavailable")
        summary = {
            "mode": "containment-v1",
            "policy_version": POLICY_VERSION,
            "policy_hash": sha256(data).hexdigest(),
            "policy_decision": policy.disposition if policy_error is None else "unavailable",
            "rules": list(policy.rules),
            "decision": "block",
            "approval": "not_requested",
        }
        try:
            raw = _bounded(self.check_timeout, self.detector, events, data)
            score, threshold, version = _signal(raw)
            alert = score >= threshold
            summary.update(
                model_score=score,
                model_threshold=threshold,
                model_version=version,
                detector="alert" if alert else "below_threshold",
            )
        except Exception as exc:
            failure = "detector_timeout" if isinstance(exc, TimeoutError) else "detector_failure"
            return {**summary, "detector": failure, "reason": policy_error or failure}
        if policy.disposition == "block":
            return {**summary, "reason": policy_error or "hard_block"}
        if not registered:
            return {**summary, "decision": "review", "reason": "unregistered_tool"}
        if policy.disposition == "review":
            # Host review of a model alert never expands the trusted task's scope.
            return {**summary, "decision": "review", "reason": "scope_unverified"}
        if not alert:
            return {**summary, "decision": "allow", "reason": "checks_passed"}
        summary.update(decision="review", reason="model_alert")
        if self.approve is None:
            return summary
        event = events[-1]
        target = next(
            (
                v
                for k, v in event.arguments.items()
                if k in {"file_id", "document_id", "destination"} and isinstance(v, str)
            ),
            None,
        )
        request = ReviewRequest(
            call.run_id,
            call.call_id,
            event.sequence,
            event.metadata.scenario_id,
            event.tool_name,
            target,
            score,
            threshold,
            version,
        )
        try:
            approved = _bounded(self.approval_timeout, self.approve, request) is True
        except Exception as exc:
            status = "timeout" if isinstance(exc, TimeoutError) else "failure"
            return {**summary, "approval": status}
        if approved:
            return {
                **summary,
                "approval": "approved",
                "decision": "allow",
                "reason": "host_approved",
            }
        return {**summary, "approval": "declined"}
