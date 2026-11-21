"""Strict adapters for frozen predictions. No training or inference imports."""

import math
from dataclasses import dataclass

from runtime_security.datasets.schema import fingerprint


@dataclass(frozen=True)
class Prediction:
    flagged: bool
    score: float | None = None
    disposition: str | None = None


def probability(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected a finite probability")
    return float(value)


def _digest(value):
    return (
        value
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)
        )
        else None
    )


def identity(report, corpus, manifest, schema, hashes):
    expected = {
        "schema_version": schema,
        "dataset": fingerprint(corpus.model_dump()),
        "manifest": fingerprint(manifest),
        "source": corpus.source,
        "membership": manifest["splits"],
        hashes: corpus.hashes,
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Prediction artifact does not match the frozen corpus and manifest")


def proposal_ids(corpus, manifest, partition):
    members = set(manifest["splits"][partition])
    return {a.proposal for r in corpus.runs if r.run_id in members for a in r.actions}


def baseline_predictions(report, corpus, manifest):
    identity(report, corpus, manifest, "1.0", "source_hashes")
    if report.get("target_definitions", {}).get("pending") != (
        "Phase 5 action.violation_attempted at the proposal"
    ):
        raise ValueError("Unsupported baseline target")
    methods = {}
    for name, method in sorted(report["methods"].items()):
        if method["family"] not in {"policy_only", "ml_only"}:
            raise ValueError("Unsupported baseline family")
        predictions = {}
        raw_threshold = method["calibration"]["threshold"]
        threshold = probability(raw_threshold) if raw_threshold is not None else None
        if set(method["metrics"]) != set(manifest["splits"]):
            raise ValueError("Baseline partitions are incomplete")
        for partition, metrics in method["metrics"].items():
            seen = set()
            for row in metrics["proposals"]:
                pid = row["proposal_id"]
                if pid in predictions:
                    raise ValueError("Duplicate prediction")
                seen.add(pid)
                decision = row["decision"]
                if (decision is not None) != (method["family"] == "policy_only"):
                    raise ValueError("Prediction type does not match method family")
                if decision is not None:
                    if row["score"] is not None:
                        raise ValueError("Policy prediction cannot contain a model score")
                    disposition = decision["disposition"]
                    if disposition not in {"allow", "block", "review"}:
                        raise ValueError("Invalid disposition")
                    score = None
                    flagged = disposition != "allow"
                else:
                    disposition = None
                    score = probability(row["score"])
                    if threshold is None:
                        raise ValueError("Scored predictions need a threshold")
                    flagged = score >= threshold
                if type(row["flagged"]) is not bool or row["flagged"] != flagged:
                    raise ValueError("Saved flag disagrees with saved threshold or decision")
                predictions[pid] = Prediction(flagged, score, disposition)
            if seen != proposal_ids(corpus, manifest, partition):
                raise ValueError("Baseline proposal coverage does not match split")
        methods["phase6/" + name] = {
            "predictions": predictions,
            "threshold": threshold,
            "calibration": method["calibration"]["method"],
            "representation": method["representation"],
            "latency": {
                "saved_summary": report.get("latency_ns", {}).get(name),
                "method": report.get("latency_method"),
                "comparable_added_gateway_latency": False,
            },
        }
    return methods


def neural_predictions(report, corpus, manifest, baseline=None):
    identity(report, corpus, manifest, "phase7-v1", "hashes")
    if report.get("target") != "violation_attempted":
        raise ValueError("Unsupported neural target")
    if baseline is not None and report.get("phase6_report_hash") != fingerprint(baseline):
        raise ValueError("Neural report references a different baseline artifact")
    methods = {}
    for name, variant in sorted(report["variants"].items()):
        threshold = probability(variant["calibration"]["threshold"])
        predictions = {}
        for row in variant["proposals"]:
            pid, score = row["id"], probability(row["score"])
            if pid in predictions:
                raise ValueError("Duplicate neural prediction")
            predictions[pid] = Prediction(score >= threshold, score)
        if set(predictions) != proposal_ids(corpus, manifest, "test"):
            raise ValueError("Neural proposal coverage does not match test split")
        methods["phase7/" + name] = {
            "predictions": predictions,
            "threshold": threshold,
            "calibration": variant["calibration"]["method"],
            "representation": name,
            "latency": {
                "saved_summary": variant.get("latency_ns"),
                "comparable_added_gateway_latency": False,
            },
        }
    return methods


def gateway_predictions(events, corpus):
    """Caller verifies the JSONL byte hash against the corpus before this join."""
    runs = {r.run_id: r for r in corpus.runs if not r.issues}
    all_run_ids = {r.run_id for r in corpus.runs}
    seen_events = set()
    predictions = {}
    policies, detectors, diagnostics = {}, {}, []
    pending = {}
    ignored = 0
    for event in events:
        rid = fingerprint(event.run_id)
        if rid not in all_run_ids:
            raise ValueError("Unknown telemetry run")
        if rid not in runs:
            ignored += 1
            continue
        run = runs[rid]
        seen_events.add(fingerprint(event.event_id))
        mapped = next((e for e in run.events if e.event_id == fingerprint(event.event_id)), None)
        if (
            mapped is None
            or mapped.sequence != event.sequence
            or mapped.features.kind != event.kind
            or mapped.call_id != (fingerprint(event.call_id) if event.call_id else None)
        ):
            raise ValueError("Telemetry event does not match corpus event")
        if event.kind == "proposal":
            pending[rid] = mapped.event_id
        if event.kind != "decision" or event.summary.get("mode") != "containment-v1":
            continue
        pid = pending[rid]
        disposition = event.summary.get("decision")
        if disposition not in {"allow", "block", "review"} or pid in predictions:
            raise ValueError("Invalid or duplicate gateway decision")
        # Final host-approved disposition, not the detector's initial alert.
        predictions[pid] = Prediction(disposition != "allow", disposition=disposition)
        policy = event.summary.get("policy_decision")
        if policy in {"allow", "block", "review"}:
            policies[pid] = Prediction(policy != "allow", disposition=policy)
        signal = event.summary.get("detector")
        score, threshold = None, None
        if signal in {"alert", "below_threshold"}:
            score = probability(event.summary.get("model_score"))
            threshold = probability(event.summary.get("model_threshold"))
            if (score >= threshold) != (signal == "alert"):
                raise ValueError("Detector status disagrees with saved signal")
            detectors[pid] = Prediction(score >= threshold, score)
        # Whitelist status values; never copy free-form reasons or tool content.
        approval = event.summary.get("approval")
        diagnostics.append(
            {
                "proposal_id": pid,
                "policy_available": pid in policies,
                "detector_available": pid in detectors,
                "detector_score": score,
                "detector_threshold": threshold,
                "detector_status": signal
                if signal
                in {
                    "alert",
                    "below_threshold",
                    "detector_timeout",
                    "detector_failure",
                    "unavailable",
                    "not_evaluated",
                }
                else "unknown",
                "approval": approval
                if approval
                in {
                    "approved",
                    "declined",
                    "timeout",
                    "failure",
                    "not_requested",
                }
                else "unknown",
                "model_version": _digest(event.summary.get("model_version")),
                "policy_hash": _digest(event.summary.get("policy_hash")),
            }
        )
    if seen_events != {e.event_id for r in runs.values() for e in r.events}:
        raise ValueError("Telemetry event coverage differs from corpus")
    return {
        "phase8/" + name: {
            "predictions": values,
            "threshold": None,
            "calibration": "recorded_final_disposition"
            if name == "gateway"
            else "recorded_per_action_signal",
            "representation": name,
            "diagnostics": diagnostics,
            "ignored_excluded_events": ignored,
            "latency": {"saved_summary": None, "reason": "no_monotonic_per_call_timing"},
        }
        for name, values in (
            ("gateway", predictions),
            ("recorded_policy", policies),
            ("recorded_detector", detectors),
        )
    }
