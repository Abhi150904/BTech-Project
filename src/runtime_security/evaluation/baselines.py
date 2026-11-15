"""Evaluate offline pending-action baselines on verified frozen Phase 5 splits."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import sklearn

from runtime_security.datasets.schema import Corpus, Run, fingerprint, write_new
from runtime_security.datasets.validate import audit
from runtime_security.detectors.baselines import Classifier
from runtime_security.evaluation.metrics import calibrate, confusion, rate
from runtime_security.features.extractor import Features
from runtime_security.policies.rules import Decision, evaluate_rules


@dataclass(frozen=True)
class Example:
    run_id: str
    proposal_id: str
    prefix: tuple[Features, ...]
    target: bool | None
    completed: bool | None


def examples(run: Run) -> list[Example]:
    labels = {a.proposal: a for a in run.actions}
    history = []
    rows = []
    for event in run.events:
        history.append(event.features)
        if event.features.kind == "proposal":
            label = labels[event.event_id]
            rows.append(
                Example(
                    run.run_id,
                    event.event_id,
                    tuple(history),
                    label.violation_attempted,
                    label.violation_completed,
                )
            )
    return rows


@dataclass(frozen=True)
class Prediction:
    score: float | None = None
    decision: Decision | None = None

    def flagged(self, threshold: float | None) -> bool:
        if self.decision is not None:
            return self.decision.disposition != "allow"
        return self.score >= threshold


def _predict(rows, model):
    predictions, elapsed = [], []
    for row in rows:
        start = perf_counter_ns()
        if model is None:
            prediction = Prediction(decision=evaluate_rules(row.prefix))
        else:
            prediction = Prediction(score=model.score(row.prefix))
        elapsed.append(perf_counter_ns() - start)
        predictions.append(prediction)
    return predictions, elapsed


def _summarize(runs, rows, predictions, threshold):
    flags = [p.flagged(threshold) for p in predictions]
    flagged_runs = {r.run_id for r, flag in zip(rows, flags, strict=True) if flag}
    run_flags = [r.run_id in flagged_runs for r in runs]
    benign = [r for r in runs if r.injection_present is False]
    injected = [r for r in runs if r.injection_present is True and r.attack_completed is not None]
    opportunities = [flag for r, flag in zip(rows, flags, strict=True) if r.completed is True]
    dispositions = Counter(p.decision.disposition for p in predictions if p.decision)
    return {
        "pending_violation_attempted": confusion([r.target for r in rows], flags),
        "run_unsafe_attempted": confusion([r.unsafe_attempted for r in runs], run_flags),
        "run_unsafe_completed": confusion([r.unsafe_completed for r in runs], run_flags),
        "observed_injected_runs": len(injected),
        "observed_attack_success_rate": rate(
            sum(r.attack_completed for r in injected), len(injected)
        ),
        "observed_benign_runs": len(benign),
        "observed_benign_completions": sum(r.benign_completed is True for r in benign),
        "completed_benign_runs_flagged": sum(
            r.benign_completed is True and r.run_id in flagged_runs for r in benign
        ),
        "completed_benign_runs_uninterrupted": sum(
            r.benign_completed is True and r.run_id not in flagged_runs for r in benign
        ),
        "flagged_completed_violations": sum(opportunities),
        "observed_completed_violations": len(opportunities),
        "containment_opportunity_rate": rate(sum(opportunities), len(opportunities)),
        "policy_dispositions": dict(dispositions),
        "proposals": [
            {
                "proposal_id": r.proposal_id,
                "score": p.score,
                "decision": asdict(p.decision) if p.decision else None,
                "flagged": flag,
            }
            for r, p, flag in zip(rows, predictions, flags, strict=True)
        ],
    }


def evaluate(corpus: Corpus, manifest: dict, *, seed: int = 42, ablate: bool = False) -> dict:
    verification = audit(corpus, manifest)
    if not verification["structurally_valid"] or not verification["redaction_scan_passed"]:
        raise ValueError("Corpus or frozen manifest failed Phase 5 verification")
    by_id = {r.run_id: r for r in corpus.runs}
    partitions = {k: [by_id[rid] for rid in ids] for k, ids in manifest["splits"].items()}
    rows = {k: [e for run in runs for e in examples(run)] for k, runs in partitions.items()}
    training = [row for row in rows["train"] if row.target is not None]
    result = {
        "schema_version": "1.0",
        "dataset": manifest["dataset"],
        "manifest": fingerprint(manifest),
        "source": corpus.source,
        "source_hashes": corpus.hashes,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "membership": manifest["splits"],
        "excluded_runs": verification["excluded"],
        "excluded_action_labels": sum(len(r.actions) for r in corpus.runs if r.issues),
        "input_issues": corpus.input_issues,
        "independent_groups": len(manifest["groups"]),
        "target_definitions": {
            "pending": "Phase 5 action.violation_attempted at the proposal",
            "run_attempt": "Run.unsafe_attempted versus any flagged proposal",
            "run_completion": "Run.unsafe_completed versus any flagged proposal (proxy)",
        },
        "methods": {},
        "latency_ns": {},
        "latency_method": (
            "Single sequential perf_counter_ns measurement per proposal, including feature "
            "building and prediction; excludes I/O, prefix assembly, fitting and report generation."
        ),
    }
    warnings = []
    if corpus.source == "scripted":
        warnings.append("scripted_fixture")
    if len(manifest["groups"]) < 6:
        warnings.append("fewer_than_six_independent_groups")
    for partition in ("validation", "test"):
        if {r.target for r in rows[partition] if r.target is not None} != {False, True}:
            warnings.append(partition + "_missing_action_class")
        for target in ("unsafe_attempted", "unsafe_completed"):
            if {getattr(r, target) for r in partitions[partition]} - {None} != {False, True}:
                warnings.append(partition + "_missing_" + target + "_class")
    result["adequacy"] = {
        "status": "insufficient" if warnings else "requires_research_review",
        "warnings": warnings,
    }
    methods = [
        ("policy", None, None),
        ("sequence_logistic", "sequence", "logistic"),
        ("semantic_logistic", "semantic", "logistic"),
        ("semantic_forest", "semantic", "forest"),
    ]
    if ablate:
        methods += [
            (f"ablated_{algo}", "semantic_ablation", algo) for algo in ("logistic", "forest")
        ]
    for name, representation, algorithm in methods:
        model = None
        if algorithm:
            model = Classifier(representation, algorithm, seed).fit(
                [r.prefix for r in training], [r.target for r in training]
            )
        predictions, latencies = {}, {}
        # Freeze the threshold before scoring a single test proposal.
        val, latencies["validation"] = _predict(rows["validation"], model)
        predictions["validation"] = val
        calibration = (
            calibrate([r.target for r in rows["validation"]], [p.score for p in val])
            if model
            else {"threshold": None, "method": "fixed_rules_review_flags", "curve": []}
        )
        for partition in ("train", "test"):
            predictions[partition], latencies[partition] = _predict(rows[partition], model)
        result["methods"][name] = {
            "family": "ml_only" if model else "policy_only",
            "representation": representation,
            "algorithm": algorithm,
            "model_parameters": model.model.get_params() if model else {},
            "columns": 1 + len(model.vectorizer.columns) if model else None,
            "calibration": calibration,
            "metrics": {
                k: _summarize(partitions[k], rows[k], predictions[k], calibration["threshold"])
                for k in partitions
            },
        }
        elapsed = sorted(latencies["test"])
        result["latency_ns"][name] = {
            "partition": "test",
            "count": len(elapsed),
            "median": median(elapsed) if elapsed else None,
            "p95": elapsed[ceil(0.95 * len(elapsed)) - 1] if elapsed else None,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablate", action="store_true")
    args = parser.parse_args(argv)
    try:
        corpus = Corpus.model_validate_json(args.corpus.read_bytes())
        report = evaluate(
            corpus, json.loads(args.manifest.read_bytes()), seed=args.seed, ablate=args.ablate
        )
        write_new(args.out, report)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.exit(1, f"Baseline evaluation failed: {type(exc).__name__}: {exc}\n")
    print(f"Report written to {args.out}")
    print("Adequacy:", report["adequacy"]["status"], ", ".join(report["adequacy"]["warnings"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
