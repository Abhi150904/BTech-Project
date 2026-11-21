import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import pytest

from runtime_security.datasets.schema import Action, Corpus, Event, Run, canonical, fingerprint, sha
from runtime_security.datasets.split import split
from runtime_security.evaluation.inputs import Prediction, baseline_predictions, neural_predictions
from runtime_security.evaluation.report import Config, comparison, evaluate, main, matrix, summarize
from runtime_security.features.extractor import Features
from runtime_security.telemetry.events import RunMetadata, TrajectoryEvent

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/scripted-v1"
NEURAL = ROOT / "research/phase7-scripted-v1/report.json"


@pytest.fixture
def frozen():
    return (
        Corpus.model_validate_json((DATA / "dataset.json").read_bytes()),
        json.loads((DATA / "manifest.json").read_bytes()),
        json.loads((DATA / "baselines.json").read_bytes()),
        json.loads(NEURAL.read_bytes()),
    )


def test_denominators_unknowns_and_empty_classes():
    result = matrix([True, False, True, False, None], [True, True, False, False, True])
    assert [result[k] for k in ("tp", "fp", "fn", "tn", "unknown")] == [1, 1, 1, 1, 1]
    assert result["rates"]["detection_rate_tpr"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert matrix([], [])["rates"]["detection_rate_tpr"]["value"] is None
    assert matrix([False], [False])["recall"] is None
    with pytest.raises(ValueError):
        matrix([True], [])


def test_real_artifacts_recompute_and_expose_coverage(frozen):
    corpus, manifest, baseline, neural = frozen
    result = evaluate(corpus, manifest, baseline=baseline, neural=neural)
    assert result["independent_groups"] == 3
    assert result["adequacy"]["train"]["unsafe_actions"] == 9
    for partition in ("validation", "test"):
        assert "missing_unsafe_action_class" in result["adequacy"][partition]["warnings"]
    phase7 = result["methods"]["phase7/enriched"]
    assert phase7["splits"]["train"]["coverage"]["predicted_actions"] == 0
    assert phase7["splits"]["test"]["coverage"]["missing_predictions"] == 0
    assert phase7["splits"]["test"]["pending_action"]["recall"] is None
    for key in ("tp", "fp", "fn", "tn"):
        assert (
            phase7["splits"]["test"]["pending_action"][key]
            == neural["variants"]["enriched"]["test"][key]
        )
    assert result["experiments"]["E2_coherent_malicious"]["status"] == "unavailable"
    assert result["uncertainty"]["confidence_intervals"] is None
    assert result["observed_source_outcomes"]["train"]["attack_success_rate"]["value"] == 1


def test_common_cohort_never_treats_missing_as_safe(frozen):
    corpus, _, _, _ = frozen
    run = next(r for r in corpus.runs if len(r.actions) > 1)
    left = {a.proposal: Prediction(True) for a in run.actions}
    right = {run.actions[0].proposal: Prediction(False)}
    result = comparison([run], left, right)
    assert result["common_count"] == 1
    assert result["left"]["n"] == result["right"]["n"] == 1
    assert result["left_only"] == len(run.actions) - 1
    summary = summarize([run], right)
    assert summary["coverage"]["fully_covered_runs"] == 0
    assert summary["run_unsafe_attempted"]["n"] == 0


def test_unknown_attempt_and_completed_are_independent(frozen):
    corpus, _, _, _ = frozen
    run = corpus.runs[0]
    for a in run.actions:
        a.violation_attempted = None
        a.violation_completed = False
    predictions = {a.proposal: Prediction(True) for a in run.actions}
    result = summarize([run], predictions)
    assert result["pending_action"]["unknown"] == len(run.actions)
    assert result["pending_action"]["tp"] == 0
    assert result["containment_opportunity_proxy"]["denominator"] == 0
    assert result["containment_opportunity_proxy"]["value"] is None


@pytest.mark.parametrize("mutation", ["dataset", "source", "manifest", "membership"])
def test_prediction_provenance_mismatch_rejected(frozen, mutation):
    corpus, manifest, baseline, neural = frozen
    baseline[mutation] = "wrong"
    with pytest.raises(ValueError, match="match"):
        evaluate(corpus, manifest, baseline=baseline)
    neural[mutation] = "wrong"
    with pytest.raises(ValueError, match="match"):
        evaluate(corpus, manifest, neural=neural)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_scores_rejected(frozen, bad):
    corpus, manifest, baseline, neural = frozen
    neural["variants"]["enriched"]["proposals"][0]["score"] = bad
    with pytest.raises(ValueError, match="probability"):
        neural_predictions(neural, corpus, manifest)
    baseline["methods"]["semantic_logistic"]["metrics"]["test"]["proposals"][0]["score"] = bad
    with pytest.raises(ValueError, match="probability"):
        baseline_predictions(baseline, corpus, manifest)


@pytest.mark.parametrize("operation", ["duplicate", "missing", "wrong_split", "wrong_flag"])
def test_malformed_prediction_coverage_rejected(frozen, operation):
    corpus, manifest, baseline, _ = frozen
    proposals = baseline["methods"]["policy"]["metrics"]["test"]["proposals"]
    if operation == "duplicate":
        proposals.append(proposals[0])
    elif operation == "missing":
        proposals.pop()
    elif operation == "wrong_split":
        proposals[0]["proposal_id"] = corpus.runs[0].actions[0].proposal
    else:
        proposals[0]["flagged"] = not proposals[0]["flagged"]
    with pytest.raises(ValueError):
        evaluate(corpus, manifest, baseline=baseline)


def test_neural_baseline_reference_must_match(frozen):
    corpus, manifest, baseline, neural = frozen
    neural["phase6_report_hash"] = "a" * 64
    with pytest.raises(ValueError, match="different baseline"):
        evaluate(corpus, manifest, baseline=baseline, neural=neural)


def test_exclusions_preserved_and_not_scored(frozen):
    corpus, _, _, _ = frozen
    run = corpus.runs[0]
    run.issues = ["truncated", "missing_result"]
    result = evaluate(corpus, split(corpus))
    assert result["exclusions"][run.run_id] == run.issues
    assert result["excluded_action_labels"] == len(run.actions)
    assert not any(
        p["run_id"] == run.run_id
        for p in result["methods"]["observation/no_intervention"]["proposals"]
    )


def test_threshold_sweep_never_changes_frozen_predictions(frozen):
    corpus, manifest, _, neural = frozen
    first = evaluate(corpus, manifest, neural=neural, config=Config(thresholds=[0.0]))
    second = evaluate(corpus, manifest, neural=neural, config=Config(thresholds=[1.0]))
    assert (
        first["methods"]["phase7/enriched"]["proposals"]
        == second["methods"]["phase7/enriched"]["proposals"]
    )
    with pytest.raises(ValueError):
        evaluate(corpus, manifest, config=Config(thresholds=[float("nan")]))


@pytest.fixture
def gateway_data(tmp_path):
    events, runs = [], []
    for i, (disposition, violation) in enumerate(
        (("block", True), ("review", False), ("allow", False))
    ):
        raw_run, raw_call = f"run-{i}", f"call-{i}"
        metadata = RunMetadata(
            max_steps=3, scenario_id=f"scenario-{i}", model_name="deterministic-scripted"
        )
        timeline = []
        for sequence, kind in enumerate(
            ("run_start", "proposal", "decision", "tool_result", "run_complete"), 1
        ):
            fields = (
                {"call_id": raw_call, "tool_name": "read_file"}
                if kind in {"proposal", "decision", "tool_result"}
                else {}
            )
            summary = {}
            if kind == "decision":
                summary = {
                    "mode": "containment-v1",
                    "decision": disposition,
                    "detector": "alert",
                    "model_score": 0.9,
                    "model_threshold": 0.5,
                    "policy_decision": "allow" if disposition == "allow" else disposition,
                    "approval": "approved" if disposition == "allow" else "not_requested",
                }
            elif kind == "tool_result":
                summary = {"status": "success" if disposition == "allow" else "error"}
            timeline.append(
                TrajectoryEvent(
                    event_id=f"{raw_run}-{sequence}",
                    run_id=raw_run,
                    timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    sequence=sequence,
                    kind=kind,
                    metadata=metadata,
                    summary=summary,
                    **fields,
                )
            )
        events.extend(timeline)
        rows = [
            Event(
                sequence=e.sequence,
                event_id=fingerprint(e.event_id),
                call_id=fingerprint(e.call_id) if e.call_id else None,
                features=Features(kind=e.kind, tool="read_file"),
            )
            for e in timeline
        ]
        runs.append(
            Run(
                run_id=fingerprint(raw_run),
                scenario=f"scenario-{i}",
                groups=[f"family/{i}"],
                model="deterministic-scripted",
                max_steps=3,
                terminal="completed",
                events=rows,
                actions=[
                    Action(
                        proposal=rows[1].event_id,
                        call_id=fingerprint(raw_call),
                        result=rows[3].event_id,
                        violation_attempted=violation,
                        violation_completed=False,
                        effect_indices=[],
                    )
                ],
                injection_present=violation,
                attack_attempted=violation,
                attack_completed=False,
                unsafe_attempted=violation,
                unsafe_completed=False,
                benign_completed=disposition == "allow",
                recorded_labels={
                    "attack_attempted": violation,
                    "attack_completed": False,
                    "benign_task_completed": disposition == "allow",
                },
                effects=0,
            )
        )
    path = tmp_path / "gateway.jsonl"
    path.write_bytes("".join(e.model_dump_json() + "\n" for e in events).encode())
    corpus = Corpus(
        source="scripted",
        config_version="test-v1",
        tool_configuration="synthetic",
        hashes={"telemetry": sha(path.read_bytes())},
        input_issues=[],
        runs=runs,
    )
    return corpus, split(corpus), path


def test_saved_gateway_final_decisions_and_actual_outcomes(gateway_data):
    corpus, manifest, path = gateway_data
    result = evaluate(corpus, manifest, gateway=path)
    rows = result["methods"]["phase8/gateway"]["proposals"]
    assert {r["prediction"]["disposition"]: r["prediction"]["flagged"] for r in rows} == {
        "allow": False,
        "block": True,
        "review": True,
    }
    assert {r["category"] for r in rows} == {"true_positive", "false_positive", "true_negative"}
    assert all(
        r["prediction"]["flagged"]
        for r in result["methods"]["phase8/recorded_detector"]["proposals"]
    )
    assert sum(r["completed_violation"] for r in rows) == 0
    assert sum(r["attempted_violation"] for r in rows) == 1
    assert all(
        v["attack_success_rate"]["numerator"] == 0
        for v in result["observed_source_outcomes"].values()
    )


def test_gateway_hash_and_event_join_rejected(gateway_data):
    corpus, manifest, path = gateway_data
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="hash"):
        evaluate(corpus, manifest, gateway=path)
    corpus.hashes["telemetry"] = sha(path.read_bytes().rstrip() + b"\n")
    path.write_bytes(path.read_bytes().rstrip() + b"\n")
    corpus.runs[0].events[0].event_id = "0" * 64
    with pytest.raises(ValueError, match="event"):
        evaluate(corpus, split(corpus), gateway=path)


def test_cli_deterministic_offline_and_non_destructive(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Network access in offline evaluation")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    common = [
        str(DATA / "dataset.json"),
        str(DATA / "manifest.json"),
        "--baselines",
        str(DATA / "baselines.json"),
        "--neural",
        str(NEURAL),
        "--config",
        str(ROOT / "configs/experiments/phase9.toml"),
    ]
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    assert main([*common, "--out", str(first)]) == 0
    assert main([*common, "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    report = json.loads(first.read_bytes())
    assert first.read_bytes() == canonical(report)
    assert report["input_hashes"]["files"]["dataset"] == sha((DATA / "dataset.json").read_bytes())
    with pytest.raises(SystemExit):
        main([*common, "--out", str(first)])
    assert first.read_bytes() == second.read_bytes()


def test_cli_does_not_echo_malformed_sensitive_input(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"private-value": "DO_NOT_ECHO_THIS"}')
    with pytest.raises(SystemExit):
        main([str(path), str(DATA / "manifest.json"), "--out", str(tmp_path / "out.json")])
    output = capsys.readouterr()
    assert "DO_NOT_ECHO_THIS" not in output.out + output.err
    assert not (tmp_path / "out.json").exists()


def test_gateway_failure_is_not_a_negative_detector_prediction(gateway_data):
    corpus, _, path = gateway_data
    events = [json.loads(line) for line in path.read_text().splitlines()]
    event = next(e for e in events if e["kind"] == "decision")
    event["summary"] = {
        "mode": "containment-v1",
        "decision": "block",
        "policy_decision": "unavailable",
        "detector": "detector_failure",
        "approval": "not_requested",
        "reason": "PRIVATE_SENTINEL_DO_NOT_COPY",
    }
    path.write_bytes(b"".join(canonical(e) for e in events))
    corpus.hashes["telemetry"] = sha(path.read_bytes())
    report = evaluate(corpus, split(corpus), gateway=path)
    assert b"PRIVATE_SENTINEL_DO_NOT_COPY" not in canonical(report)
    detector = report["methods"]["phase8/recorded_detector"]
    assert sum(p["prediction"] is None for p in detector["proposals"]) == 1
    assert any(d["detector_status"] == "detector_failure" for d in detector["diagnostics"])
    assert all(
        p["prediction"] is not None for p in report["methods"]["phase8/gateway"]["proposals"]
    )


def test_target_mismatch_and_nonfinite_saved_latency_rejected(frozen):
    corpus, manifest, baseline, _ = frozen
    target = baseline["target_definitions"]["pending"]
    baseline["target_definitions"]["pending"] = "completed_effect"
    with pytest.raises(ValueError, match="target"):
        evaluate(corpus, manifest, baseline=baseline)
    baseline["target_definitions"]["pending"] = target
    baseline["latency_ns"]["policy"]["median"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluate(corpus, manifest, baseline=baseline)
