import json
from dataclasses import replace
from pathlib import Path

import pytest

from runtime_security.datasets.schema import Corpus, canonical
from runtime_security.evaluation.metrics import calibrate, confusion
from runtime_security.features.extractor import Features
from runtime_security.policies.rules import evaluate_rules

ROOT = Path(__file__).resolve().parents[1] / "research/scripted-v1"
READ = Features(
    kind="proposal",
    tool="read_file",
    operation="read",
    resource_class="file",
    sensitivity="synthetic_sensitive",
    authorization="authorized",
)
SEND = Features(
    kind="proposal",
    tool="mock_http_post",
    operation="send",
    resource_class="destination",
    destination_trust="approved",
    authorization="authorized",
)


def test_rule_decisions_at_proposal():
    assert evaluate_rules([READ]).disposition == "allow"
    assert evaluate_rules([replace(READ, sensitivity="unknown")]).disposition == "review"
    assert evaluate_rules([replace(READ, authorization="unauthorized")]).rules == (
        "sensitive_read_out_of_scope",
    )
    assert evaluate_rules([replace(READ, authorization="unknown")]).disposition == "review"
    assert evaluate_rules([SEND]).disposition == "review"
    assert evaluate_rules([SEND], payload_check=("public", "public")).disposition == "allow"
    assert evaluate_rules([READ, SEND], payload_check=("different", "public")).rules == (
        "payload_out_of_scope",
    )
    assert evaluate_rules(
        [SEND, replace(SEND, authorization="unauthorized", destination_trust="unapproved")]
    ).rules == ("unapproved_destination", "send_out_of_scope")
    with pytest.raises(ValueError, match="proposal"):
        evaluate_rules([SEND, replace(SEND, kind="tool_result")])


def test_metrics_unknowns_and_empty_classes():
    m = confusion([True, False, True, False, None], [True, False, False, True, True])
    assert [m[k] for k in ("tp", "fp", "fn", "tn", "unknown")] == [1, 1, 1, 1, 1]
    assert [m[k] for k in ("precision", "recall", "f1", "fpr")] == [0.5] * 4
    empty = confusion([], [])
    assert empty["n"] == 0 and all(empty[k] is None for k in ("precision", "recall", "f1", "fpr"))
    assert confusion([False], [True])["recall"] is None


def test_validation_threshold():
    best = calibrate([True, False, True, None], [0.3, 0.1, 0.9, 0.5])
    assert best["threshold"] == 0.3
    assert calibrate([False], [0.9])["threshold"] == 0.5


@pytest.fixture
def fixture_data():
    pytest.importorskip("sklearn")
    return Corpus.model_validate_json((ROOT / "dataset.json").read_bytes()), json.loads(
        (ROOT / "manifest.json").read_bytes()
    )


@pytest.mark.parametrize("representation", ["sequence", "semantic", "semantic_ablation"])
def test_train_only_columns_and_prefix_isolation(fixture_data, representation):
    from runtime_security.detectors.baselines import Vectorizer
    from runtime_security.evaluation.baselines import examples

    corpus, _ = fixture_data
    run = next(r for r in corpus.runs if len(r.actions) >= 2)
    original = examples(run)[0]
    encoder = Vectorizer(representation).fit([original.prefix])
    before = encoder.transform(original.prefix)
    run.run_id = "a" * 64
    run.scenario = "leakage-test"
    run.outcome = "error"
    run.actions[0].violation_attempted = None
    run.events[-1].features = replace(run.events[-1].features, result_status="changed")
    after = examples(run)[0]
    assert encoder.transform(after.prefix) == before
    assert original.prefix[-1].kind == "proposal"
    changed = (*original.prefix[:-1], replace(original.prefix[-1], tool="never_seen"))
    assert len(encoder.transform(changed)) == len(before)
    assert encoder.transform(changed)[0] > 0
    assert encoder.transform(original.prefix) == before
    if representation == "sequence":
        changed = (
            *original.prefix[:-1],
            replace(original.prefix[-1], sensitivity="changed", authorization="unknown"),
        )
        assert before == encoder.transform(changed)


def test_smoke_and_reproducible_results(fixture_data):
    from runtime_security.evaluation.baselines import evaluate

    corpus, manifest = fixture_data
    a, b = evaluate(corpus, manifest), evaluate(corpus, manifest)
    for report in (a, b):
        assert report["membership"] == manifest["splits"]
        assert "test_missing_action_class" in report["adequacy"]["warnings"]
        timings = report.pop("latency_ns")
        for name, method in report["methods"].items():
            assert timings[name]["count"] == 2
            assert timings[name]["median"] >= 0
            assert method["metrics"]["test"]["pending_violation_attempted"]["unknown"] == 0
            for part in manifest["splits"]:
                expected = {
                    a.proposal
                    for r in corpus.runs
                    if r.run_id in manifest["splits"][part]
                    for a in r.actions
                }
                assert {p["proposal_id"] for p in method["metrics"][part]["proposals"]} == expected
            if name != "policy":
                assert method["calibration"]["method"] == "fixed_fallback_missing_validation_class"
    assert canonical(a) == canonical(b)
    validation = a["methods"]["policy"]["metrics"]["validation"]
    assert validation["policy_dispositions"]["review"] == 2
    assert validation["completed_benign_runs_flagged"] == 2


@pytest.mark.parametrize("tamper", ["source", "dataset", "membership"])
def test_manifest_mismatch_fails(fixture_data, tamper):
    from runtime_security.evaluation.baselines import evaluate

    corpus, manifest = fixture_data
    if tamper == "membership":
        manifest["splits"]["test"].append(manifest["splits"]["train"][0])
    else:
        manifest[tamper] = "other"
    with pytest.raises(ValueError, match="manifest"):
        evaluate(corpus, manifest)


def test_test_labels_cannot_change_threshold_or_predictions(fixture_data):
    from runtime_security.datasets.split import split
    from runtime_security.evaluation.baselines import evaluate

    corpus, manifest = fixture_data
    before = evaluate(corpus, manifest)
    for run in corpus.runs:
        if run.run_id in manifest["splits"]["test"]:
            run.actions[0].violation_attempted = True
            run.unsafe_attempted = True
    # Synthetic counterfactual uses the same membership, with a new valid artifact hash.
    updated = split(corpus, manifest["seed"])
    assert updated["splits"] == manifest["splits"]
    after = evaluate(corpus, updated)
    for name, old in before["methods"].items():
        new = after["methods"][name]
        assert old["calibration"] == new["calibration"]
        assert old["metrics"]["test"]["proposals"] == new["metrics"]["test"]["proposals"]
