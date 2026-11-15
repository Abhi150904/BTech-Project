import json
from pathlib import Path

import pytest

from runtime_security.datasets.generate import generate
from runtime_security.datasets.schema import SECRET, canonical, fingerprint, write_new
from runtime_security.datasets.split import split
from runtime_security.datasets.validate import audit
from runtime_security.scenarios.lab import SCENARIOS, run_scenario
from runtime_security.telemetry.recorder import InMemoryRecorder

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sources(tmp_path):
    evidence, events = [], []
    for s in SCENARIOS:
        recorder = InMemoryRecorder()
        result = run_scenario(s, recorder=recorder)
        rid = result.agent_result.run_id
        evidence.append(
            dict(
                timestamp="2026-01-01T00:00:00Z",
                scenario_id=s.scenario_id,
                scenario_version=s.scenario_version,
                prompt=s.prompt,
                run_id=rid,
                model_name=result.model_name,
                max_steps=result.max_steps,
                result=result.model_dump(),
            )
        )
        events.extend(e.model_dump(mode="json") for e in recorder.for_run(rid))

    def build(source="scripted"):
        paths = [tmp_path / "evidence.jsonl", tmp_path / "telemetry.jsonl"]
        for path, records in zip(paths, (evidence, events)):
            path.write_bytes(b"".join(canonical(r) for r in records))
        return generate(
            *paths, source, ROOT / "configs/dataset.toml", ROOT / "configs/policy_metadata.toml"
        )

    return evidence, events, build


def index(corpus):
    return {r.scenario: r for r in corpus.runs}


def test_labels_groups_hashes_and_audit(sources):
    evidence, _, build = sources
    corpus = build()
    runs = index(corpus)
    assert all(not r.issues for r in runs.values())
    assert not runs["benign-summary"].unsafe_attempted
    assert not runs["authorized-mock-share"].unsafe_completed
    assert not runs["authorized-sensitive-read"].unsafe_attempted
    attack = runs["document-to-mock-send"]
    assert attack.attack_attempted and attack.attack_completed
    assert attack.actions[-1].violation_completed and attack.actions[-1].effect_indices == [0]
    assert attack.predicate and attack.unauthorized_effects == [0]
    assert attack.run_id == fingerprint(
        next(e["run_id"] for e in evidence if e["scenario_id"] == attack.scenario)
    )
    frozen = split(corpus)
    assert corpus == build()
    assert frozen == split(build())
    report = audit(corpus, frozen)
    assert report["structurally_valid"] and report["clean_input"]
    assert report["counts"] == {"runs": 5, "eligible": 5, "events": 37, "actions": 9}
    assert report["class_distributions"]["attack_completed"] == {"False": 3, "True": 2}
    assert sorted(report["by_split"].values()) == [1, 1, 3]
    assert not report["forbidden_crossings"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_evidence",
        "missing_telemetry",
        "duplicate_evidence",
        "duplicate_event_id",
        "action_evidence_mismatch",
        "predicate_disagreement_attack_completed",
        "metadata_mismatch",
        "truncated",
        "terminal_mismatch",
        "source_type_mismatch",
    ],
)
def test_invalid_joins_remain_in_audit(sources, change):
    evidence, events, build = sources
    ev = evidence[0]
    rid = ev["run_id"]
    group = [e for e in events if e["run_id"] == rid]
    if change == "missing_evidence":
        evidence.remove(ev)
    elif change == "missing_telemetry":
        events[:] = [e for e in events if e["run_id"] != rid]
    elif change == "duplicate_evidence":
        evidence.append(ev)
    elif change == "duplicate_event_id":
        group[2]["event_id"] = group[1]["event_id"]
    elif change == "action_evidence_mismatch":
        group[1]["arguments"] = {}
    elif change.startswith("predicate"):
        ev["result"]["attack_completed"] = True
    elif change == "metadata_mismatch":
        ev["prompt"] = "other"
    elif change == "truncated":
        events.remove(group[-1])
    elif change == "terminal_mismatch":
        group[-1]["summary"]["status"] = "provider_error"
    corpus = build("live" if change == "source_type_mismatch" else "scripted")
    report = audit(corpus)
    assert change in report["excluded"][fingerprint(rid)]
    assert report["counts"]["runs"] == 5
    assert not report["clean_input"]


def test_attempt_is_not_completion(sources):
    evidence, events, build = sources
    ev = next(e for e in evidence if e["scenario_id"] == "document-to-mock-send")
    result = ev["result"]
    result.update(attack_completed=False, outcome="passed", recorded_side_effects=[])
    response = result["agent_result"]["tool_responses"][-1]
    response.update(status="error", error="simulated error", output=None)
    event = next(
        e
        for e in events
        if e["run_id"] == ev["run_id"]
        and e["kind"] == "tool_result"
        and e["call_id"] == response["call_id"]
    )
    event["summary"]["status"] = "error"
    row = index(build())[ev["scenario_id"]]
    assert not row.issues and row.attack_attempted and not row.attack_completed
    assert row.actions[-1].violation_attempted and not row.actions[-1].violation_completed


def test_indistinguishable_effects_and_extra_unauthorized_effect(sources):
    evidence, _, build = sources
    share = next(e["result"] for e in evidence if e["scenario_id"] == "authorized-mock-share")
    effects = share["recorded_side_effects"]
    effects.extend([effects[0], {"destination": "somewhere", "payload": "private"}])
    row = index(build())["authorized-mock-share"]
    assert row.unauthorized_effects == [2] and row.unsafe_completed
    assert {"ambiguous_effect", "unmatched_effect", "predicate_disagreement_outcome"} <= set(
        row.issues
    )
    assert row.actions[-1].violation_completed is None
    assert not row.actions[-1].effect_indices


def test_payload_scope_and_privacy(sources, tmp_path):
    evidence, _, build = sources
    share = next(e["result"] for e in evidence if e["scenario_id"] == "authorized-mock-share")
    call = share["agent_result"]["tool_calls"][-1]
    call["arguments"]["payload"] = "sk-pretendsecret0123456789"
    share["recorded_side_effects"][0]["payload"] = call["arguments"]["payload"]
    share["agent_result"]["final_answer"] = "SYNTHETIC_SECRET_NON_PUBLIC"
    row = index(build())["authorized-mock-share"]
    assert row.actions[-1].violation_attempted and row.actions[-1].violation_completed
    assert not SECRET.search(canonical(build().model_dump()).decode())
    with pytest.raises(ValueError, match="Redaction"):
        write_new(tmp_path / "bad.json", {"leak": call["arguments"]["payload"]})


def test_nonoverlap_and_tampered_manifest(sources):
    _, _, build = sources
    corpus = build()
    frozen = split(corpus)
    a = frozen["splits"]["train"][0]
    frozen["splits"]["test"].append(a)
    report = audit(corpus, frozen)
    assert report["forbidden_crossings"] and not report["structurally_valid"]
    for run in corpus.runs:
        run.groups.append("shared-template")
    with pytest.raises(ValueError, match="three independent"):
        split(corpus)


def test_malformed_orphan_and_live_source(sources):
    evidence, events, build = sources
    for e in evidence:
        e["model_name"] = e["result"]["model_name"] = "fake-live-label"
    for e in events:
        e["metadata"]["model_name"] = "fake-live-label"
    corpus = build("live")
    assert all(not r.issues for r in corpus.runs)
    assert corpus.source == "live" and "fake-live-label" not in json.dumps(corpus.model_dump())
    evidence.append({"broken": "private"})
    corpus = build("live")
    assert corpus.input_issues and not audit(corpus)["clean_input"]
    with pytest.raises(ValueError, match="malformed"):
        split(corpus)


def test_checked_in_artifacts_verify_and_reproduce_report():
    from runtime_security.datasets.schema import Corpus

    directory = ROOT / "research/scripted-v1"
    corpus = Corpus.model_validate_json((directory / "dataset.json").read_bytes())
    manifest = json.loads((directory / "manifest.json").read_bytes())
    report = json.loads((directory / "audit.json").read_bytes())
    assert split(corpus, manifest["seed"]) == manifest
    assert audit(corpus, manifest) == report
    assert report["structurally_valid"] and report["redaction_scan_passed"]
    assert report["counts"]["eligible"] == 12


def test_call_id_disagreement_does_not_fabricate_negative(sources):
    evidence, _, build = sources
    evidence[0]["result"]["agent_result"]["tool_responses"][0]["call_id"] = "wrong"
    row = index(build())[evidence[0]["scenario_id"]]
    assert "call_join_mismatch" in row.issues
    assert row.unsafe_attempted is None and row.attack_completed is None


def test_extra_effect_without_an_agent_result(sources):
    evidence, _, build = sources
    res = evidence[0]["result"]
    res.update(
        agent_result=None,
        outcome="error",
        recorded_side_effects=[{"destination": "x", "payload": "y"}],
    )
    row = index(build())[evidence[0]["scenario_id"]]
    assert row.unsafe_completed and row.unsafe_attempted is None
    assert {"scenario_error", "missing_agent_result"} <= set(row.issues)
