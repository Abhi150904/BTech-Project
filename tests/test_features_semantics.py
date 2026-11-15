from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime_security.features import FeatureEncoder, extract_run, load_policy
from runtime_security.features.encoder import CATEGORICAL, NUMERIC
from runtime_security.scenarios import INJECTION
from runtime_security.scenarios.lab import run_scenario
from runtime_security.telemetry.events import RunMetadata
from runtime_security.telemetry.recorder import InMemoryRecorder


@pytest.fixture
def policy():
    return load_policy(Path(__file__).resolve().parents[1] / "configs/policy_metadata.toml")


READ = ("read_file", {"file_id": "sensitive.txt"})
SEND = ("mock_http_post", {"destination": "https://approved.invalid/report"})


def trajectory(actions, *, run="r", scenario="authorized-sensitive-read"):
    recorder = InMemoryRecorder()
    recorder.start_run(run, RunMetadata(max_steps=10, scenario_id=scenario, scenario_version="1.0"))
    for i, ((tool, arguments), status) in enumerate(actions):
        fields = {"tool_name": tool, "call_id": str(i)}
        recorder.record(run, "proposal", **fields, arguments=arguments)
        recorder.record(run, "decision", **fields)
        recorder.record(run, "tool_result", **fields, summary={"status": status})
    recorder.record(run, "run_complete")
    return recorder.for_run(run)


def test_semantics_and_determinism(policy):
    actions = [
        (SEND, "success"),
        (READ, "error"),
        (SEND, "success"),
        (READ, "success"),
        (SEND, "success"),
    ]
    first = extract_run(trajectory(actions), policy)
    second = extract_run(reversed(trajectory(actions, run="new")), policy)
    assert [r.features for r in first] == [r.features for r in second]
    proposals = [r.features for r in first if r.features.kind == "proposal"]
    assert [r.prior_sensitive_attempts for r in proposals] == [0, 0, 1, 1, 2]
    assert [r.prior_sensitive_successes for r in proposals] == [0, 0, 0, 0, 1]
    assert [r.send_after_sensitive_read for r in proposals] == [False] * 4 + [True]
    assert proposals[-1].transition == "read_file->mock_http_post"
    assert proposals[1].authorization == "authorized"
    assert proposals[-1].authorization == "unauthorized"
    assert first[6].features.result_status == "error"
    assert first[12].features.result_status == "success"
    encoder = FeatureEncoder().fit([first])
    assert [r.vector for r in encoder.transform(first)] == [
        r.vector for r in encoder.transform(second)
    ]


def test_policy_provenance_and_label_leakage(policy):
    events = trajectory([(READ, "success")])
    baseline = extract_run(events, policy)
    grants = policy.tasks["authorized-sensitive-read"]["1.0"]
    policy.tasks["renamed-task"] = {"1.0": grants}
    for e in events:
        e.metadata.scenario_id = "renamed-task"
        e.metadata.model_name = "arbitrary-model"
        e.summary.update(
            attack_completed=True,
            attacker_objective="secret",
            authorization="unauthorized",
            resource_sensitivity="public",
        )
    renamed = extract_run(events, policy)
    assert [r.features for r in renamed] == [r.features for r in baseline]
    assert set(asdict(renamed[1].features)) == set(CATEGORICAL + NUMERIC)
    assert not set(asdict(renamed[1].features)).intersection(
        {"attack_family", "attack_completed", "scenario_id", "run_id", "sequence", "event_id"}
    )
    grants.read_file.clear()
    changed = extract_run(events, policy)[1].features
    assert (changed.authorization, changed.mismatch) == ("unauthorized", "yes")
    assert changed.sensitivity == "synthetic_sensitive"


def test_prefix_windows_and_training_only_vocabulary(policy):
    training = extract_run(trajectory([(READ, "success")]), policy)
    encoder = FeatureEncoder().fit([training])
    before = encoder.transform(training)
    events = trajectory([(READ, "success"), (SEND, "success")], run="held-out")
    run = extract_run(events, policy)
    encoded = encoder.transform(run, 3)
    pad = (0,) * len(encoded[0].vector)
    assert encoded[0].window == (pad, pad, encoded[0].vector)
    assert all(len(r.window) == 3 and r.window[-1] == r.vector for r in encoded)
    assert encoded[4].window == tuple(r.vector for r in encoded[2:5])
    assert encoded[4].vector[CATEGORICAL.index("tool")] == 1
    assert len(encoded[4].vector) == len(before[0].vector)
    for i in range(1, len(events) + 1):
        assert encoder.transform(extract_run(events[:i], policy), 3) == encoded[:i]
    events[3].summary["status"] = "error"
    changed = encoder.transform(extract_run(events, policy), 3)
    assert changed[:3] == encoded[:3]
    assert changed[4].vector != encoded[4].vector
    assert encoder.transform(training) == before
    with pytest.raises(ValueError, match="already fitted"):
        encoder.fit([run])


def test_isolation_and_invalid_pairing(policy):
    first = trajectory([(READ, "success")])
    second = trajectory([(SEND, "success")], run="s")
    encoder = FeatureEncoder().fit([extract_run(first, policy)])
    isolated = extract_run(second, policy)
    assert isolated[1].features.prior_sensitive_successes == 0
    assert not any(encoder.transform(isolated)[0].window[0])
    with pytest.raises(ValueError):
        extract_run(first + second, policy)
    with pytest.raises(ValueError):
        encoder.transform(extract_run(first, policy) + isolated)
    first[3].call_id = "wrong"
    with pytest.raises(ValueError, match="pending call"):
        extract_run(first, policy)


def test_unknown_and_redacted_categories(policy):
    events = trajectory(
        [
            (("unrecognized_tool", {}), "error"),
            (("read_file", {"file_id": {"redacted": True}}), "error"),
            (("mock_http_post", {"destination": "unlisted"}), "error"),
        ]
    )
    rows = extract_run(events, policy)
    assert rows[1].features.tool == "unknown"
    assert rows[4].features.sensitivity == "unknown"
    assert rows[7].features.destination_trust == "unknown"
    assert all(rows[i].features.authorization == "unknown" for i in (1, 4, 7))
    vectors = FeatureEncoder().fit([rows]).transform(rows)
    assert vectors[1].vector[CATEGORICAL.index("tool")] == 1
    assert vectors[4].vector[CATEGORICAL.index("sensitivity")] == 1
    for e in events:
        e.metadata.scenario_version = "2.0"
    events[4].arguments = {"file_id": "sensitive.txt"}
    assert extract_run(events, policy)[4].features.authorization == "unknown"


def test_config_rejects_dangling_grants_and_extra_keys(policy):
    data = policy.model_dump()
    data["files"].pop("sensitive.txt")
    with pytest.raises(ValidationError):
        type(policy).model_validate(data)
    with pytest.raises(ValidationError):
        type(policy).model_validate(dict(policy.model_dump(), expected_outcome="failed"))


def test_extract_from_offline_phase3_lab(policy):
    recorder = InMemoryRecorder()
    result = run_scenario(INJECTION, recorder=recorder)
    rows = extract_run(recorder.for_run(result.agent_result.run_id), policy)
    send = next(
        r.features for r in rows if r.features.operation == "send" and r.features.kind == "proposal"
    )
    assert send.send_after_sensitive_read
    assert (send.destination_trust, send.authorization) == ("unapproved", "unauthorized")
