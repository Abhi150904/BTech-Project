"""Offline neural tests. The core-only installation can skip this optional suite."""

# ruff: noqa: E402
import copy
import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime_security.datasets.schema import SECRET, Corpus, canonical, fingerprint
from runtime_security.detectors.bilstm import AttentionBiLSTM, Config, Preprocessing, visible
from runtime_security.detectors.inference import load_checkpoint, score_prefix
from runtime_security.detectors.train import Sample, fit, prepare, samples, train
from runtime_security.features.encoder import CATEGORICAL, NUMERIC
from runtime_security.features.extractor import Features
from runtime_security.telemetry.events import RunMetadata
from runtime_security.telemetry.recorder import InMemoryRecorder

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "research/scripted-v1"
PUBLISHED = ROOT / "research/phase7-scripted-v1"
POLICY = ROOT / "configs/policy_metadata.toml"
READ = Features(kind="proposal", tool="read_file", authorization="authorized")


@pytest.fixture
def artifacts():
    return (
        Corpus.model_validate_json((FROZEN / "dataset.json").read_bytes()),
        json.loads((FROZEN / "manifest.json").read_bytes()),
        json.loads((FROZEN / "baselines.json").read_bytes()),
    )


@pytest.fixture
def config():
    return Config(**tomllib.loads((ROOT / "configs/model.toml").read_text())).model_copy(
        update={"epochs": 3, "patience": 1}
    )


def test_mask_and_padding_invariance(config):
    rows = [(READ,), (READ, READ, READ)]
    prep = Preprocessing.fit(rows, "enriched", 3)
    model = AttentionBiLSTM(config, prep).eval()
    cats, nums, lengths = prep.batch(rows, 3)
    logits, weights = model(cats, nums, lengths)
    assert torch.isfinite(logits).all()
    assert weights[0].tolist() == [1, 0, 0]
    torch.testing.assert_close(weights.sum(1), torch.ones(2))
    cats[0, 1:] = 123456
    nums[0, 1:] = float("nan")
    second, _ = model(cats, nums, lengths)
    torch.testing.assert_close(logits, second)
    one, _ = model(*prep.batch([rows[0]], 3))
    torch.testing.assert_close(one, second[:1])
    with pytest.raises(ValueError, match="length"):
        model(cats, nums, torch.tensor([0, 3]))


@pytest.mark.parametrize("variant", ["tool_only", "enriched"])
def test_causal_samples_and_ablation(variant, artifacts):
    corpus, _, _ = artifacts
    run = next(r for r in corpus.runs if len(r.actions) > 1)
    before = list(samples(run))
    for sample in before:
        pos = next(i for i, e in enumerate(run.events) if e.event_id == sample.proposal)
        assert sample.prefix == tuple(e.features for e in run.events[: pos + 1])
        assert sample.prefix[-1].kind == "proposal"
    first = before[0]
    run.actions[0].violation_attempted = None
    run.model = run.scenario = run.run_id = "arbitrary-private-text"
    run.outcome = "error"
    result = next(e for e in run.events if e.features.kind == "tool_result")
    result.features = replace(result.features, result_status="error")
    assert next(samples(run)).prefix == first.prefix
    assert next(samples(run)).target is None
    prep = Preprocessing.fit([s.prefix for s in before], variant, 32)
    assert prep.fields == (("tool",) if variant == "tool_only" else CATEGORICAL)
    assert len(prep.means) == (0 if variant == "tool_only" else len(NUMERIC))
    truncated = visible(before[-1].prefix, variant, 2)
    assert truncated[-1] == before[-1].prefix[-1]
    assert len(truncated) <= 2
    if variant == "tool_only":
        assert all(f.kind == "proposal" for f in truncated)
        changed = [
            replace(f, authorization="unknown", prior_sensitive_attempts=99)
            for f in before[-1].prefix
        ]
        for a, b in zip(prep.batch([changed], 2), prep.batch([before[-1].prefix], 2), strict=True):
            torch.testing.assert_close(a, b)
    for invalid in [[], [result.features]]:
        with pytest.raises(ValueError, match="ending"):
            prep.batch([invalid], 32)


def test_training_vocab_and_normalization():
    rows = [(READ,), (replace(READ, prior_sensitive_attempts=2),)]
    prep = Preprocessing.fit(rows, "enriched", 5)
    assert prep.means == [1, 0, 0] and prep.scales == [1, 1, 1]
    for tool in ["unknown", "search_files", "private payload"]:
        cats, nums, _ = prep.batch([(replace(READ, tool=tool, prior_sensitive_attempts=5),)], 5)
        assert cats[0, 0, 0] == 1
        assert nums[0, 0].tolist() == [4, 0, 0]
    assert prep.vocab["tool"] == ["read_file"]
    extra = Preprocessing.fit([(replace(READ, tool="private payload"),)], "enriched", 1)
    assert "private payload" not in extra.model_dump_json()


def test_fit_supervised_labels_and_validation(config):
    good = Sample("a", [READ], False)
    bad = Sample("b", [replace(READ, tool="mock_http_post")], True)
    unknown = Sample("c", [replace(READ, tool="search_files")], None)
    for invalid in [[], [unknown], [good, unknown], [bad]]:
        with pytest.raises(ValueError, match="positives and negatives"):
            fit(invalid, [good], config, "tool_only")
    with pytest.raises(ValueError, match="known validation"):
        fit([good, bad], [unknown], config, "enriched")
    model, prep, info, cal = fit([good, bad, unknown], [good, unknown], config, "tool_only")
    reference, _, _, _ = fit([good, bad], [good], config, "tool_only")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key])
    assert "search_files" not in prep.vocab["tool"]
    assert info["unknown_excluded"] == 1
    losses = info["validation_bce"]
    assert info["best_epoch"] == 1 + losses.index(min(losses))
    assert info["stopping_reason"] in {"epoch_budget", "validation_patience"}
    assert cal["threshold"] == 0.5 and cal["curve"][0]["unknown"] == 1


def test_frozen_integrity(artifacts):
    corpus, manifest, baseline = artifacts
    sets, metrics = prepare(*artifacts)
    assert set(metrics) == {"sequence_logistic", "semantic_logistic", "semantic_forest"}
    assert len(sets["test"]) == 2
    swapped = copy.deepcopy(manifest)
    swapped["splits"]["test"].reverse()
    for key, val in [
        ("manifest", "0" * 64),
        ("dataset", "0" * 64),
        ("membership", swapped["splits"]),
    ]:
        with pytest.raises(ValueError, match="different artifacts"):
            prepare(corpus, manifest, {**baseline, key: val})
    with pytest.raises(ValueError, match="manifest"):
        prepare(corpus, swapped, baseline)
    baseline["methods"]["semantic_forest"]["metrics"]["test"]["proposals"][0]["proposal_id"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="proposal membership"):
        prepare(corpus, manifest, baseline)


def test_test_labels_do_not_select_model(artifacts, config, monkeypatch):
    sets, comparison = prepare(*artifacts)
    # Vary held-out truth after verifying the frozen input, keeping all prefixes fixed.
    monkeypatch.setattr(
        "runtime_security.detectors.train.prepare", lambda *args: (sets, comparison)
    )
    first, report = train(*artifacts, config)
    sets["test"] = [replace(r, target=True) for r in sets["test"]]
    second, changed = train(*artifacts, config)
    for variant, ck in first.items():
        assert ck == second[variant]
        before, after = report["variants"][variant], changed["variants"][variant]
        assert before["proposals"] == after["proposals"]
        assert before["test"]["positive"] == 0 and after["test"]["positive"] == 2
    assert report["adequacy"] == "insufficient" and "validation_missing_class" in report["warnings"]


def sanitized_prefix():
    recorder = InMemoryRecorder()
    recorder.start_run(
        "one",
        RunMetadata(max_steps=5, scenario_id="authorized-sensitive-read", scenario_version="1.0"),
    )
    recorder.record(
        "one",
        "proposal",
        tool_name="read_file",
        call_id="call",
        arguments={"file_id": "sensitive.txt"},
    )
    return recorder.for_run("one")


@pytest.mark.parametrize("variant", ["enriched", "tool_only"])
def test_export_and_single_prefix_match_frozen_batch(variant, artifacts, tmp_path):
    loaded = load_checkpoint(PUBLISHED / (variant + ".json"))
    report = json.loads((PUBLISHED / "report.json").read_bytes())
    sets, _ = prepare(*artifacts)
    ck = loaded.checkpoint
    assert loaded.model_version == report["variants"][variant]["checkpoint_hash"]
    assert ck.training_membership == fingerprint(report["membership"]["train"])
    assert ck.manifest == report["manifest"] and ck.dataset == report["dataset"]
    score = score_prefix(loaded, sanitized_prefix(), POLICY)
    logits, _ = loaded.model(*ck.prep.batch([s.prefix for s in sets["test"]], ck.config.max_length))
    expected = logits.sigmoid().tolist()
    assert expected == pytest.approx([p["score"] for p in report["variants"][variant]["proposals"]])
    assert score["score"] == pytest.approx(expected[0])
    assert score["unsafe_proposal"] == (score["score"] >= ck.threshold)
    text = canonical([report, ck.model_dump()]).decode()
    assert not SECRET.search(text)
    for forbidden in ['"arguments"', '"payload"', '"prompt"', "sensitive.txt", "https://"]:
        assert forbidden not in text
    damaged = ck.model_dump()
    damaged["weights"]["head.1.weight"] = []
    path = tmp_path / "invalid.json"
    path.write_bytes(canonical(damaged))
    with pytest.raises(ValueError, match="weight tensor"):
        load_checkpoint(path)


def test_inference_rejects_invalid_prefix_and_policy(tmp_path):
    loaded = load_checkpoint(PUBLISHED / "tool_only.json")
    start, proposal = sanitized_prefix()
    for events in [
        [],
        [start],
        [proposal, start],
        [start, proposal.model_copy(update={"sequence": 3})],
        [start, proposal.model_copy(update={"run_id": "other"})],
        [start, proposal.model_copy(update={"event_id": start.event_id})],
    ]:
        with pytest.raises(ValueError):
            score_prefix(loaded, events, POLICY)
    path = tmp_path / "policy.toml"
    path.write_bytes(POLICY.read_bytes() + b"# changed")
    with pytest.raises(ValueError, match="policy digest"):
        score_prefix(loaded, [start, proposal], path)
