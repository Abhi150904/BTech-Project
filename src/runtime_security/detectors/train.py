"""Train matched CPU BiLSTMs offline against an existing frozen Phase 6 comparison."""

import argparse
import copy
import json
import tempfile
import tomllib
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
from torch.nn import functional as F

from runtime_security.assets import resource_path
from runtime_security.datasets.schema import Corpus, fingerprint, write_new
from runtime_security.datasets.validate import audit
from runtime_security.detectors.bilstm import (
    AttentionBiLSTM,
    Config,
    Prefix,
    Preprocessing,
    Variant,
)
from runtime_security.detectors.inference import Checkpoint
from runtime_security.evaluation.metrics import calibrate, confusion
from runtime_security.features.encoder import NUMERIC


class TrainingError(ValueError):
    """Safe diagnostic from data verification or supervised fitting."""


@dataclass(frozen=True)
class Sample:
    proposal: str
    prefix: Prefix
    target: bool | None


def samples(run):
    labels = {a.proposal: a.violation_attempted for a in run.actions}
    history = []
    for index, event in enumerate(run.events, 1):
        if event.sequence != index:
            raise ValueError("Invalid event ordering")
        history.append(event.features)
        if event.features.kind == "proposal":
            yield Sample(event.event_id, tuple(history), labels[event.event_id])


def prepare(corpus, manifest, baseline):
    if not isinstance(manifest, dict) or not audit(corpus, manifest)["structurally_valid"]:
        raise TrainingError("Invalid Phase 5 manifest or corpus")
    if (
        baseline["manifest"] != fingerprint(manifest)
        or baseline["dataset"] != manifest["dataset"]
        or baseline["membership"] != manifest["splits"]
        or baseline["source"] != corpus.source
    ):
        raise TrainingError("Phase 6 comparison uses different artifacts")
    by_id = {r.run_id: r for r in corpus.runs}
    sets = {
        k: [s for rid in ids for s in samples(by_id[rid])] for k, ids in manifest["splits"].items()
    }
    targets = {s.proposal: s.target for s in sets["test"]}
    comparison = {}
    for key in ("sequence_logistic", "semantic_logistic", "semantic_forest"):
        metrics = baseline["methods"][key]["metrics"]["test"]
        proposals = metrics["proposals"]
        if len(proposals) != len(targets) or {p["proposal_id"] for p in proposals} != set(targets):
            raise TrainingError("Baseline proposal membership mismatch")
        checked = confusion(
            [targets[p["proposal_id"]] for p in proposals], [p["flagged"] for p in proposals]
        )
        if checked != metrics["pending_violation_attempted"]:
            raise TrainingError("Baseline pending-action metrics mismatch")
        comparison[key] = checked
    return sets, comparison


@torch.inference_mode()
def _scores(model, prep, cfg, rows):
    model.eval()
    scores = []
    for i in range(0, len(rows), cfg.batch_size):
        batch = prep.batch([r.prefix for r in rows[i : i + cfg.batch_size]], cfg.max_length)
        scores.extend(model(*batch)[0].sigmoid().tolist())
    return scores


def fit(training, validation, cfg: Config, variant: Variant):
    train = [s for s in training if s.target is not None]
    val = [s for s in validation if s.target is not None]
    if {s.target for s in train} != {True, False}:
        raise TrainingError("Training requires known positives and negatives")
    if not val:
        raise TrainingError("Validation loss requires at least one known validation label")
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    prep = Preprocessing.fit([s.prefix for s in train], variant, cfg.max_length)
    model = AttentionBiLSTM(cfg, prep)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    generator = torch.Generator().manual_seed(cfg.seed)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    stop = "epoch_budget"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for indices in torch.randperm(len(train), generator=generator).split(cfg.batch_size):
            rows = [train[i] for i in indices.tolist()]
            batch = prep.batch([r.prefix for r in rows], cfg.max_length)
            labels = torch.tensor([r.target for r in rows], dtype=torch.float32)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(*batch)[0], labels)
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.inference_mode():
            for start in range(0, len(val), cfg.batch_size):
                rows = val[start : start + cfg.batch_size]
                batch = prep.batch([r.prefix for r in rows], cfg.max_length)
                labels = torch.tensor([r.target for r in rows], dtype=torch.float32)
                total += F.binary_cross_entropy_with_logits(
                    model(*batch)[0], labels, reduction="sum"
                ).item()
        loss = total / len(val)
        if not isfinite(loss):
            raise TrainingError("Nonfinite validation loss")
        history.append(loss)
        if loss < best_loss:
            best_loss, best_epoch, best = loss, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            stop = "validation_patience"
            break
    model.load_state_dict(best)
    model.eval()
    calibration = calibrate([s.target for s in validation], _scores(model, prep, cfg, validation))
    info = {
        "positive": sum(s.target is True for s in train),
        "negative": sum(s.target is False for s in train),
        "unknown_excluded": len(training) - len(train),
        "loss": "unweighted BCEWithLogits",
        "sampling": "shuffle without replacement",
        "monitor": "minimum validation BCE",
        "validation_bce": history,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "stopping_reason": stop,
    }
    return model, prep, info, calibration


def train(corpus: Corpus, manifest: dict, baseline: dict, cfg: Config):
    sets, comparison = prepare(corpus, manifest, baseline)
    checkpoints = {}
    warnings = []
    if corpus.source == "scripted":
        warnings.append("scripted_pipeline_fixture")
    if len(manifest["groups"]) < 6:
        warnings.append("few_independent_groups")
    for part in ("validation", "test"):
        if {s.target for s in sets[part] if s.target is not None} != {False, True}:
            warnings.append(part + "_missing_class")
    report = {
        "schema_version": "phase7-v1",
        "dataset": manifest["dataset"],
        "source": corpus.source,
        "manifest": fingerprint(manifest),
        "membership": manifest["splits"],
        "hashes": corpus.hashes,
        "config": cfg.model_dump(),
        "framework": str(torch.__version__),
        "target": "violation_attempted",
        "phase6_report_hash": fingerprint(baseline),
        "phase6_test": comparison,
        "excluded_runs": [r.run_id for r in corpus.runs if r.issues],
        "excluded_action_labels": sum(len(r.actions) for r in corpus.runs if r.issues),
        "unknown_by_partition": {k: sum(s.target is None for s in v) for k, v in sets.items()},
        "adequacy": "insufficient" if warnings else "requires_research_review",
        "warnings": warnings,
        "independent_groups": len(manifest["groups"]),
        "variants": {},
    }
    for variant in ("tool_only", "enriched"):
        model, prep, info, cal = fit(sets["train"], sets["validation"], cfg, variant)
        ck = Checkpoint(
            schema_version="phase7-v1",
            framework=str(torch.__version__),
            config=cfg,
            prep=prep,
            threshold=cal["threshold"],
            calibration=cal,
            training=info,
            dataset=manifest["dataset"],
            manifest=report["manifest"],
            training_membership=fingerprint(manifest["splits"]["train"]),
            policy_hash=corpus.hashes["policy"],
            config_hash=fingerprint(cfg.model_dump()),
            weights={k: v.tolist() for k, v in model.state_dict().items()},
        )
        checkpoints[variant] = ck
        scores, times = [], []
        for sample in sets["test"]:
            start = perf_counter_ns()
            scores.extend(_scores(model, prep, cfg, [sample]))
            times.append(perf_counter_ns() - start)
        times.sort()
        metrics = confusion([r.target for r in sets["test"]], [s >= ck.threshold for s in scores])
        report["variants"][variant] = {
            "training": info,
            "categorical_fields": prep.fields,
            "numeric_fields": list(NUMERIC) if variant == "enriched" else [],
            "calibration": cal,
            "test": metrics,
            "checkpoint_hash": fingerprint(ck.model_dump()),
            "proposals": [
                {"id": r.proposal, "score": s} for r, s in zip(sets["test"], scores, strict=True)
            ],
            "latency_ns": {
                "count": len(times),
                "median": median(times) if times else None,
                "p95": times[ceil(0.95 * len(times)) - 1] if times else None,
                "method": "one CPU prefix; preprocessing plus forward/sigmoid, perf_counter_ns",
            },
        }
    return checkpoints, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "manifest", "baselines"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--config", type=Path, default=resource_path("configs/model.toml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            parser.exit(1, "Choose a new output directory.\n")
        corpus = Corpus.model_validate_json(args.corpus.read_bytes())
        config = Config.model_validate(tomllib.loads(args.config.read_text(encoding="utf8")))
        ck, report = train(
            corpus,
            json.loads(args.manifest.read_bytes()),
            json.loads(args.baselines.read_bytes()),
            config,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out.parent) as staging:
            dest = Path(staging) / "result"
            dest.mkdir()
            for variant, checkpoint in ck.items():
                write_new(dest / (variant + ".json"), checkpoint.model_dump())
            write_new(dest / "report.json", report)
            dest.rename(args.out)
    except TrainingError as exc:
        parser.exit(1, f"Training failed: {exc}.\n")
    except (KeyError, TypeError, OSError, ValueError) as exc:
        parser.exit(
            1, f"Training failed ({type(exc).__name__}). Verify artifacts, labels and paths.\n"
        )
    print(f"Saved checkpoints and report under {args.out}. Adequacy: {report['adequacy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
