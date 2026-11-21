"""Load a data-only checkpoint and score a single sanitized Phase 3 prefix."""

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, model_validator

from runtime_security.assets import resource_path
from runtime_security.contracts import Contract
from runtime_security.datasets.schema import Digest, fingerprint, sha
from runtime_security.detectors.bilstm import AttentionBiLSTM, Config, Preprocessing
from runtime_security.features import extract_run
from runtime_security.features.metadata import Policy
from runtime_security.telemetry.events import TrajectoryEvent


class Checkpoint(Contract):
    schema_version: Literal["phase7-v1"]
    framework: str
    config: Config
    prep: Preprocessing
    threshold: float = Field(ge=0, le=1)
    calibration: dict
    training: dict
    dataset: Digest
    manifest: Digest
    training_membership: Digest
    policy_hash: Digest
    config_hash: Digest
    weights: dict[str, list]

    @model_validator(mode="after")
    def compatible(self):
        if self.framework.split(".", 1)[0] != torch.__version__.split(".", 1)[0]:
            raise ValueError("Incompatible torch major version")
        if self.config_hash != fingerprint(self.config.model_dump()):
            raise ValueError("Model configuration digest mismatch")
        if self.threshold != self.calibration.get("threshold"):
            raise ValueError("Inconsistent checkpoint threshold")
        return self


@dataclass
class Loaded:
    checkpoint: Checkpoint
    model: AttentionBiLSTM
    model_version: str


def load_checkpoint(path: Path) -> Loaded:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Checkpoint exceeds JSON size limit")
    data = path.read_bytes()
    ck = Checkpoint.model_validate_json(data)
    model = AttentionBiLSTM(ck.config, ck.prep)
    expected = model.state_dict()
    if set(expected) != set(ck.weights):
        raise ValueError("Incorrect checkpoint parameter names")
    state = {}
    for key, param in expected.items():
        value = torch.tensor(ck.weights[key], dtype=param.dtype)
        if value.shape != param.shape or not torch.isfinite(value).all():
            raise ValueError("Invalid weight tensor")
        state[key] = value
    model.load_state_dict(state, strict=True)
    return Loaded(ck, model.eval(), fingerprint(ck.model_dump()))


def score_prefix(loaded: Loaded, events: list[TrajectoryEvent], policy_path: Path) -> dict:
    return score_prefix_data(loaded, events, policy_path.read_bytes())


@torch.inference_mode()
def score_prefix_data(loaded: Loaded, events: list[TrajectoryEvent], data: bytes) -> dict:
    """Score with the same immutable policy snapshot as the caller's hard rules."""
    if not events or events[-1].kind != "proposal":
        raise ValueError("Prefix must end at a pending proposal")
    if any(e.sequence != i for i, e in enumerate(events, 1)):
        raise ValueError("Prefix must be in contiguous sequence order")
    ck = loaded.checkpoint
    lf = data.replace(b"\r\n", b"\n")
    if ck.policy_hash not in {sha(data), sha(lf), sha(lf.replace(b"\n", b"\r\n"))}:
        raise ValueError("Feature policy digest mismatch")
    rows = extract_run(events, Policy.model_validate(tomllib.loads(data.decode("utf8"))))
    batch = ck.prep.batch([[r.features for r in rows]], ck.config.max_length)
    logits, _ = loaded.model(*batch)
    score = float(logits.sigmoid().item())
    return {
        "score": score,
        "threshold": ck.threshold,
        "unsafe_proposal": score >= ck.threshold,
        "model_version": loaded.model_version,
        "variant": ck.prep.variant,
        "sequence": len(events),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("prefix", type=Path, help="Single run, JSONL through pending proposal")
    parser.add_argument(
        "--policy", type=Path, default=resource_path("configs/policy_metadata.toml")
    )
    args = parser.parse_args(argv)
    try:
        events = [
            TrajectoryEvent.model_validate_json(line)
            for line in args.prefix.read_bytes().splitlines()
        ]
        print(json.dumps(score_prefix(load_checkpoint(args.checkpoint), events, args.policy)))
    except (ValueError, OSError, RuntimeError, TypeError):
        parser.exit(1, "Invalid checkpoint, incompatible policy or invalid trajectory prefix.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
