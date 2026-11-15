"""Shareable schema and canonical, content-addressed artifacts."""

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from runtime_security.contracts import Contract
from runtime_security.features.extractor import Features

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Source = Literal["live", "scripted"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def fingerprint(obj) -> str:
    return sha(canonical(obj))


class Event(Contract):
    sequence: int
    event_id: Digest
    call_id: Digest | None
    features: Features


class Action(Contract):
    proposal: Digest
    call_id: Digest
    result: Digest | None
    violation_attempted: bool | None
    violation_completed: bool | None
    effect_indices: list[int]


class Run(Contract):
    run_id: Digest
    scenario: str | None = None
    version: str | None = None
    model: str | None = None
    max_steps: int | None = None
    groups: list[str] = Field(default_factory=list)
    task: str | None = None
    goal: str | None = None
    injection_present: bool | None = None
    predicate: str | None = None
    outcome: Literal["passed", "failed", "error"] | None = None
    recorded_labels: dict[str, bool] = Field(default_factory=dict)
    terminal: str = "unknown"
    issues: list[str] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    attack_attempted: bool | None = None
    attack_completed: bool | None = None
    benign_completed: bool | None = None
    unsafe_attempted: bool | None = None
    unsafe_completed: bool | None = None
    effects: int | None = None
    unauthorized_effects: list[int] = Field(default_factory=list)


class Corpus(Contract):
    schema_version: Literal["1.0"] = "1.0"
    source: Source
    config_version: str
    tool_configuration: str
    hashes: dict[str, Digest]
    input_issues: list[str]
    runs: list[Run]


# Only fixed schema fields and trusted catalogs can supply shareable free text.
SECRET = re.compile(
    r"SYNTHETIC_SECRET_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{12,}|"
    r"gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]+|"
    r"AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"Bearer\s+[A-Za-z0-9_.-]{8,}",
    re.I,
)


def write_new(path: Path, obj):
    data = canonical(obj)
    if SECRET.search(data.decode()):
        raise ValueError("Redaction scan failed")
    with path.open("xb") as stream:
        stream.write(data)
