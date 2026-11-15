"""CPU proposal-prefix encoding, packed BiLSTM and masked attention."""

from collections.abc import Sequence
from math import isfinite
from statistics import fmean, pstdev
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from runtime_security.contracts import Contract
from runtime_security.features.encoder import CATEGORICAL, NUMERIC
from runtime_security.features.extractor import NA, TOOLS, Features

Variant = Literal["tool_only", "enriched"]
Prefix = Sequence[Features]

# Only Phase 4 semantic values can enter an exported vocabulary.
_TOOLS = {*TOOLS, "search_files", "unknown"}
_PREVIOUS = _TOOLS | {"start"}
_DOMAINS = {
    "tool": _TOOLS,
    "previous_tool": _PREVIOUS,
    "transition": {f"{a}->{b}" for a in _PREVIOUS for b in _TOOLS},
    "operation": {"read", "send", "search"},
    "resource_class": {"document", "file", "destination"},
    "sensitivity": {"public", "synthetic_sensitive"},
    "destination_trust": {"approved", "unapproved"},
    "authorization": {"authorized", "unauthorized"},
    "kind": {"run_start", "proposal", "decision", "tool_result", "run_complete"},
    "result_status": {"success", "error"},
    "mismatch": {"yes", "no"},
}


class Config(Contract):
    embedding: int = Field(ge=1, le=128)
    hidden: int = Field(ge=1, le=256)
    layers: int = Field(ge=1, le=4)
    dropout: float = Field(ge=0, lt=1)
    max_length: int = Field(ge=1, le=1024)
    seed: int = Field(ge=0, le=2**32 - 1)
    epochs: int = Field(ge=1)
    patience: int = Field(ge=1)
    learning_rate: float = Field(gt=0, le=1)
    batch_size: int = Field(ge=1)


def visible(prefix: Prefix, variant: Variant, limit: int) -> list[Features]:
    if limit < 1:
        raise ValueError("Sequence limit must be positive")
    if not prefix or prefix[-1].kind != "proposal":
        raise ValueError("Expected a nonempty prefix ending at a pending proposal")
    rows = [f for f in prefix if variant == "enriched" or f.kind == "proposal"]
    return rows[-limit:]


class Preprocessing(Contract):
    variant: Variant
    vocab: dict[str, list[str]]
    means: list[float]
    scales: list[float]

    @property
    def fields(self):
        return CATEGORICAL if self.variant == "enriched" else ("tool",)

    @model_validator(mode="after")
    def compatible(self):
        if set(self.vocab) != set(self.fields):
            raise ValueError("Incompatible categorical fields")
        for name, values in self.vocab.items():
            domain = (_DOMAINS[name] | {NA}) - {"unknown"}
            if values != sorted(set(values)) or not set(values) <= domain:
                raise ValueError("Incompatible vocabulary")
        size = len(NUMERIC) if self.variant == "enriched" else 0
        if len(self.means) != size or len(self.scales) != size:
            raise ValueError("Incompatible numeric fields")
        if not all(isfinite(x) for x in self.means + self.scales) or any(
            x <= 0 for x in self.scales
        ):
            raise ValueError("Invalid normalization statistics")
        return self

    @classmethod
    def fit(cls, prefixes: Sequence[Prefix], variant: Variant, limit: int):
        rows = [f for p in prefixes for f in visible(p, variant, limit)]
        if not rows:
            raise ValueError("Empty training prefixes")
        fields = CATEGORICAL if variant == "enriched" else ("tool",)
        vocab = {
            name: sorted(({getattr(f, name) for f in rows} & (_DOMAINS[name] | {NA})) - {"unknown"})
            for name in fields
        }
        nums = (
            [[float(getattr(f, name)) for f in rows] for name in NUMERIC]
            if variant == "enriched"
            else []
        )
        return cls(
            variant=variant,
            vocab=vocab,
            means=[fmean(col) for col in nums],
            scales=[pstdev(col) or 1.0 for col in nums],
        )

    def batch(self, prefixes: Sequence[Prefix], limit: int):
        if not prefixes:
            raise ValueError("Empty batch")
        rows = [visible(p, self.variant, limit) for p in prefixes]
        lengths = torch.tensor([len(p) for p in rows], dtype=torch.long)
        shape = (len(rows), int(lengths.max()))
        cats = torch.zeros((*shape, len(self.fields)), dtype=torch.long)
        nums = torch.zeros((*shape, len(self.means)), dtype=torch.float32)
        codes = [{v: i for i, v in enumerate(self.vocab[k], 2)} for k in self.fields]
        for i, prefix in enumerate(rows):
            cats[i, : len(prefix)] = torch.tensor(
                [
                    [c.get(getattr(f, k), 1) for c, k in zip(codes, self.fields, strict=True)]
                    for f in prefix
                ]
            )
            if self.means:
                nums[i, : len(prefix)] = torch.tensor(
                    [
                        [
                            (float(getattr(f, k)) - m) / s
                            for k, m, s in zip(NUMERIC, self.means, self.scales, strict=True)
                        ]
                        for f in prefix
                    ]
                )
        return cats, nums, lengths


class AttentionBiLSTM(nn.Module):
    def __init__(self, cfg: Config, prep: Preprocessing):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(len(prep.vocab[k]) + 2, cfg.embedding, padding_idx=0) for k in prep.fields
        )
        self.rnn = nn.LSTM(
            cfg.embedding * len(prep.fields) + len(prep.means),
            cfg.hidden,
            num_layers=cfg.layers,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = nn.Linear(2 * cfg.hidden, 1)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(2 * cfg.hidden, 1))

    def forward(self, cats, nums, lengths):
        if (
            cats.ndim != 3
            or lengths.shape != (cats.shape[0],)
            or not lengths.numel()
            or ((lengths < 1) | (lengths > cats.shape[1])).any()
        ):
            raise ValueError("Invalid sequence lengths")
        mask = torch.arange(cats.shape[1])[None, :] < lengths[:, None]
        cats = cats.masked_fill(~mask.unsqueeze(-1), 0)
        embedded = [emb(cats[:, :, i]) for i, emb in enumerate(self.embeddings)]
        packed = pack_padded_sequence(
            torch.cat([*embedded, nums], dim=-1), lengths, batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.rnn(packed)
        states, _ = pad_packed_sequence(encoded, batch_first=True, total_length=cats.shape[1])
        weights = self.attention(states).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(dim=1)
        pooled = (states * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1), weights
