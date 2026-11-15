"""Field-local categorical IDs and fixed-width causal windows."""

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from runtime_security.features.extractor import FeatureRecord

CATEGORICAL = (
    "tool",
    "operation",
    "resource_class",
    "sensitivity",
    "destination_trust",
    "authorization",
    "kind",
    "previous_tool",
    "transition",
    "result_status",
    "mismatch",
)
NUMERIC = ("prior_sensitive_attempts", "prior_sensitive_successes", "send_after_sensitive_read")
Vector = tuple[int, ...]


@dataclass(frozen=True)
class EncodedRecord:
    run_id: str
    event_id: str
    sequence: int
    vector: Vector
    window: tuple[Vector, ...]


class FeatureEncoder:
    def __init__(self):
        self._codes: dict[str, dict[str, int]] | None = None

    def fit(self, training_runs: Iterable[Sequence[FeatureRecord]]) -> "FeatureEncoder":
        if self._codes is not None:
            raise ValueError("Encoder already fitted")
        categories = {name: set() for name in CATEGORICAL}
        for run in training_runs:
            for record in run:
                for name in CATEGORICAL:
                    categories[name].add(getattr(record.features, name))
        if not categories["kind"]:
            raise ValueError("Training requires at least one event")
        self._codes = {
            name: {v: i for i, v in enumerate(sorted(values - {"unknown"}), 2)}
            for name, values in categories.items()
        }
        return self

    def transform(self, run: Sequence[FeatureRecord], window_size: int = 4) -> list[EncodedRecord]:
        if self._codes is None:
            raise ValueError("Fit encoder on training runs first")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        pad = (0,) * (len(CATEGORICAL) + len(NUMERIC))
        history = deque([pad] * window_size, maxlen=window_size)
        result = []
        for sequence, row in enumerate(run, 1):
            if row.sequence != sequence or row.run_id != run[0].run_id:
                raise ValueError("Expected one ordered run")
            vector = tuple(
                self._codes[name].get(getattr(row.features, name), 1) for name in CATEGORICAL
            ) + tuple(int(getattr(row.features, name)) for name in NUMERIC)
            history.append(vector)
            result.append(EncodedRecord(row.run_id, row.event_id, sequence, vector, tuple(history)))
        return result
