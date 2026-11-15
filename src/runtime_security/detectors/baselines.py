"""Transparent count representations with frozen training vocabularies."""

from collections import Counter
from collections.abc import Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from runtime_security.features.encoder import CATEGORICAL, NUMERIC
from runtime_security.features.extractor import Features

Prefix = Sequence[Features]
POLICY_FIELDS = {
    "sensitivity",
    "destination_trust",
    "authorization",
    "mismatch",
    "send_after_sensitive_read",
    "prior_sensitive_attempts",
    "prior_sensitive_successes",
}


def counts(prefix: Prefix, representation: str) -> Counter:
    if not prefix or prefix[-1].kind != "proposal":
        raise ValueError("Expected a pending proposal")
    out = Counter()
    if representation == "sequence":
        tools = [f.tool for f in prefix if f.kind == "proposal"]
        for n in (1, 2, 3):
            for i in range(n, len(tools) + 1):
                out[("ngram" + str(n), "/".join(tools[i - n : i]))] += 1
        out[("pending_tool", tools[-1])] = 1
        return out
    if representation not in ("semantic", "semantic_ablation"):
        raise ValueError("Unknown representation")
    for name in CATEGORICAL:
        if representation == "semantic_ablation" and name in POLICY_FIELDS:
            continue
        for f in prefix:
            out[("count_" + name, getattr(f, name))] += 1
        out[("pending_" + name, getattr(prefix[-1], name))] = 1
    for name in NUMERIC:
        if representation == "semantic" or name not in POLICY_FIELDS:
            out[("numeric", name)] = int(getattr(prefix[-1], name))
    return out


class Vectorizer:
    def __init__(self, representation: str):
        self.representation = representation
        self.columns = None

    def fit(self, prefixes: Sequence[Prefix]):
        if self.columns is not None:
            raise ValueError("Vocabulary already fitted")
        self.columns = {
            key: i
            for i, key in enumerate(
                sorted({key for prefix in prefixes for key in counts(prefix, self.representation)}),
                start=1,
            )
        }
        return self

    def transform(self, prefix: Prefix) -> list[int]:
        if self.columns is None:
            raise ValueError("Fit on training data first")
        vector = [0] * (1 + len(self.columns))
        for key, value in counts(prefix, self.representation).items():
            vector[self.columns.get(key, 0)] += value
        return vector


class Classifier:
    def __init__(self, representation: str, algorithm: str, seed: int):
        self.vectorizer = Vectorizer(representation)
        self.scaler = StandardScaler() if algorithm == "logistic" else None
        if algorithm == "logistic":
            self.model = LogisticRegression(C=1.0, solver="liblinear", random_state=seed)
        elif algorithm == "forest":
            self.model = RandomForestClassifier(n_estimators=64, random_state=seed, n_jobs=1)
        else:
            raise ValueError("Unknown classifier")

    def fit(self, prefixes: Sequence[Prefix], labels: Sequence[bool]):
        if set(labels) != {True, False}:
            raise ValueError("Training requires known positive and negative pending-action labels")
        self.vectorizer.fit(prefixes)
        matrix = [self.vectorizer.transform(p) for p in prefixes]
        if self.scaler is not None:
            matrix = self.scaler.fit_transform(matrix)
        self.model.fit(matrix, labels)
        return self

    def score(self, prefix: Prefix) -> float:
        matrix = [self.vectorizer.transform(prefix)]
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        return float(self.model.predict_proba(matrix)[0, 1])
