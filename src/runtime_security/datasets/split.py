"""Allocate connected components of all forbidden shared keys."""

from collections import defaultdict

from runtime_security.datasets.schema import Corpus, fingerprint


def components(runs):
    leaders = {r.run_id: r.run_id for r in runs}

    def root(key):
        while leaders[key] != key:
            leaders[key] = leaders[leaders[key]]
            key = leaders[key]
        return key

    seen = {}
    for run in runs:
        for key in run.groups:
            other = seen.setdefault(key, run.run_id)
            leaders[root(other)] = root(run.run_id)
    result = defaultdict(list)
    for run in runs:
        result[root(run.run_id)].append(run.run_id)
    return sorted(sorted(group) for group in result.values())


def split(corpus: Corpus, seed: int = 42) -> dict:
    if corpus.input_issues:
        raise ValueError("Unassigned malformed input lines; correct sources before freezing")
    if len({r.run_id for r in corpus.runs}) != len(corpus.runs):
        raise ValueError("Duplicate run IDs")
    eligible = {r.run_id for r in corpus.runs if not r.issues}
    # Excluded runs can still connect two otherwise eligible families.
    groups = [g for g in components(corpus.runs) if eligible.intersection(g)]
    if len(groups) < 3:
        raise ValueError(f"Need three independent eligible groups; found {len(groups)}")
    groups.sort(key=lambda g: (-len(eligible.intersection(g)), fingerprint([seed, g])))
    names = ("train", "validation", "test")
    assignments = {name: [] for name in names}
    frozen = []
    total = len(eligible)
    target = dict(zip(names, (0.6 * total, 0.2 * total, 0.2 * total)))
    for index, group in enumerate(groups):
        name = (
            names[index] if index < 3 else max(names, key=lambda n: target[n] - len(assignments[n]))
        )
        members = sorted(eligible.intersection(group))
        assignments[name].extend(members)
        frozen.append({"group": fingerprint(group), "split": name, "runs": members})
    return {
        "schema_version": "1.0",
        "dataset": fingerprint(corpus.model_dump()),
        "source": corpus.source,
        "config_version": corpus.config_version,
        "hashes": corpus.hashes,
        "algorithm": "connected-components-largest-first-v1",
        "seed": seed,
        "target_weights": [0.6, 0.2, 0.2],
        "groups": frozen,
        "splits": assignments,
    }
