"""Summarize evidence quality separately from split and artifact integrity."""

from collections import Counter

from runtime_security.datasets.schema import SECRET, Corpus, canonical, fingerprint
from runtime_security.datasets.split import components, split


def audit(corpus: Corpus, manifest=None):
    rows = corpus.runs
    errors = []
    counts = Counter()
    seen_events = set()
    seen_runs = set()
    duplicates = Counter(key for r in rows for key in set(r.groups))
    exclusions = {}
    for r in rows:
        if r.run_id in seen_runs:
            errors.append("duplicate_run_id")
        seen_runs.add(r.run_id)
        event_map = {e.event_id: e for e in r.events}
        if len(event_map) != len(r.events) or seen_events.intersection(event_map):
            errors.append("duplicate_event_id")
        seen_events.update(event_map)
        if [e.sequence for e in r.events] != list(range(1, len(r.events) + 1)):
            errors.append("event_order")
        if r.issues:
            exclusions[r.run_id] = r.issues
        elif (
            not r.groups
            or r.terminal != "completed"
            or r.scenario is None
            or r.attack_completed is None
        ):
            errors.append("invalid_eligible_record")
        proposals = set()
        if not r.issues:
            derived = {
                "attack_attempted": r.attack_attempted,
                "attack_completed": r.attack_completed,
                "benign_task_completed": r.benign_completed,
            }
            if r.recorded_labels != derived:
                errors.append("unreported_predicate_disagreement")
            if r.unsafe_attempted != any(a.violation_attempted for a in r.actions):
                errors.append("inconsistent_run_attempt")
            if r.unsafe_completed != bool(
                r.unauthorized_effects or any(a.violation_completed for a in r.actions)
            ):
                errors.append("inconsistent_run_completion")
        for a in r.actions:
            p, t = event_map.get(a.proposal), event_map.get(a.result)
            if (
                not p
                or p.features.kind != "proposal"
                or p.call_id != a.call_id
                or not t
                or t.features.kind != "tool_result"
                or t.call_id != a.call_id
                or t.sequence <= p.sequence
                or a.proposal in proposals
                or any(i < 0 or i >= (r.effects or 0) for i in a.effect_indices)
            ):
                errors.append("invalid_action_join")
            proposals.add(a.proposal)
        if not r.issues and proposals != {
            e.event_id for e in r.events if e.features.kind == "proposal"
        }:
            errors.append("missing_action_labels")
        for label in (
            "injection_present",
            "attack_attempted",
            "attack_completed",
            "unsafe_attempted",
            "unsafe_completed",
            "benign_completed",
        ):
            counts[(label, getattr(r, label))] += 1
        for a in r.actions:
            for label in ("violation_attempted", "violation_completed"):
                counts[(label, getattr(a, label))] += 1
    crossings = []
    distribution = {}
    if manifest is not None:
        by_id = {r.run_id: r for r in rows}
        owners = {}
        for name, ids in manifest["splits"].items():
            distribution[name] = len(ids)
            for rid in ids:
                if rid not in by_id:
                    errors.append("unrecognized_split_run")
                    continue
                for key in ["run/" + rid, *by_id[rid].groups]:
                    if owners.setdefault(key, name) != name:
                        crossings.append(key)
        try:
            if split(corpus, manifest["seed"]) != manifest:
                errors.append("manifest_mismatch")
        except ValueError:
            errors.append("invalid_manifest")
    clean = not SECRET.search(canonical([corpus.model_dump(), manifest]).decode())
    if not clean:
        errors.append("redaction_scan_failed")
    eligible = [r for r in rows if not r.issues]
    n_positive = sum(r.unsafe_completed is True for r in eligible)
    eligible_ids = {r.run_id for r in eligible}
    return {
        "dataset_hash": fingerprint(corpus.model_dump()),
        "hashes": corpus.hashes,
        "schema_version": corpus.schema_version,
        "config_version": corpus.config_version,
        "structurally_valid": not errors and not crossings,
        "clean_input": not exclusions and not corpus.input_issues and bool(rows),
        "errors": sorted(set(errors)),
        "input_issues": corpus.input_issues,
        "excluded": exclusions,
        "counts": {
            "runs": len(rows),
            "events": sum(len(r.events) for r in rows),
            "actions": sum(len(r.actions) for r in rows),
            "eligible": len(eligible),
        },
        "by_source": {corpus.source: len(rows)},
        "by_scenario": dict(Counter(r.scenario or "unknown" for r in rows)),
        "by_task": dict(Counter(r.task or "unknown" for r in rows)),
        "by_split": distribution,
        "class_distributions": {
            label: {str(value): n for (k, value), n in counts.items() if k == label}
            for label, _ in counts
        },
        "by_terminal": dict(Counter(r.terminal for r in rows)),
        "issue_counts": dict(Counter(issue for r in rows for issue in r.issues)),
        "shared_group_keys": {key: n for key, n in duplicates.items() if n > 1},
        "by_split_unsafe_completed": {
            name: dict(Counter(str(by_id[i].unsafe_completed) for i in members if i in by_id))
            for name, members in (manifest or {}).get("splits", {}).items()
        },
        "independent_eligible_groups": sum(
            any(i in eligible_ids for i in g) for g in components(rows)
        ),
        "forbidden_crossings": sorted(set(crossings)),
        "redaction_scan_passed": clean,
        "eligible_positive_fraction": n_positive / len(eligible) if eligible else None,
        "balance_warning": n_positive * 2 != len(eligible) or not eligible,
        "research_adequacy": "unestablished; inspect group counts and per-split coverage",
    }
