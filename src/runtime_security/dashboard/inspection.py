"""Local file loading and presentation helpers; no evaluator or runtime imports."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from runtime_security.dashboard.schema import Fraction, Report

MAX_BYTES = 32 * 1024 * 1024
_PRIVATE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]+|"
    r"-----BEGIN [^-]*PRIVATE KEY-----|"
    r"[A-Za-z]:[\\/][^\s\"<>]*|\\\\[^\s\"<>]+|"
    r"(?<![\w:/<])/(?!/)[^\s\"<>]+)"
)


class ReportError(ValueError):
    """Safe error text suitable for the UI; never includes report values or paths."""


def display_text(value):
    """Native widgets display literal text, never HTML, Tcl scripts, or links."""
    text = str(value) if value is not None else "N/A"
    text = "".join(c for c in text if c in "\n\t" or ord(c) >= 32)
    return _PRIVATE.sub("[redacted]", text)


def percentage(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def fraction_text(value: Fraction):
    return f"{percentage(value.value)} ({value.numerator}/{value.denominator})"


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReportError("Malformed JSON: duplicate object fields.")
        result[key] = value
    return result


def _constant(_):
    raise ReportError("Malformed JSON: nonfinite numbers are not supported.")


@dataclass(frozen=True)
class Inspection:
    report: Report
    fingerprint: str

    def warnings(self, split):
        adequacy = self.report.adequacy[split]
        warnings = []
        if self.report.source == "scripted":
            warnings.append("SCRIPTED FIXTURE — these results do not establish generalization.")
        if adequacy.unsafe_actions == 0:
            warnings.append("No unsafe actions in this split: detection rate is undefined.")
        if adequacy.safe_actions == 0:
            warnings.append("No safe actions in this split: false-intervention rate is undefined.")
        if adequacy.independent_groups < self.report.config.minimum_independent_groups:
            warnings.append(
                f"Only {adequacy.independent_groups} independent group(s) in this split."
            )
        if self.report.exclusions or self.report.input_issues:
            warnings.append("Excluded runs or input issues exist; inspect Runs and Evidence.")
        return warnings

    def totals(self):
        return {
            "runs": len(self.report.runs),
            "eligible_proposals": sum(
                a.safe_actions + a.unsafe_actions + a.unknown_actions
                for a in self.report.adequacy.values()
            ),
            "excluded_action_labels": self.report.excluded_action_labels,
            "events": None,
            "independent_groups": self.report.independent_groups,
        }

    def proposals(self, method, split, query=""):
        runs = {r.run_id: r for r in self.report.runs}
        rows = []
        for p in self.report.methods[method].proposals:
            run = runs[p.run_id]
            searchable = " ".join(
                str(v or "")
                for v in (
                    run.scenario,
                    run.task,
                    run.model,
                    run.version,
                    p.category,
                    p.run_id,
                    p.security_features.tool,
                )
            ).casefold()
            if p.split == split and query.casefold() in searchable:
                rows.append(p)
        return sorted(rows, key=lambda p: (p.run_id, p.sequence))

    def runs(self, split, query="", excluded=False):
        ids = self.report.exclusions if excluded else self.report.membership[split]
        return [
            r
            for r in self.report.runs
            if r.run_id in ids
            and query.casefold()
            in (
                " ".join(
                    str(v or "")
                    for v in (r.run_id, r.scenario, r.task, r.model, r.version, *r.issues)
                ).casefold()
            )
        ]

    def comparison(self, left, right, split):
        for pair in self.report.paired_comparisons:
            if {left, right} == {pair.left_method, pair.right_method}:
                return pair, pair.splits[split]
        return None


def load_report(path) -> Inspection:
    try:
        raw_path = str(path)
        if raw_path.startswith(("\\\\", "//")) or "://" in raw_path:
            raise ReportError(
                "Choose a report on a local disk; URLs and UNC paths are unsupported."
            )
        file = Path(path).resolve()
        if str(file).startswith(("\\\\", "//")):
            raise ReportError("Network paths are unsupported.")
        with file.open("rb") as stream:
            data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ReportError("Report exceeds the 32 MiB local inspection limit.")
        raw = json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
        if not isinstance(raw, dict):
            raise ReportError("A Phase 9 report must be a JSON object.")
        if raw.get("schema_version") != "phase9-v1":
            raise ReportError("Unsupported report version. Expected phase9-v1.")
        report = Report.model_validate(raw)
        return Inspection(report, hashlib.sha256(data).hexdigest())
    except ReportError:
        raise
    except ValidationError as exc:
        # Nested locations may contain attacker-controlled keys; show only known root fields.
        first = exc.errors(include_input=False)[0]["loc"]
        field = first[0] if first and first[0] in Report.model_fields else "report links"
        raise ReportError(
            f"Invalid phase9-v1 report ({field}). Check required fields, types, counts "
            "and split/proposal links. Unknown fields are unsupported."
        ) from None
    except (OSError, ValueError, RecursionError):
        raise ReportError(
            "Cannot read report. Choose a readable local file containing valid JSON."
        ) from None
