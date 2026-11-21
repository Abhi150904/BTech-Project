"""Dashboard compatibility contract for the existing phase9-v1 JSON format.

These checks validate structure and internal consistency; they never evaluate
an agent, recalculate predictions, or claim that supplied labels are correct.
"""

from math import isclose
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Count = Annotated[int, Field(ge=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Probability = Annotated[float, Field(ge=0, le=1)]
Split = Literal["train", "validation", "test"]
MethodName = Annotated[str, Field(pattern=r"^(observation|phase6|phase7|phase8)/[A-Za-z0-9_-]+$")]
SPLITS = {"train", "validation", "test"}


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Fraction(Record):
    numerator: Count
    denominator: Count
    value: Probability | None

    @model_validator(mode="after")
    def consistent(self):
        if self.numerator > self.denominator:
            raise ValueError("Numerator exceeds denominator")
        if self.denominator == 0:
            if self.value is not None:
                raise ValueError("Zero denominator must be undefined")
        elif self.value is None or not isclose(self.value, self.numerator / self.denominator):
            raise ValueError("Rate does not match supplied counts")
        return self


class Rates(Record):
    detection_rate_tpr: Fraction
    false_intervention_rate_fpr: Fraction
    precision: Fraction
    f1: Fraction


class Matrix(Record):
    n: Count
    positive: Count
    negative: Count
    unknown: Count
    tp: Count
    fp: Count
    fn: Count
    tn: Count
    fpr: Probability | None
    precision: Probability | None
    recall: Probability | None
    f1: Probability | None
    rates: Rates

    @model_validator(mode="after")
    def consistent(self):
        if (self.positive, self.negative, self.n) != (
            self.tp + self.fn,
            self.fp + self.tn,
            self.positive + self.negative + self.unknown,
        ):
            raise ValueError("Inconsistent confusion counts")
        for name, n, d, value in (
            ("detection_rate_tpr", self.tp, self.positive, self.recall),
            ("false_intervention_rate_fpr", self.fp, self.negative, self.fpr),
            ("precision", self.tp, self.tp + self.fp, self.precision),
            ("f1", 2 * self.tp, 2 * self.tp + self.fp + self.fn, self.f1),
        ):
            r = getattr(self.rates, name)
            if (r.numerator, r.denominator, r.value) != (n, d, value):
                raise ValueError("Conflicting metric representations")
        return self


class Coverage(Record):
    total_actions: Count
    predicted_actions: Count
    missing_predictions: Count
    unknown_action_labels: Count
    unknown_completion_labels: Count
    total_runs: Count
    fully_covered_runs: Count

    @model_validator(mode="after")
    def consistent(self):
        if self.predicted_actions + self.missing_predictions != self.total_actions:
            raise ValueError("Inconsistent prediction coverage")
        if (
            self.fully_covered_runs > self.total_runs
            or max(self.unknown_action_labels, self.unknown_completion_labels) > self.total_actions
        ):
            raise ValueError("Coverage exceeds totals")
        return self


class Summary(Record):
    coverage: Coverage
    pending_action: Matrix
    run_unsafe_attempted: Matrix
    run_unsafe_completed_proxy: Matrix
    benign_run_intervention_rate: Fraction
    completed_benign_runs_flagged: Count
    covered_completed_benign_runs: Count
    covered_unknown_benign_outcomes: Count
    containment_opportunity_proxy: Fraction
    dispositions: dict[Literal["allow", "block", "review", "score_only"], Count]

    @model_validator(mode="after")
    def consistent(self):
        if self.pending_action.n != self.coverage.predicted_actions or (
            self.run_unsafe_attempted.n != self.coverage.fully_covered_runs
            or self.run_unsafe_completed_proxy.n != self.coverage.fully_covered_runs
        ):
            raise ValueError("Metrics and coverage disagree")
        if sum(self.dispositions.values()) != self.coverage.predicted_actions:
            raise ValueError("Disposition coverage disagrees")
        return self


class Classes(Record):
    positive: Count
    negative: Count
    unknown: Count


class Adequacy(Record):
    runs: Count
    independent_groups: Count
    unsafe_actions: Count
    safe_actions: Count
    unknown_actions: Count
    run_classes: dict[Literal["unsafe_attempted", "unsafe_completed"], Classes]
    warnings: list[str]


class Observed(Record):
    runs: Count
    injected_runs: Count
    benign_runs: Count
    unknown_injection_labels: Count
    attack_success_rate: Fraction
    unknown_attack_outcomes: Count
    benign_task_completion_rate: Fraction
    unknown_benign_outcomes: Count
    unsafe_attempted_runs: Count
    unsafe_completed_runs: Count


class Run(Record):
    run_id: Digest
    scenario: str | None
    version: str | None
    task: str | None
    model: str | None
    max_steps: Annotated[int, Field(gt=0)] | None
    injection_present: bool | None
    attack_attempted: bool | None
    attack_completed: bool | None
    benign_completed: bool | None
    unsafe_attempted: bool | None
    unsafe_completed: bool | None
    recorded_mock_effect_count: Count | None
    unauthorized_effect_indices: list[Count]
    issues: list[str]


class Features(Record):
    tool: str
    operation: str
    resource_class: str
    sensitivity: str
    destination_trust: str
    authorization: str
    kind: Literal["proposal"]
    previous_tool: str
    transition: str
    result_status: str
    mismatch: str
    prior_sensitive_attempts: Count
    prior_sensitive_successes: Count
    send_after_sensitive_read: bool


class Prediction(Record):
    flagged: bool
    score: Probability | None
    disposition: Literal["allow", "block", "review"] | None


class Proposal(Record):
    run_id: Digest
    proposal_id: Digest
    call_id: Digest
    sequence: Annotated[int, Field(ge=1)]
    result_id: Digest | None
    effect_indices: list[Count]
    security_features: Features
    split: Split
    attempted_violation: bool | None
    completed_violation: bool | None
    prediction: Prediction | None
    category: Literal[
        "missing_prediction",
        "unknown_label",
        "true_positive",
        "false_negative",
        "false_positive",
        "true_negative",
    ]


class CurvePoint(Record):
    threshold: Probability
    metrics: Matrix


class Diagnostic(Record):
    proposal_id: Digest
    policy_available: bool
    detector_available: bool
    detector_score: Probability | None
    detector_threshold: Probability | None
    detector_status: str
    approval: str
    model_version: Digest | None
    policy_hash: Digest | None


class Timing(Record):
    count: Count
    median: Annotated[float, Field(ge=0)] | None
    p95: Annotated[float, Field(ge=0)] | None
    method: str | None = None
    partition: Split | None = None


class Latency(Record):
    saved_summary: Timing | None
    comparable_added_gateway_latency: bool | None = None
    reason: str | None = None
    method: str | None = None


class Method(Record):
    threshold: Probability | None
    calibration: str
    representation: str | None
    latency: Latency
    splits: dict[Split, Summary]
    proposals: list[Proposal]
    breakdowns_within_split: dict[Split, dict[str, dict[str, Summary]]] = Field(
        default_factory=dict
    )
    descriptive_threshold_sweeps: dict[Split, list[CurvePoint]] = Field(default_factory=dict)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    ignored_excluded_events: Count | None = None


class CommonCohort(Record):
    cohort: Literal["exact_common_proposal_ids_within_split"]
    common_count: Count
    left_only: Count
    right_only: Count
    left: Matrix
    right: Matrix
    disagreements: list[Digest]


class Comparison(Record):
    left_method: str
    right_method: str
    splits: dict[Split, CommonCohort]


class Hashes(Record):
    dataset: Digest
    manifest: Digest
    phase6: Digest | None
    phase7: Digest | None
    source_artifacts: dict[str, Digest]
    files: dict[str, Digest] = Field(default_factory=dict)


class Configuration(Record):
    version: Literal["phase9-v1"]
    minimum_independent_groups: Annotated[int, Field(ge=3)]
    thresholds: list[Probability]


class Experiment(Record):
    available: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    status: str | None = None
    reason: str | None = None


class Uncertainty(Record):
    confidence_intervals: None
    reason: str


class Report(Record):
    schema_version: Literal["phase9-v1"]
    config: Configuration
    input_hashes: Hashes
    source: Literal["scripted", "live"]
    membership: dict[Split, list[Digest]]
    definitions: dict[str, str]
    adequacy: dict[Split, Adequacy]
    independent_groups: Count
    exclusions: dict[Digest, list[str]]
    excluded_action_labels: Count
    input_issues: list[str]
    observed_source_outcomes: dict[Split, Observed]
    runs: list[Run]
    methods: dict[MethodName, Method]
    paired_comparisons: list[Comparison] = Field(default_factory=list)
    experiments: dict[str, Experiment] = Field(default_factory=dict)
    uncertainty: Uncertainty

    @model_validator(mode="after")
    def linked_records(self):
        if any(
            set(section) != SPLITS
            for section in (self.membership, self.adequacy, self.observed_source_outcomes)
        ):
            raise ValueError("All three splits are required")
        required_definitions = {
            "positive",
            "intervention",
            "score_alert",
            "rates",
            "unknowns",
            "run_prediction",
            "observed_outcomes",
            "containment_opportunity_proxy",
            "run_completion_proxy",
            "threshold_sweeps",
        }
        if set(self.definitions) != required_definitions:
            raise ValueError("Metric definitions are incomplete or unsupported")
        run_map = {r.run_id: r for r in self.runs}
        members = [rid for ids in self.membership.values() for rid in ids]
        if len(run_map) != len(self.runs) or len(set(members)) != len(members):
            raise ValueError("Duplicate run or split membership")
        if (
            set(members) & self.exclusions.keys()
            or set(members) | self.exclusions.keys() != run_map.keys()
        ):
            raise ValueError("Run membership and exclusions disagree")
        for rid, run in run_map.items():
            if run.issues != self.exclusions.get(rid, []):
                raise ValueError("Run exclusion reasons disagree")
        if sum(a.independent_groups for a in self.adequacy.values()) != self.independent_groups:
            raise ValueError("Independent group counts disagree")
        for part, members_in_split in self.membership.items():
            if self.adequacy[part].runs != len(members_in_split) or (
                self.observed_source_outcomes[part].runs != len(members_in_split)
            ):
                raise ValueError("Split run counts disagree")
        evidence = {}
        for method in self.methods.values():
            if set(method.splits) != SPLITS:
                raise ValueError("Method splits are incomplete")
            ids = set()
            for p in method.proposals:
                if p.proposal_id in ids or p.run_id not in self.membership[p.split]:
                    raise ValueError("Duplicate or misplaced proposal")
                ids.add(p.proposal_id)
                facts = p.model_dump(exclude={"prediction", "category"})
                if evidence.setdefault(p.proposal_id, facts) != facts:
                    raise ValueError("Methods disagree on proposal evidence")
            for part, summary in method.splits.items():
                rows = [p for p in method.proposals if p.split == part]
                adequacy = self.adequacy[part]
                if (
                    len(rows)
                    != adequacy.safe_actions + adequacy.unsafe_actions + adequacy.unknown_actions
                ):
                    raise ValueError("Method action coverage disagrees with split adequacy")
                if (
                    summary.coverage.total_actions != len(rows)
                    or (
                        summary.coverage.predicted_actions
                        != sum(p.prediction is not None for p in rows)
                    )
                    or summary.coverage.total_runs != len(self.membership[part])
                ):
                    raise ValueError("Proposal rows and coverage disagree")
            if any(d.proposal_id not in ids for d in method.diagnostics):
                raise ValueError("Unlinked gateway diagnostic")
        pairs = set()
        for pair in self.paired_comparisons:
            names = (pair.left_method, pair.right_method)
            key = tuple(sorted(names))
            if names[0] == names[1] or key in pairs or any(n not in self.methods for n in names):
                raise ValueError("Invalid comparison methods")
            pairs.add(key)
            if set(pair.splits) != SPLITS:
                raise ValueError("Comparison splits are incomplete")
            for part, cohort in pair.splits.items():
                left, right = [
                    {
                        p.proposal_id
                        for p in self.methods[n].proposals
                        if p.split == part and p.prediction is not None
                    }
                    for n in names
                ]
                common = left & right
                if (
                    (cohort.common_count, cohort.left_only, cohort.right_only)
                    != (len(common), len(left - right), len(right - left))
                    or cohort.left.n != len(common)
                    or cohort.right.n != len(common)
                ):
                    raise ValueError("Comparison cohort counts disagree")
                if (
                    len(set(cohort.disagreements)) != len(cohort.disagreements)
                    or not set(cohort.disagreements) <= common
                ):
                    raise ValueError("Unlinked comparison disagreements")
        return self
