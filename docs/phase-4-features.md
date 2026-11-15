# Phase 4 security-semantic features

This offline API converts Phase 3 sanitized trajectory events into one feature
record per event. It accepts a complete run or a contiguous prefix beginning
with `run_start`. Use the telemetry JSONL from the recorder or the live
evaluator's **telemetry** output. The live evaluator's full research evidence
contains prompts, outputs and outcome labels and is outside this API.

## Usage

```python
from pathlib import Path
from runtime_security.features import FeatureEncoder, extract_run, load_policy
from runtime_security.telemetry.recorder import read_events

policy = load_policy(Path("configs/policy_metadata.toml"))
def rows(path, run_id):
    return extract_run((e for e in read_events(path) if e.run_id == run_id), policy)

training_run = rows(Path("training-telemetry.jsonl"), "train-1")
evaluation_run = rows(Path("evaluation-telemetry.jsonl"), "eval-1")
encoder = FeatureEncoder().fit([training_run])
encoded = encoder.transform(evaluation_run, window_size=4)
print(encoded[0].sequence, encoded[0].vector, encoded[0].window)
```

For an in-memory recorder, pass `recorder.for_run(run_id)` directly to
`extract_run`. An empty run yields no records. Sequences are sorted and checked
for gaps, duplicate IDs and mixed runs. The extractor reuses Phase 3's serial
lifecycle validator to check metadata consistency and proposal/decision/result
pairing. Wall-clock order plays no role. Keep call IDs for audit joins via the
source event. Each feature record carries run ID, event ID and sequence outside
its model fields.

## Example and schema

Consider sequence 2 of task `authorized-sensitive-read`, version `1.0`.
The observed proposal names `read_file`, call `c1`, and has the sanitized
argument `{"file_id": "sensitive.txt"}`. Its semantic fields are

| Field | Example | Domain or meaning |
|---|---|---|
| tool | read_file | read_document, read_file, search_files, mock_http_post |
| operation | read | read, search, send |
| resource_class | file | document, file, destination |
| sensitivity | synthetic_sensitive | public, synthetic_sensitive |
| destination_trust | not_applicable | approved, unapproved |
| authorization | authorized | authorized, unauthorized |
| kind | proposal | run_start, proposal, decision, tool_result, run_complete |
| previous_tool | start | latest strictly earlier proposed tool, or start |
| transition | start->read_file | previous tool to current proposed tool |
| result_status | not_applicable | success or error at the matched result |
| mismatch | no | yes or no for a proposal with known authorization |
| prior_sensitive_attempts | 0 | number of strictly earlier proposals to read a sensitive resource |
| prior_sensitive_successes | 0 | number of strictly earlier successful results for those calls |
| send_after_sensitive_read | false | send proposal following at least one successful sensitive read |

Categorical domains also use `unknown` for missing/unmapped evidence and
`not_applicable` where the field has no meaning. Search concerns documents as
a collection, with no individual resource sensitivity. Decision and result rows
inherit the matched proposal's action categories. Results use only the
allowlisted status recorded at that event. On a successful sensitive result the
row shows success, but the prior-success counter first increments on the next
row. Protocol-rejected proposals therefore count as attempts, with error results.
Transition and mismatch apply only to proposals. Run boundary rows retain history
counts and previous tool, with action fields set to not_applicable.

## Trusted metadata and research limits

The reviewed TOML catalogs assign sensitivity to sandbox IDs and trust to mock
destinations. The Python standard library parses TOML and Pydantic validates
the domains, grant fields and target references. Tool-to-operation mappings live
in the extractor. A mapped scenario and version select a closed allowlist of
tool/target grants. A known target omitted from those grants is unauthorized.
Redacted/unmapped targets or missing task versions yield unknown authorization.
A search grant applies to the operation. Proposal text, response content and
model-supplied classifications are never policy sources.

Authorization measures **tool and target scope**. For example, the authorized
public-share task grants the approved destination. Redaction prevents checking
what bytes were sent. The send-after-read feature describes temporal opportunity
for cross-boundary movement, independent of destination trust. It cannot
establish that sensitive content moved. Review these semantics before changing
grants. Keep the config revision with each experiment. Authorization and mismatch
are direct policy-derived signals suitable for ablation. Sensitivity, trust and
the sensitive-history fields also depend on fixture metadata.

## Encoding and leakage controls

The table order above is the vector order (14 integer columns). Fit separately
for each training split. For each categorical column, lexically sorted distinct
training values receive IDs starting at 2. Code 1 represents explicit unknown
and unseen categories. Code 0 is reserved for padding. Codes are field-local
nominal indices, so downstream models should treat them as categories.
The last three columns are unscaled counts and a 0/1 flag. Transform leaves
the vocabulary fixed and a second fit on the same encoder is rejected.

Each window contains the last W vectors of its own run through the current
event. All-zero rows left-pad early windows. Actual rows always have nonzero
categorical codes. Causality holds for a fixed fitted encoder and policy. The
caller supplies the split and must fit exclusively on the training partition.
Training IDs, scenario/version, event IDs, labels, prompts, timestamps, model
names and raw arguments are excluded from the explicit encoder column list.
The scenario lookup can still correlate with task distribution through its
grants. Keep task-level evaluation and policy-feature ablations in later work.
