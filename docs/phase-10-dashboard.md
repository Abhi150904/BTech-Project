# Phase 10: local evaluation results dashboard

The dashboard opens one saved Phase 9 report at a time. It displays the report's
metrics, denominators, labels and compatible comparisons without running the
evaluator, a model, an agent, or a tool. It uses Python's standard-library
Tkinter desktop interface and the project's existing Pydantic dependency.
There is no HTTP server, browser, external asset, API call, or new dependency.

## Launch on Windows

From the repository root, using the existing editable installation:

```powershell
.\.venv\Scripts\python.exe -m runtime_security.dashboard `
  research/phase9-scripted-v1/report.json
```

To start with no report loaded, omit the path. Use **Open local report…** to
choose a JSON file from disk. Opening another report replaces the current one;
scripted and live results are never pooled. The header always identifies the
source and selected split. A failed load clears the previous report and shows
a safe error message, rather than leaving stale metrics on display.

For a fresh installation, follow the README's environment instructions and run:

```powershell
python -m pip install -e ".[dev]"
runtime-security-dashboard research/phase9-scripted-v1/report.json
```

`runtime-security-dashboard` is the equivalent installed entry point. Refresh
an existing editable install to create that entry point, or use the module
command immediately. Standard python.org Windows installations include Tk.
Linux distributions may supply Tk separately; a desktop display is required
for the GUI. Headless environments can validate a report without importing Tk:

```powershell
.\.venv\Scripts\python.exe -m runtime_security.dashboard `
  research/phase9-scripted-v1/report.json --check
```

The dashboard does not generate reports. For that separate workflow, see
[Phase 9 evaluation](phase-9-evaluation.md). Inputs are opened read-only; the
dashboard has no save, export, execution, approval, or training controls.

## Sections

| Section | What to inspect |
|---|---|
| Overview | Report-wide run, eligible proposal and independent-group counts; selected-split class coverage and observed attack/benign outcomes; coverage for all three splits. Full event count is explicitly N/A. |
| Methods | Action confusion counts and TPR, FPR, precision and F1, with numerator/denominator displays. Select a method row for the selected split's run matrices, benign interventions, coverage, saved threshold and latency scope. |
| Comparisons | Choose two methods. Only a saved Phase 9 pair is displayed, using its exact common cohort within the selected split. Method-only counts remain visible. No saved pair or no shared proposals means no usable comparison. |
| Breakdowns | Saved scenario, task, model or version summaries for the selected method and split. Empty optional breakdowns produce an empty table. Selecting a row exposes all saved denominators and coverage. |
| Proposals | Filter by scenario, task, model, version, outcome category (such as `false_positive`), tool or run ID. Rows stay ordered by run and sequence. Select a row for hashed IDs, labels, predictions, safe features, result links, recorded effects and available gateway diagnostics. |
| Runs | Selected-split runs, or explicitly selected report-wide excluded runs. Uses the Proposals text filter. Select a run for its exclusion reasons and the selected method's ordered proposals. |
| Thresholds | Saved curve points and TPR/FPR plot. Undefined values are gaps, not zero. The selected method's saved threshold is marked when present; per-action gateway thresholds remain in diagnostics. |
| Evidence & definitions | The report's metric definitions, E1–E6 evidence gaps, uncertainty statement, input issues and provenance hashes. |

Row filters affect inspection only; they never recalculate headline metrics.
Clear the filter to restore rows. Tables have horizontal/vertical scrollbars,
and the divider above the read-only detail pane can be dragged to resize it.

The default split is **test**. Threshold curves are descriptive analysis; the
UI explicitly warns when viewing train/test curves and forbids interpreting
test curves as a basis for threshold selection. Phase 7 has no saved validation
curve in the current report, which is shown as unavailable. Phase 6 validation
curves can be inspected by selecting a Phase 6 scoring method and `validation`.

## Schema and safety boundary

`dashboard/schema.py` defines a strict dashboard-side compatibility contract
for **phase9-v1**, matching `evaluation/report.py` without changing that writer.
It requires the core source, configuration, hashes, definitions, membership,
adequacy, observed outcomes, runs, methods and uncertainty sections. Comparisons,
experiment descriptions, breakdowns, curves and gateway diagnostics may be
absent or empty. Present optional content must still validate.

Validation rejects unsupported versions, unknown fields (including raw prompts
or tool arguments), duplicate JSON fields, nonfinite numbers, negative counts,
rates inconsistent with their supplied denominators, contradictory confusion
counts, misplaced/duplicate run or proposal IDs, and mismatched comparison
cohorts. These are consistency checks on supplied data, not a rerun of the
evaluator or a proof that the labels and provenance are authentic. The loader
does not follow artifact paths or retrieve data using hashes.

`inspection.py` opens local files only, limits input to 32 MiB, and rejects URLs
and UNC paths. Use reports stored on a local disk, not mapped network storage.
Error messages omit the input values and absolute filename. Native Tk text
APIs render strings literally: HTML tags, script tags, Tcl substitutions and
URLs have no executable or link behavior. There is no HTML renderer or `eval`.
Display helpers additionally mask recognizable token and absolute-path
patterns and remove control characters. Pattern masking is not a comprehensive
secret classifier; use the sanitized Phase 9 export, not raw research evidence.

The source report remains unchanged. Replacing a report cannot combine data
with the previous source. Rates remain fractions in the validated model and
are formatted as percentages for display; `null` appears as **N/A**. BLOCK and
REVIEW count as interventions under the report's definition. A saved score
alert is not silently interpreted as a completed block or prevented effect.

## Guide alignment and unavailable information

The implementation playbook asks for read-only run/timeline/outcome/decision
endpoints, filters and protected-versus-unprotected comparison. This phase's
explicit offline/no-network scope is implemented with local Python inspection
interfaces and native views instead of HTTP endpoints or a web framework. The
comprehensive guide describes FastAPI/Next.js as technology suggestions.

The existing Phase 9 format does **not** contain a complete event history or
total event count, raw legitimate task or injection text, matched policy rule
IDs, or paired executed protected/unprotected experiments. The UI exposes task
labels, ordered proposals, result IDs, semantic features, final dispositions,
available detector/policy diagnostics and recorded mock-effect counts. It does
not invent the missing information or infer a full timeline from sequence
numbers. Inspecting a method named `phase8/gateway` requires a report that
actually contains that saved evidence.

Current fixtures have 12 runs, 23 eligible proposals, three independent groups,
and no unsafe actions in validation/test. This class absence, scarce groups,
and scripted source are prominent warnings. Observed attack outcomes and benign
completion describe the original runs. Offline flags do not establish prevented
effects, post-intervention task completion, generalization or production security.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_dashboard.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```

Tests load the committed Phase 9 report and check schema errors, null metrics,
class warnings, source/split isolation, exact common cohorts, optional sections,
filters, literal rendering, safe errors and local-only loading. A fresh-process
test blocks network connections and evaluator/model/agent/tool imports, and
verifies the report bytes remain unchanged. Native-widget tests exercise split
and method changes, threshold messaging, non-executable text, and failed-load
clearing. Native tests skip explicitly when Tk or a display is unavailable;
the schema and headless checks do not need a display.
