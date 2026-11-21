# BTech-Project

Behavioral Runtime Security Monitoring for Prompt-Injected LLM Agents.

The repository includes a bounded LangGraph agent, synthetic sandbox tools,
scenario and telemetry helpers, and shared Pydantic contracts. The agent gates
tool execution using trusted policy, a Phase 7 detector and optional host review.
See the [Phase 8 containment workflow](docs/phase-8-containment.md) for setup,
task grants and offline examples.

## Install (Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The example selects `gpt-4.1-mini`; change
`OPENAI_MODEL` there to choose another model. Agent runs use the OpenAI API;
tests use mocked model responses and do not need a key.

## Run the CLI

```powershell
runtime-security "Read summary.txt and summarize the Atlas project."
runtime-security --max-steps 6 "Find the Atlas project milestones."
```

The CLI prints the final answer and the tool calls with their recorded response
statuses. Tools read only allowlisted synthetic sandbox fixtures; the mock POST
tool records requests locally and does not send network traffic.

Run the deterministic Phase 2 scenario checks without an OpenAI API call:

```powershell
runtime-security-scenarios
```

The suite covers a benign summary, injected sensitive-read and mock-send
attempts, and explicitly authorized read and sharing lookalikes. It prints
attempted actions and completed effects as separate outcomes.

See [Phase 2 scenario lab](docs/phase-2-scenarios.md) for scenario definitions
and outcome criteria.

For live OpenAI observations, run one scenario at a time with
`runtime-security-live-eval`. See [live evaluation usage](docs/phase-3-live-evaluation.md)
for repeat counts, evidence files, and safety notes.

## Development

Inspect a saved report in the local, read-only
[Phase 10 results dashboard](docs/phase-10-dashboard.md):

```powershell
python -m runtime_security.dashboard research/phase9-scripted-v1/report.json
```

Generate deterministic comparisons from saved labels and predictions with the
[Phase 9 offline evaluation workflow](docs/phase-9-evaluation.md). Reports include
coverage, explicit denominators, exclusions and research-evidence limitations.

Train the two offline sequence detectors or score a pending tool proposal with
the [Phase 7 BiLSTM workflow](docs/phase-7-bilstm.md).

For rule, sequence and conventional ML comparisons on a frozen dataset, see
the [Phase 6 baseline workflow](docs/phase-6-baselines.md).

For offline corpus generation, label audits and frozen splits, see the
[Phase 5 dataset workflow](docs/phase-5-datasets.md).

Phase 3 adds sanitized runtime events, append-only JSONL, and read-only replay.
See [monitoring usage and limitations](docs/phase-3-monitoring.md) for offline
scenario recording, event fields, redaction, and recorder failure behavior.

```powershell
pytest
ruff check .
```

The `src/runtime_security/` package contains `agent`, `tools`, `contracts`,
`telemetry`, `scenarios`, and the CLI entry point. `sandbox/synthetic/` contains
research fixtures. Use synthetic data only; never commit credentials or real private files.

See [shared interfaces](docs/interfaces.md) for the contracts used across modules.
Dependencies use bounded major versions; freeze a shared lockfile when the
integrated environment is agreed.
