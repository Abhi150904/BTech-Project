# BTech-Project

Behavioral Runtime Security Monitoring for Prompt-Injected LLM Agents.

The repository includes a bounded LangGraph agent, synthetic sandbox tools,
scenario and telemetry helpers, and shared Pydantic contracts. The agent is an
intentionally unprotected baseline for controlled research; no security gateway
or production protections are implemented.

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

## Development

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
