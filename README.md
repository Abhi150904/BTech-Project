# Behavioral Runtime Security Monitoring for Prompt-Injected LLM Agents

A BTech research prototype with a bounded LangGraph agent, synthetic tools,
ordered telemetry, semantic features, offline datasets/baselines/BiLSTM,
pre-execution containment, evaluation reports and a read-only desktop dashboard.
The 12-run, three-group scripted corpus is a pipeline fixture. Validation/test
contain no unsafe proposals; it does not establish detector generalization or
production security.

## Setup

Python 3.11+ is declared; Windows x64 with CPython 3.12.10 is the tested handoff
environment. From a clean checkout, one command creates/reuses `.venv`, installs
development/build dependencies and checks dependencies:

```powershell
python scripts/setup.py
# Include scikit-learn and Torch for all tests, training and protected demos:
python scripts/setup.py --full
```

The script applies recorded constraints on Windows/Python 3.12. Other platforms
resolve declared ranges and require separate validation. Installation can
download packages; offline runs below need no API key.

For manual installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
# Extras: dev (tests/lint), build, baselines (sklearn), neural (Torch)
.\.venv\Scripts\python.exe -m pip install -e ".[dev,build,baselines,neural]"
```

Core installation supports the deterministic lab, dataset/evaluation commands
using saved predictions, and dashboard. Training/scoring need their ML extra.
The pinned containment detector requires `neural`; missing detector support
fails closed and is not a successful protected demonstration.

## Offline demonstration and validation

```powershell
.\.venv\Scripts\runtime-security-scenarios.exe
# Existing benign and injected scenarios, without versus with the gateway:
.\.venv\Scripts\python.exe scripts/offline_demo.py
# Core-only alternative:
.\.venv\Scripts\python.exe scripts/offline_demo.py --unprotected-only
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
```

Unprotected injected scenarios intentionally report security failure. The demo
checks recorded mock effects independently of agent text. Tools read only
allowlisted synthetic fixtures; `mock_http_post` records an in-memory request
and never sends network traffic. Protected refusals can leave the legitimate
task incomplete; security and utility remain separate outcomes.

## Saved evaluation and dashboard

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
.\.venv\Scripts\runtime-security-evaluate.exe `
  research/scripted-v1/dataset.json research/scripted-v1/manifest.json `
  --baselines research/scripted-v1/baselines.json `
  --neural research/phase7-scripted-v1/report.json `
  --config configs/experiments/phase9.toml --out artifacts/phase9-local.json
.\.venv\Scripts\python.exe -m runtime_security.dashboard artifacts/phase9-local.json --check
.\.venv\Scripts\python.exe -m runtime_security.dashboard artifacts/phase9-local.json
```

Choose a new output filename on each evaluation. Tk/display is required for
the GUI; `--check` is headless. Outside a checkout, pass explicit saved-input
paths. The wheel includes runtime fixtures/configuration and the pinned gateway
checkpoint; the sdist includes reviewed research inputs, tests and docs.

## Build and handoff

```powershell
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe scripts/verify_distribution.py `
  dist/btech_runtime_security-0.1.0-py3-none-any.whl `
  dist/btech_runtime_security-0.1.0.tar.gz
```

See [Phase 11 handoff](docs/phase-11-handoff.md) for fresh wheel installation,
archive checks, actual validation results and platform limits. Builds stay local;
these commands do not publish a package, release, or service.

## Research and implementation references

- [Methodology/trust boundaries](docs/methodology.md), [results](docs/results.md),
  and [paper-style write-up](paper/README.md).
- [Scenarios](docs/phase-2-scenarios.md), [monitoring](docs/phase-3-monitoring.md),
  [features](docs/phase-4-features.md), [datasets](docs/phase-5-datasets.md).
- [Baselines](docs/phase-6-baselines.md), [BiLSTM](docs/phase-7-bilstm.md),
  [containment](docs/phase-8-containment.md), [evaluation](docs/phase-9-evaluation.md),
  [dashboard](docs/phase-10-dashboard.md), [shared contracts](docs/interfaces.md).

## Live use is separate

Only a deliberate live model run needs `.env` and `OPENAI_API_KEY`. Copy
`.env.example` to `.env` and set `OPENAI_MODEL` when preparing such a run.
The generic `runtime-security` CLI has no task grants; unresolved tool actions
are refused/reviewed. Read [live evaluation](docs/phase-3-live-evaluation.md) and
the containment documentation first. Packaging checks never call OpenAI.

Never commit `.env`, API keys, private data, unsanitized live evidence or
temporary outputs. Keep new reports under ignored `artifacts/phase9-*.json`
and release checks under `artifacts/release-checks/`; use a new experiment
directory for new model artifacts. Preserve frozen `research/` provenance.
The tracked `.env.example` deliberately contains an empty key.
