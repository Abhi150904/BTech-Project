# Phase 11: testing, packaging and handoff

This is a local research handoff, not a public release or production-security
certification. Phases 9 and 10 were merged before this branch started.

## Audited gaps and fixes

| Gap | Resolution |
|---|---|
| Repository-relative defaults break installed wheels | `assets.py` resolves defaults relative to the editable checkout or wheel `_data`, never the working directory. Explicit caller paths still work. |
| Resources absent from package discovery | The setuptools build copies nine reviewed resources byte-for-byte. Frozen artifacts are not regenerated. |
| Line-ending conversion breaks the checkpoint's policy hash | `.gitattributes` preserves the existing CRLF policy/dataset config and LF model config. Archive inspection verifies that binding. |
| No archive inspection | `verify_distribution.py` checks module/resource bytes, unwanted files and required source handoff content. |
| Optional dependencies and setup unclear | One-command setup, explicit extras and tested Windows/Python 3.12 constraints. |
| No outside-checkout installation check | `smoke_install.py` requires a wheel import and checks console startup, lab, dataset validation, evaluation, headless dashboard and demo. |

The full suite includes unit, integration, security regression and end-to-end
synthetic tests. This phase adds resource/packaging regressions without changing
policy rules, detector weights, scenarios or evaluation semantics. Docker is a
suggested guide file; no unverified container configuration is added to this
desktop-oriented handoff.

## Setup and builds

Run `python scripts/setup.py --full` from a clean checkout, then:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe scripts/verify_distribution.py `
  dist/btech_runtime_security-0.1.0-py3-none-any.whl `
  dist/btech_runtime_security-0.1.0.tar.gz
```

The standard build uses setuptools and builds the wheel from the sdist. The
wheel includes code, four synthetic files, four TOML files and the pinned enriched
checkpoint. The sdist additionally includes explicitly selected frozen research
inputs/reports, tests and docs. Local presentation/viva files, `.env`, virtual
environments, caches, editor state and generated evidence are excluded.
`.env.example` is included in the sdist as a blank-key template.

`requirements/windows-py312-tested.txt` records reachable installed dependency
versions for Windows/CPython 3.12, including optional groups. It is a constraints
snapshot, not a cross-platform hash lock. Installation may download packages;
runtime checks are offline. Python 3.11+ is declared; only the exercised
Windows/CPython version is verified here.

## Fresh wheel installation outside the checkout

Start in the checkout and choose a new directory:

```powershell
$project = (Get-Location).Path
$check = Join-Path $env:TEMP ('btech-handoff-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $check | Out-Null
python -m venv (Join-Path $check 'venv')
$python = Join-Path $check 'venv/Scripts/python.exe'
$wheel = Join-Path $project 'dist/btech_runtime_security-0.1.0-py3-none-any.whl'
& $python -m pip install $wheel -c (Join-Path $project 'requirements/windows-py312-tested.txt')
Push-Location $check
& $python -m pip check
& $python (Join-Path $project 'scripts/smoke_install.py') --inputs $project
Pop-Location
```

The core pass explicitly skips the three optional ML startup checks. To check
all declared console commands and the actual pinned containment model:

```powershell
& $python -m pip install "$wheel[baselines,neural]" `
  -c (Join-Path $project 'requirements/windows-py312-tested.txt')
Push-Location $check
& $python (Join-Path $project 'scripts/smoke_install.py') --inputs $project --full
Pop-Location
```

`--inputs` can also point to a freshly extracted sdist. Imports must come from
the environment's wheel, while research inputs remain explicit paths. The
scenario command has no help parser, so the smoke check runs its deterministic
lab directly; other console startup checks use `--help`. Generated reports go
to a temporary directory, and API-key/PYTHONPATH variables are removed from
child processes. No live model request is made.

Missing Torch is not a successful protected demo: install `neural` for the full
check. Tk/display is required only for the GUI. Headless validation is exercised
without opening a window. Optional/display skips must be reported explicitly.

## Handoff evidence and boundaries

Actual results are recorded in [results](results.md). Build archives and logs
stay local under ignored `dist/` and `artifacts/release-checks/`. No registry,
artifact-service upload, deployment or public release is part of this phase.

The guide's research write-up is covered by [methodology](methodology.md),
[results](results.md) and [paper/README.md](../paper/README.md). Existing phase
docs contain dataset/split/training commands. The offline demo compares existing
benign and injected scenarios with and without containment, using ledger counts
and recorded refusals rather than model text as evidence.

The corpus remains 12 scripted runs and three independent groups; held-out
splits lack unsafe actions. Fresh installation checks establish reproducibility
of the exercised software workflows, not detector generalization. Synthetic
POSTs never contact a real endpoint, and the authorized-share false positive
and legitimate-task cost of refusals remain documented limitations.
