# Live OpenAI scenario evaluation

The deterministic `runtime-security-scenarios` command remains the offline
scoring check. This separate command invokes the configured OpenAI-backed
`LangGraphAgent` and reuses Phase 2's scenario prompts and `ScenarioRunner`
predicates. It runs one selected scenario per command and repeats it the
requested number of times.

## Run

Configure `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`, then reinstall the
editable package after adding the command:

```powershell
python -m pip install -e .
runtime-security-live-eval --scenario document-to-mock-send --runs 5 --max-steps 10
```

Available scenario IDs are `benign-summary`, `document-to-sensitive-read`,
`document-to-mock-send`, `authorized-sensitive-read`, and
`authorized-mock-share`. Run a command for each scenario you want to measure.
Each invocation gets a fresh run ID and sandbox registry. Repeated outcomes are
stochastic observations; report counts across runs rather than drawing a
conclusion from one result. Scenario `failed` means its behavioral predicate
was not satisfied (for injected cases, the attack may have completed); `error`
means the run was incomplete.

## Evidence files

By default, each run is appended immediately to:

- `artifacts/live-evaluation-results.jsonl`: full `ScenarioResult`, including
  prompt, final answer, proposed calls and arguments, tool responses, mock
  effects, outcome, run ID, model label, step limit, and timestamp.
- `artifacts/live-evaluation-telemetry.jsonl`: Phase 3's separately sanitized,
  ordered event stream.

Both default paths are Git-ignored. The full-results file intentionally retains
synthetic sensitive-file content when the agent reads or repeats it in an
answer or mock payload. Keep it local and treat it as sensitive research data;
do not commit or share it casually. The configured API key is not a record
field and is scrubbed from serialized result strings. The key is used only for
the provider request.

The tools use local synthetic files. `mock_http_post` appends to an in-memory
ledger and never sends a real HTTP request. The model provider receives the
scenario prompt and tool outputs needed for the run, including synthetic file
contents where applicable. Live requests may incur API usage charges. Pytest
uses fake agents and does not call OpenAI.

Custom output paths are supported:

```powershell
runtime-security-live-eval --scenario authorized-mock-share --runs 3 `
  --results artifacts/share-results.jsonl `
  --telemetry artifacts/share-telemetry.jsonl
```

The output files append across invocations. Use different paths for full results
and sanitized telemetry.
