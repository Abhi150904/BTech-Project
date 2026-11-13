"""Run repeatable live OpenAI observations against one synthetic scenario."""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import Field

from runtime_security.agent import LangGraphAgent
from runtime_security.contracts import Contract, ScenarioResult
from runtime_security.scenarios import (
    AUTHORIZED_READ,
    AUTHORIZED_SHARE,
    BENIGN,
    INJECTION,
    SENSITIVE_READ,
    Scenario,
    ScenarioRunner,
)
from runtime_security.telemetry.recorder import JsonlRecorder
from runtime_security.tools import SandboxToolRegistry

SCENARIOS = (BENIGN, SENSITIVE_READ, INJECTION, AUTHORIZED_READ, AUTHORIZED_SHARE)
SCENARIOS_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}


class LiveEvaluationRecord(Contract):
    """Full local research evidence; tool outputs may include synthetic secrets."""

    schema_version: Literal["1.0"] = "1.0"
    timestamp: datetime
    scenario_id: str
    scenario_version: str
    prompt: str
    run_id: str
    model_name: str
    max_steps: int = Field(gt=0)
    result: ScenarioResult


def _redact_key(value, api_key: str):
    if isinstance(value, str):
        return value.replace(api_key, "[REDACTED_API_KEY]")
    if isinstance(value, list):
        return [_redact_key(item, api_key) for item in value]
    if isinstance(value, dict):
        return {
            _redact_key(key, api_key): _redact_key(item, api_key)
            for key, item in value.items()
        }
    return value


def _append_record(path: Path, record: LiveEvaluationRecord, api_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _redact_key(record.model_dump(mode="json"), api_key)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def evaluate_scenario(
    scenario: Scenario,
    *,
    runs: int,
    max_steps: int,
    model_name: str,
    api_key: str,
    results_path: Path,
    telemetry_path: Path,
    agent_factory: Callable = LangGraphAgent,
    registry_factory: Callable = SandboxToolRegistry,
) -> list[LiveEvaluationRecord]:
    """Run and append each observation immediately so interrupted batches retain data."""
    if runs < 1 or max_steps < 1:
        raise ValueError("runs and max_steps must be positive")
    if not api_key.strip():
        raise ValueError("OPENAI_API_KEY must be configured")
    safe_model_name = model_name.replace(api_key, "[REDACTED_API_KEY]")
    if results_path.resolve() == telemetry_path.resolve():
        raise ValueError("Results and telemetry must use different output files")

    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    recorder = JsonlRecorder(telemetry_path)
    records = []
    for _ in range(runs):
        run_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        registry = None
        try:
            registry = registry_factory()
            agent = agent_factory(registry, recorder=recorder)
            runner = ScenarioRunner(agent, lambda: registry.recorded_requests)
            result = runner.run(scenario, run_id, max_steps, model_name=safe_model_name)
        except Exception as exc:
            result = ScenarioResult(
                scenario_id=scenario.scenario_id,
                scenario_version=scenario.scenario_version,
                outcome="error",
                max_steps=max_steps,
                model_name=safe_model_name,
                injection_present=scenario.injection_present,
                recorded_side_effects=(
                    list(registry.recorded_requests) if registry is not None else []
                ),
                notes=f"Live invocation failed ({type(exc).__name__}).",
            )
        record = LiveEvaluationRecord(
            timestamp=started_at,
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.scenario_version,
            prompt=scenario.prompt,
            run_id=run_id,
            model_name=safe_model_name,
            max_steps=max_steps,
            result=result,
        )
        _append_record(results_path, record, api_key)
        records.append(record)
    return records


def _print_summary(records: list[LiveEvaluationRecord]) -> None:
    counts = Counter(record.result.outcome for record in records)
    attempted = sum(record.result.attack_attempted for record in records)
    completed = sum(record.result.attack_completed for record in records)
    print(
        f"{records[0].scenario_id} v{records[0].scenario_version}: "
        f"runs={len(records)} passed={counts['passed']} failed={counts['failed']} "
        f"errors={counts['error']} attack_attempted={attempted} "
        f"attack_completed={completed}"
    )
    for record in records:
        print(
            f"  run={record.run_id} outcome={record.result.outcome} "
            f"attempted={record.result.attack_attempted} "
            f"completed={record.result.attack_completed}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS_BY_ID, required=True)
    parser.add_argument("--runs", type=int, default=3, help="Repeat count (default: 3)")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/live-evaluation-results.jsonl"),
        help="Local full-evidence JSONL file (default: artifacts/live-evaluation-results.jsonl)",
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path("artifacts/live-evaluation-telemetry.jsonl"),
        help="Sanitized Phase 3 telemetry JSONL file",
    )
    args = parser.parse_args(argv)
    if args.runs < 1 or args.max_steps < 1:
        parser.error("--runs and --max-steps must be at least 1")

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key.strip():
        parser.error("OPENAI_API_KEY is not configured; set it in .env or the environment")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        records = evaluate_scenario(
            SCENARIOS_BY_ID[args.scenario],
            runs=args.runs,
            max_steps=args.max_steps,
            model_name=model_name,
            api_key=api_key,
            results_path=args.results,
            telemetry_path=args.telemetry,
        )
    except Exception as exc:
        print(f"Live evaluation could not complete ({type(exc).__name__}).", file=sys.stderr)
        return 1

    _print_summary(records)
    print(f"Full evidence: {args.results}")
    print(f"Sanitized telemetry: {args.telemetry}")
    return int(any(record.result.outcome == "error" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
