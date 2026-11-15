"""Capture private Phase 2 fixture evidence for offline dataset pipeline exercises."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from runtime_security.scenarios.lab import SCENARIOS, run_scenario
from runtime_security.scenarios.live import LiveEvaluationRecord
from runtime_security.telemetry.recorder import JsonlRecorder


def record(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    recorder = JsonlRecorder(directory / "telemetry.jsonl")
    with (directory / "evidence.jsonl").open("x", encoding="utf-8") as stream:
        for scenario in SCENARIOS:
            for _ in range(3 if scenario.injection_present else 2):
                result = run_scenario(scenario, recorder=recorder)
                record = LiveEvaluationRecord(
                    timestamp=datetime.now(timezone.utc),
                    scenario_id=scenario.scenario_id,
                    scenario_version=scenario.scenario_version,
                    prompt=scenario.prompt,
                    run_id=result.agent_result.run_id,
                    model_name=result.model_name,
                    max_steps=result.max_steps,
                    result=result,
                )
                stream.write(record.model_dump_json() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    record(parser.parse_args().directory)
