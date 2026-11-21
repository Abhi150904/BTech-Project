"""Compare existing synthetic runs with and without the existing gateway."""

import argparse
import json
from importlib.util import find_spec

from runtime_security.scenarios import BENIGN, INJECTION
from runtime_security.scenarios.lab import run_scenario
from runtime_security.telemetry.recorder import InMemoryRecorder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unprotected-only", action="store_true", help="Core-only demonstration")
    args = parser.parse_args(argv)
    if not args.unprotected_only and find_spec("torch") is None:
        parser.error("Protected demonstration requires the neural extra; use --unprotected-only")
    from runtime_security.gateway.decisions import CHECKPOINT_VERSION, Containment

    results = []
    for scenario in (BENIGN, INJECTION):
        for protected in (False,) if args.unprotected_only else (False, True):
            recorder = InMemoryRecorder()
            result = run_scenario(
                scenario,
                recorder=recorder,
                containment=Containment() if protected else None,
            )
            events = recorder.for_run(result.agent_result.run_id)
            record = {
                "scenario": result.scenario_id,
                "scenario_version": result.scenario_version,
                "model": "deterministic-scripted",
                "protected": protected,
                "checkpoint": CHECKPOINT_VERSION if protected else None,
                "max_steps": 10,
                "attack_attempted": result.attack_attempted,
                "attack_completed": result.attack_completed,
                "benign_task_completed": result.benign_task_completed,
                "recorded_mock_requests": len(result.recorded_side_effects),
                "decisions": [
                    {
                        "sequence": e.sequence,
                        "tool": e.tool_name,
                        "decision": e.summary.get("decision"),
                    }
                    for e in events
                    if e.kind == "decision"
                ],
            }
            results.append(record)
            if scenario is BENIGN:
                assert result.benign_task_completed and not result.recorded_side_effects
            elif protected:
                assert not result.attack_completed and not result.recorded_side_effects
                assert any(e.summary.get("execution") == "not_started" for e in events)
            else:
                assert result.attack_completed and len(result.recorded_side_effects) == 1
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
