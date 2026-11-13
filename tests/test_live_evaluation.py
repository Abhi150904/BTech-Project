import json

from runtime_security.scenarios import BENIGN, INJECTION
from runtime_security.scenarios.lab import DeterministicScenarioAgent
from runtime_security.scenarios.live import evaluate_scenario
from runtime_security.telemetry.recorder import read_events


def test_repeated_live_evaluation_records_evidence_without_api_key(tmp_path):
    api_key = "test-only-openai-key"
    results_path = tmp_path / "nested" / "results.jsonl"
    telemetry_path = tmp_path / "nested" / "telemetry.jsonl"

    def agent_factory(registry, recorder):
        scripted = DeterministicScenarioAgent(registry, BENIGN, recorder)

        class KeyEchoingTestAgent:
            def invoke(self, request):
                result = scripted.invoke(request)
                result.final_answer = f"scripted answer {api_key}"
                return result

        return KeyEchoingTestAgent()

    records = evaluate_scenario(
        BENIGN,
        runs=2,
        max_steps=4,
        model_name="test-model",
        api_key=api_key,
        results_path=results_path,
        telemetry_path=telemetry_path,
        agent_factory=agent_factory,
    )

    serialized = results_path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in serialized.splitlines()]
    assert len(records) == len(rows) == 2
    assert len({row["run_id"] for row in rows}) == 2
    assert all(row["scenario_id"] == BENIGN.scenario_id for row in rows)
    assert all(row["scenario_version"] == BENIGN.scenario_version for row in rows)
    assert all(row["model_name"] == "test-model" for row in rows)
    assert all(row["max_steps"] == 4 for row in rows)
    assert all(row["result"]["agent_result"]["tool_calls"] for row in rows)
    assert all(row["result"]["outcome"] == "passed" for row in rows)
    assert "scripted answer [REDACTED_API_KEY]" in serialized
    assert api_key not in serialized
    telemetry = read_events(telemetry_path)
    assert {event.run_id for event in telemetry} == {row["run_id"] for row in rows}
    assert all(event.metadata.model_name == "test-model" for event in telemetry)
    assert api_key not in telemetry_path.read_text(encoding="utf-8")


def test_live_evaluation_keeps_incomplete_runs_as_records(tmp_path):
    results_path = tmp_path / "results.jsonl"
    telemetry_path = tmp_path / "telemetry.jsonl"

    def broken_agent_factory(registry, recorder):
        raise RuntimeError("provider unavailable")

    records = evaluate_scenario(
        INJECTION,
        runs=1,
        max_steps=3,
        model_name="test-model",
        api_key="test-key",
        results_path=results_path,
        telemetry_path=telemetry_path,
        agent_factory=broken_agent_factory,
    )

    row = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0].result.outcome == "error"
    assert row["result"]["outcome"] == "error"
    assert row["run_id"]
    assert row["result"]["notes"] == "Live invocation failed (RuntimeError)."


def test_live_evaluation_preserves_injected_attack_evidence(tmp_path):
    results_path = tmp_path / "results.jsonl"
    telemetry_path = tmp_path / "telemetry.jsonl"

    def scripted_agent_factory(registry, recorder):
        return DeterministicScenarioAgent(registry, INJECTION, recorder)

    records = evaluate_scenario(
        INJECTION,
        runs=1,
        max_steps=5,
        model_name="scripted-test-model",
        api_key="test-key",
        results_path=results_path,
        telemetry_path=telemetry_path,
        agent_factory=scripted_agent_factory,
    )

    row = json.loads(results_path.read_text(encoding="utf-8"))
    result = row["result"]
    assert records[0].result.attack_attempted
    assert records[0].result.attack_completed
    assert result["agent_result"]["final_answer"]
    assert result["agent_result"]["tool_calls"]
    assert result["agent_result"]["tool_responses"]
    assert result["recorded_side_effects"][0]["destination"] == INJECTION.attack_destination
    assert INJECTION.attack_payload_marker in result["recorded_side_effects"][0]["payload"]
