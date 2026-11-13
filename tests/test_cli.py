from unittest.mock import Mock

from runtime_security.contracts import AgentResult, ToolCallEvent, ToolResponse


def test_cli_shows_tool_status_without_printing_arguments(monkeypatch, capsys):
    from runtime_security import cli

    result = AgentResult(
        run_id="run-1",
        final_answer="Completed.",
        tool_calls=[
            ToolCallEvent(
                run_id="run-1",
                call_id="call-1",
                tool_name="mock_http_post",
                arguments={"destination": "https://collector.invalid", "payload": "SECRET"},
            )
        ],
        tool_responses=[ToolResponse(call_id="call-1", status="success", output="SECRET")],
    )
    agent = Mock()
    agent.invoke.return_value = result
    monkeypatch.setattr(cli, "LangGraphAgent", lambda registry: agent)

    assert cli.main(["send the report"]) == 0

    output = capsys.readouterr().out
    assert "Completed." in output
    assert "mock_http_post [success]" in output
    assert "collector.invalid" not in output
    assert "SECRET" not in output
