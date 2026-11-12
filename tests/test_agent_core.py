from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from runtime_security.agent import LangGraphAgent
from runtime_security.contracts import AgentRequest
from runtime_security.tools import SandboxToolRegistry


def call(name="read_document", args=None, identifier="call-1"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args or {"document_id": "summary.txt"},
                "id": identifier,
            }
        ],
    )


def make_agent(messages, registry=None, executor=None):
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = messages
    return LangGraphAgent(registry or SandboxToolRegistry(), model, executor), model


def test_multistep_and_ids(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent, model = make_agent(
        [
            call(),
            call("read_file", {"file_id": "public.txt"}, "c2"),
            AIMessage(content="The prototype is complete."),
        ]
    )
    result = agent.invoke(AgentRequest(run_id="run", prompt="summary", max_steps=3))
    assert model.invoke.call_count == 3
    assert result.final_answer == "The prototype is complete."
    assert [c.run_id for c in result.tool_calls] == ["run", "run"]
    assert [r.call_id for r in result.tool_responses] == ["call-1", "c2"]
    assert all(r.status == "success" for r in result.tool_responses)
    assert model.invoke.call_args.args[0][-1].tool_call_id == "c2"


@pytest.mark.parametrize("limit", [1, 2, 15])
def test_step_limit(limit):
    agent, model = make_agent([call(identifier=f"c{i}") for i in range(limit)])
    result = agent.invoke(AgentRequest(run_id="r", prompt="go", max_steps=limit))
    assert model.invoke.call_count == limit
    assert len(result.tool_responses) == limit
    assert result.final_answer == "Maximum agent steps reached."


def test_provider_failure():
    agent, _ = make_agent([RuntimeError("provider secret")])
    result = agent.invoke(AgentRequest(run_id="r", prompt="go"))
    assert result.final_answer == "The model provider failed to complete this request."
    assert not result.tool_calls


@pytest.mark.parametrize("message", [call("unknown"), call(args={"document_id": "../x"})])
def test_tool_errors_are_returned_to_model(message):
    agent, model = make_agent([message, AIMessage(content="Unable to read that document.")])
    result = agent.invoke(AgentRequest(run_id="r", prompt="go"))
    assert result.tool_responses[0].status == "error"
    assert model.invoke.call_args.args[0][-1].tool_call_id == "call-1"


def test_executor_boundary_sees_proposals():
    executor = Mock()
    executor.execute.side_effect = RuntimeError("execution unavailable")
    agent, _ = make_agent([call(), AIMessage(content="Tool unavailable.")], executor=executor)
    result = agent.invoke(AgentRequest(run_id="r", prompt="go"))
    assert executor.execute.call_args.args[0] == result.tool_calls[0]
    assert result.tool_responses[0].error == "execution unavailable"


def test_environment_configuration(monkeypatch):
    import runtime_security.agent.core as core

    load = Mock()
    constructor = Mock()
    monkeypatch.setattr(core, "load_dotenv", load)
    monkeypatch.setattr(core, "ChatOpenAI", constructor)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    LangGraphAgent(SandboxToolRegistry())
    load.assert_called_once()
    constructor.assert_called_once_with(api_key="fake-key", model="fake-model")


def test_parallel_proposals_share_one_cycle():
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_document", "args": {"document_id": "summary.txt"}, "id": "a"},
            {"name": "read_file", "args": {"file_id": "public.txt"}, "id": "b"},
        ],
    )
    agent, model = make_agent([message])
    result = agent.invoke(AgentRequest(run_id="r", prompt="go", max_steps=1))
    assert model.invoke.call_count == 1
    assert [response.call_id for response in result.tool_responses] == ["a", "b"]


def test_provider_failure_retains_completed_events():
    agent, _ = make_agent([call(), RuntimeError("provider unavailable")])
    result = agent.invoke(AgentRequest(run_id="r", prompt="go"))
    assert len(result.tool_calls) == len(result.tool_responses) == 1
    assert result.final_answer == "The model provider failed to complete this request."


def test_duplicate_ids_are_not_executed_twice():
    agent, _ = make_agent([call(), call()])
    result = agent.invoke(AgentRequest(run_id="r", prompt="go"))
    assert len(result.tool_responses) == 1
    assert "provider failed" in result.final_answer


def test_mismatched_executor_response_is_error():
    from runtime_security.contracts import ToolResponse

    executor = Mock()
    executor.execute.return_value = ToolResponse(call_id="wrong", status="success")
    agent, _ = make_agent([call()], executor=executor)
    result = agent.invoke(AgentRequest(run_id="r", prompt="go", max_steps=1))
    assert result.tool_responses[0].call_id == "call-1"
    assert result.tool_responses[0].status == "error"
