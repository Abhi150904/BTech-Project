"""Bounded explicit model/execution graph with a replaceable execution boundary."""

import json
import os
from typing import Any, Protocol, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from runtime_security.contracts import AgentRequest, AgentResult, ToolCallEvent, ToolResponse
from runtime_security.gateway.interceptor import observe_call, record_rejected_calls
from runtime_security.telemetry.events import RunMetadata
from runtime_security.telemetry.recorder import (
    InMemoryRecorder,
    TelemetryError,
    completion_status,
    current_metadata,
)
from runtime_security.tools import ToolRegistry


class ToolExecutor(Protocol):
    def execute(self, call: ToolCallEvent) -> ToolResponse: ...


class RegistryExecutor:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    def execute(self, call: ToolCallEvent) -> ToolResponse:
        try:
            output = self.tools[call.tool_name].invoke(call.arguments)
            return ToolResponse(call_id=call.call_id, status="success", output=output)
        except Exception as exc:
            return ToolResponse(
                call_id=call.call_id, status="error", error=str(exc) or type(exc).__name__
            )


class State(TypedDict):
    messages: list[Any]
    steps: int
    calls: list[ToolCallEvent]
    responses: list[ToolResponse]
    pending: list[ToolCallEvent]
    answer: str


class LangGraphAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        model=None,
        executor: ToolExecutor | None = None,
        recorder=None,
    ):
        tools = list(registry.get_tools())
        if model is None:
            load_dotenv()
            model = ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            )
        self.model = model.bind_tools(tools)
        self.executor = executor or RegistryExecutor(tools)
        self.recorder = recorder if recorder is not None else InMemoryRecorder()
        self.model_name = getattr(model, "model_name", None)
        if not isinstance(self.model_name, str):
            self.model_name = None

    def invoke(self, request: AgentRequest) -> AgentResult:
        metadata = current_metadata()
        if metadata.get("model_name") is None:
            metadata["model_name"] = self.model_name
        self.recorder.start_run(
            request.run_id, RunMetadata(max_steps=request.max_steps, **metadata)
        )

        def model_node(state: State):
            try:
                message = self.model.invoke(state["messages"])
                raw_calls = [*message.tool_calls, *message.invalid_tool_calls]
                if message.invalid_tool_calls:
                    record_rejected_calls(
                        request.run_id, raw_calls, self.recorder, "malformed_tool_call"
                    )
                    raise ValueError("Provider supplied malformed tool calls")
                try:
                    calls = [
                        ToolCallEvent(
                            run_id=request.run_id,
                            call_id=c["id"],
                            tool_name=c["name"],
                            arguments=c["args"],
                        )
                        for c in message.tool_calls
                    ]
                except Exception:
                    record_rejected_calls(
                        request.run_id, raw_calls, self.recorder, "invalid_tool_call_contract"
                    )
                    raise
                known_ids = {call.call_id for call in state["calls"]}
                new_ids = [call.call_id for call in calls]
                if len(set(new_ids)) != len(new_ids) or known_ids.intersection(new_ids):
                    record_rejected_calls(
                        request.run_id, raw_calls, self.recorder, "duplicate_call_id"
                    )
                    raise ValueError("Provider supplied duplicate call IDs")
                return {
                    "messages": [*state["messages"], message],
                    "steps": state["steps"] + 1,
                    "pending": calls,
                    "calls": [*state["calls"], *calls],
                    "answer": (message.text or "The model returned no answer.")
                    if not calls
                    else "",
                }
            except TelemetryError:
                raise
            except Exception:
                return {
                    "answer": "The model provider failed to complete this request.",
                    "pending": [],
                    "steps": state["steps"] + 1,
                }

        def execute_node(state: State):
            responses = []
            messages = []
            for call in state["pending"]:
                response = observe_call(call, self.executor.execute, self.recorder)
                responses.append(response)
                messages.append(
                    ToolMessage(
                        tool_call_id=call.call_id, content=json.dumps(response.model_dump())
                    )
                )
            return {
                "responses": [*state["responses"], *responses],
                "messages": [*state["messages"], *messages],
                "answer": "Maximum agent steps reached.",
            }

        graph = StateGraph(State)
        graph.add_node("model", model_node)
        graph.add_node("execute", execute_node)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", lambda s: "execute" if s["pending"] else END)
        graph.add_conditional_edges(
            "execute", lambda s: "model" if s["steps"] < request.max_steps else END
        )
        try:
            result = graph.compile().invoke(
                State(
                    messages=[HumanMessage(content=request.prompt)],
                    steps=0,
                    calls=[],
                    responses=[],
                    pending=[],
                    answer="",
                ),
                {"recursion_limit": request.max_steps * 2 + 2},
            )
        except TelemetryError:
            raise
        except Exception:
            self.recorder.record(
                request.run_id, "run_complete", summary={"status": "execution_error"}
            )
            raise
        self.recorder.record(
            request.run_id,
            "run_complete",
            summary={"status": completion_status(result["answer"]), "steps": result["steps"]},
        )
        return AgentResult(
            run_id=request.run_id,
            final_answer=result["answer"],
            tool_calls=result["calls"],
            tool_responses=result["responses"],
        )
