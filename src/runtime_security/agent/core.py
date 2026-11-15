"""Bounded explicit model/execution graph with a replaceable execution boundary."""

import json
import os
from typing import Any, Protocol, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from runtime_security.contracts import AgentRequest, AgentResult, ToolCallEvent, ToolResponse
from runtime_security.gateway.decisions import Containment
from runtime_security.gateway.interceptor import enforce_call, record_rejected_calls
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
    """Mandatory boundary for registered tools and host-supplied executors."""

    def __init__(self, tools, *, recorder=None, containment=None, delegate=None):
        self.__tools = {tool.name: tool for tool in tools}
        self.__delegate = delegate
        self.recorder = recorder if recorder is not None else InMemoryRecorder()
        self.containment = containment if containment is not None else Containment()

    def execute(self, call: ToolCallEvent) -> ToolResponse:
        return enforce_call(
            call,
            self.__dispatch,
            self.recorder,
            self.containment,
            registered=call.tool_name in self.__tools,
        )

    def __dispatch(self, call):
        tool = self.__tools[call.tool_name]
        if self.__delegate is not None:
            return self.__delegate.execute(call)
        return ToolResponse(
            call_id=call.call_id, status="success", output=tool.invoke(call.arguments)
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
        *,
        containment: Containment | None = None,
    ):
        tools = list(registry.get_tools())
        if model is None:
            load_dotenv()
            model = ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            )
        self.model = model.bind_tools([convert_to_openai_tool(tool) for tool in tools])
        self._gateway = RegistryExecutor(
            tools, recorder=recorder, containment=containment, delegate=executor
        )
        self.model_name = getattr(model, "model_name", None)
        if not isinstance(self.model_name, str):
            self.model_name = None

    @property
    def recorder(self):
        return self._gateway.recorder

    @recorder.setter
    def recorder(self, value):
        self._gateway.recorder = value

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
                response = self._gateway.execute(call)
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
