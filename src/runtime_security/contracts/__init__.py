"""Shared data contracts; observed outcomes are supplied by the caller."""
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCallEvent(Contract):
    """A proposed call; its presence does not imply execution."""

    run_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ToolResponse(Contract):
    call_id: str = Field(min_length=1)
    status: Literal["success", "error"]
    output: JsonValue = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "error" and not self.error:
            raise ValueError("Error responses require an error message")
        if self.status == "success" and self.error is not None:
            raise ValueError("Successful responses cannot contain an error")
        return self


class AgentRequest(Contract):
    run_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    max_steps: int = Field(default=10, gt=0)


class AgentResult(Contract):
    run_id: str = Field(min_length=1)
    final_answer: str
    tool_calls: list[ToolCallEvent] = Field(default_factory=list)
    tool_responses: list[ToolResponse] = Field(default_factory=list)


class ScenarioResult(Contract):
    scenario_id: str = Field(min_length=1)
    scenario_version: str = "1.0"
    outcome: Literal["passed", "failed", "error"]
    max_steps: int | None = Field(default=None, gt=0)
    model_name: str | None = None
    injection_present: bool = False
    attack_attempted: bool = False
    attack_completed: bool = False
    benign_task_completed: bool = False
    agent_result: AgentResult | None = None
    recorded_side_effects: list[JsonValue] = Field(default_factory=list)
    notes: str = ""
