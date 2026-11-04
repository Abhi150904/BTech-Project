# Shared interfaces

Import contracts from `runtime_security.contracts`, `Agent` from
`runtime_security.agent`, and `ToolRegistry` from `runtime_security.tools`.

Agent core implements synchronous `invoke(AgentRequest) -> AgentResult` and enforces
`max_steps`. The request carries the caller's run ID. Tools implements
`get_tools() -> Sequence[langchain_core.tools.BaseTool]`; each tool owns its argument
schema. Registry construction and sandbox configuration belong to that branch.

ToolCallEvent records a proposal, not proof of execution. ToolResponse references
its call ID and records success or failure. Agent core must preserve run/call IDs
and pair responses with calls. Scenario code supplies externally assessed outcomes;
agent text is never authoritative security metadata. ScenarioResult keeps actual
recorded mock side effects separate from proposed calls. Contracts validate shape,
not cross-record integrity or security policy. Telemetry persistence is not implemented.

Coordinate changes to these shared signatures across the three feature branches.
