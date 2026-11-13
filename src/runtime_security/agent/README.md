# Agent integration

`Agent` remains the existing structural protocol. Construct its implementation with
`LangGraphAgent(SandboxToolRegistry())`, then call `invoke(AgentRequest(...))`.
No shared contracts or signatures changed. The explicit StateGraph alternates
model and execution nodes. One step means one model turn and execution of all
tool calls it proposes. The last permitted turn's calls execute; no extra model
turn is made to summarize after the budget is exhausted. A final-answer-only
turn also consumes one step. Exhaustion returns `Maximum agent steps reached.`

Production construction calls `load_dotenv()` (existing environment takes priority)
and reads `OPENAI_API_KEY` and `OPENAI_MODEL`, defaulting the model to
`gpt-4o-mini`. Missing credentials/configuration errors fail at construction.
Provider invocation errors, malformed calls and duplicate call IDs terminate
with `The model provider failed to complete this request.` and retain prior
events. Empty final content returns `The model returned no answer.`
Tool errors become paired error responses and are fed back to the model, which
can recover within the remaining budget. Exceptions contain no provider details
in the user-facing provider failure answer.

All execution passes through `ToolExecutor.execute(ToolCallEvent) -> ToolResponse`.
The default `RegistryExecutor` invokes only registered tools. Supply an executor
to inspect calls before execution; no gateway or security policy is implemented.
The executor must preserve the call ID and return JSON-compatible output.
`ToolResponse` has no run ID field: its run association is via the enclosing
AgentResult and paired ToolCallEvent. Registry/tool execution is synchronous.

Tests inject a model exposing `bind_tools()` and `invoke()` and never construct
a real OpenAI client. Run from the checkout using `.venv/Scripts/python -m pytest`.
