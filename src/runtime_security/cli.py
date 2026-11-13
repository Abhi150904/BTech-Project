"""Command-line entry point for a sandboxed agent run."""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from runtime_security.agent import LangGraphAgent
from runtime_security.contracts import AgentRequest
from runtime_security.telemetry.recorder import JsonlRecorder
from runtime_security.tools import SandboxToolRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the runtime security research agent.")
    parser.add_argument("prompt", help="Task to give the agent")
    parser.add_argument("--telemetry", type=Path, help="Append sanitized audit events to JSONL")
    parser.add_argument(
        "--max-steps", type=int, default=10, help="Maximum model/tool cycles (default: 10)"
    )
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    try:
        options = {"recorder": JsonlRecorder(args.telemetry)} if args.telemetry else {}
        result = LangGraphAgent(SandboxToolRegistry(), **options).invoke(
            AgentRequest(run_id=str(uuid4()), prompt=args.prompt, max_steps=args.max_steps)
        )
    except Exception as exc:
        print(f"Could not start agent: {exc}", file=sys.stderr)
        return 1

    print(result.final_answer)
    if args.telemetry:
        print(f"Telemetry run ID: {result.run_id}")
    if result.tool_calls:
        print("\nTool calls:")
        responses = {response.call_id: response for response in result.tool_responses}
        for call in result.tool_calls:
            response = responses.get(call.call_id)
            status = response.status if response else "no response"
            print(f"- {call.tool_name} [{status}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
