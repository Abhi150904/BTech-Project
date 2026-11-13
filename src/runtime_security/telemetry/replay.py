"""Read-only replay of sanitized event history; never executes tools."""

import argparse
from pathlib import Path

from runtime_security.telemetry.recorder import read_events


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("run_id")
    args = parser.parse_args(argv)
    try:
        events = [e for e in read_events(args.path) if e.run_id == args.run_id]
    except (OSError, ValueError):
        parser.exit(1, "Cannot replay malformed or unreadable telemetry.\n")
    if not events:
        parser.exit(1, "Run ID not found.\n")
    for event in events:
        print(event.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
