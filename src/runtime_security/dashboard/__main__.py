"""Launch the offline Phase 9 report viewer."""

import argparse
import sys

from runtime_security.dashboard.inspection import ReportError, load_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", nargs="?", help="Local phase9-v1 JSON report; omit for file picker"
    )
    parser.add_argument("--check", action="store_true", help="Validate without opening a window")
    args = parser.parse_args(argv)
    if args.check:
        if not args.report:
            parser.error("--check requires a local report file")
        try:
            loaded = load_report(args.report)
        except ReportError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"Valid phase9-v1 report | source={loaded.report.source} "
            f"| runs={len(loaded.report.runs)}"
        )
        return 0
    try:
        import tkinter as tk

        from runtime_security.dashboard.app import Dashboard

    except ImportError:
        parser.exit(1, "Tkinter unavailable. Use a desktop Python with Tk support.\n")
    try:
        root = tk.Tk()
    except tk.TclError:
        parser.exit(1, "Display unavailable. Launch from a desktop session or use --check.\n")
    Dashboard(root, args.report)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
