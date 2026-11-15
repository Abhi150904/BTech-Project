"""Offline dataset commands."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from runtime_security.datasets.generate import generate
from runtime_security.datasets.schema import SECRET, Corpus, write_new
from runtime_security.datasets.split import split
from runtime_security.datasets.validate import audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--evidence", required=True, type=Path)
    gen.add_argument("--telemetry", required=True, type=Path)
    gen.add_argument("--source", required=True, choices=["live", "scripted"])
    gen.add_argument("--config", type=Path, default=Path("configs/dataset.toml"))
    gen.add_argument("--policy", type=Path, default=Path("configs/policy_metadata.toml"))
    gen.add_argument("--out", type=Path, required=True)
    freeze = commands.add_parser("split")
    freeze.add_argument("dataset", type=Path)
    freeze.add_argument("--seed", type=int, default=42)
    freeze.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("dataset", type=Path)
    check.add_argument("--manifest", type=Path)
    check.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            corpus = generate(args.evidence, args.telemetry, args.source, args.config, args.policy)
            write_new(args.out, corpus.model_dump())
            return 0
        raw = args.dataset.read_text(encoding="utf-8")
        if SECRET.search(raw):
            parser.exit(1, "Dataset redaction scan failed.\n")
        corpus = Corpus.model_validate_json(raw)
        if json.loads(raw) != corpus.model_dump(mode="json"):
            parser.exit(1, "Dataset contains noncanonical fields or missing metadata.\n")
        manifest = None
        if args.command == "validate" and args.manifest:
            manifest = json.loads(args.manifest.read_bytes())
        report = audit(corpus, manifest)
        if args.command == "split":
            if not report["structurally_valid"]:
                parser.exit(1, "Invalid dataset; inspect validation report.\n")
            write_new(args.out, split(corpus, args.seed))
        elif not report["redaction_scan_passed"]:
            parser.exit(1, "Redaction scan failed.\n")
        elif args.out:
            write_new(args.out, report)
        else:
            print(json.dumps(report, indent=2))
        return int(not report["structurally_valid"])
    except ValidationError:
        parser.exit(1, "Invalid artifact schema.\n")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Do not echo malformed raw field values in diagnostics.
        if isinstance(exc, ValueError) and str(exc).startswith("Need three independent"):
            parser.exit(1, str(exc) + "\n")
        parser.exit(
            1, f"Dataset operation failed ({type(exc).__name__}). Check inputs and output path.\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
