"""Verify a wheel installation from outside the source tree; no live model calls."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import runtime_security


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Extracted sdist or checkout")
    parser.add_argument(
        "--full", action="store_true", help="Verify ML commands and real checkpoint"
    )
    args = parser.parse_args(argv)
    if not Path(runtime_security.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        parser.error("Expected an installed wheel, not an editable/source import")
    if Path.cwd().resolve().is_relative_to(args.inputs.resolve()):
        parser.error("Run outside the input source tree")
    scripts = Path(sys.executable).parent
    suffix = ".exe" if os.name == "nt" else ""
    optional = {"runtime-security-baselines", "runtime-security-train", "runtime-security-score"}
    passed, skipped = [], []
    env = {k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "PYTHONPATH"}}
    env.update(LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")

    def run(name, *arguments):
        result = subprocess.run(
            [str(scripts / (name + suffix)), *map(str, arguments)],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Smoke command failed: {name} (exit {result.returncode})")
        passed.append(name + (" --help" if arguments == ("--help",) else " offline"))

    for entry in importlib.metadata.distribution("btech-runtime-security").entry_points:
        if entry.group != "console_scripts":
            continue
        if entry.name in optional and not args.full:
            skipped.append(entry.name + ": optional extra omitted from core check")
            continue
        run(entry.name, *(() if entry.name == "runtime-security-scenarios" else ("--help",)))
    root = args.inputs.resolve()
    dataset = root / "research/scripted-v1"
    run(
        "runtime-security-dataset",
        "validate",
        dataset / "dataset.json",
        "--manifest",
        dataset / "manifest.json",
    )
    run("runtime-security-dashboard", root / "research/phase9-scripted-v1/report.json", "--check")
    with tempfile.TemporaryDirectory(prefix="report-check-") as directory:
        report = Path(directory) / "report.json"
        run(
            "runtime-security-evaluate",
            dataset / "dataset.json",
            dataset / "manifest.json",
            "--baselines",
            dataset / "baselines.json",
            "--neural",
            root / "research/phase7-scripted-v1/report.json",
            "--config",
            root / "configs/experiments/phase9.toml",
            "--out",
            report,
        )
        run("runtime-security-dashboard", report, "--check")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/offline_demo.py"),
            *([] if args.full else ["--unprotected-only"]),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode:
        raise RuntimeError("Offline demo failed in installed environment")
    print(
        json.dumps(
            {
                "installed_from_environment": True,
                "passed": passed,
                "skipped": skipped,
                "demo": json.loads(result.stdout),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
