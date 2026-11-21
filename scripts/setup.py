"""One-command local setup; installation may download dependencies, demos do not."""

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true", help="Also install baseline and neural extras"
    )
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    root = Path(__file__).resolve().parents[1]
    env = root / ".venv"
    if env.exists() and not (env / "pyvenv.cfg").is_file():
        parser.error("Existing .venv is not a virtual environment; it was not modified")
    if not env.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    extras = "dev,build,baselines,neural" if args.full else "dev,build"
    command = [str(python), "-m", "pip", "install", "-e", f"{root}[{extras}]"]
    if os.name == "nt" and sys.version_info[:2] == (3, 12):
        command += ["-c", str(root / "requirements/windows-py312-tested.txt")]
    subprocess.run(command, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    print("Setup complete. No API key is needed for offline workflows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
