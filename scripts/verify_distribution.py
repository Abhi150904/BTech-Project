"""Inspect local archives for resource completeness and unwanted files."""

import argparse
import hashlib
import json
import runpy
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = runpy.run_path(str(ROOT / "src/runtime_security/assets.py"))["RESOURCE_FILES"]


def inspect(path):
    wheel = path.suffix == ".whl"
    if wheel:
        with zipfile.ZipFile(path) as archive:
            contents = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    else:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            if any(m.issym() or m.islnk() for m in members):
                raise ValueError("Unexpected archive link")
            contents = {
                m.name.split("/", 1)[1]: archive.extractfile(m).read()
                for m in members
                if m.isfile() and "/" in m.name
            }
    for name in contents:
        parts = PurePosixPath(name).parts
        if (
            PurePosixPath(name).is_absolute()
            or ".." in parts
            or any(
                p
                in {
                    ".venv",
                    ".git",
                    ".codex",
                    ".aws",
                    ".agents",
                    "__pycache__",
                    "presentation",
                    "artifacts",
                    ".pytest_cache",
                    ".ruff_cache",
                    ".idea",
                    ".vscode",
                }
                or p.startswith(".env")
                and p != ".env.example"
                for p in parts
            )
            or name.endswith((".pyc", ".pem", ".key"))
        ):
            raise ValueError("Forbidden local file in distribution")
    prefix = "runtime_security/_data/" if wheel else ""
    for relative in RESOURCES:
        if contents.get(prefix + relative) != (ROOT / relative).read_bytes():
            raise ValueError("Resource missing or changed in distribution")
    if wheel:
        allowed_data = {"runtime_security/_data/" + p for p in RESOURCES}
        modules = {
            p.relative_to(ROOT / "src").as_posix(): p.read_bytes()
            for p in (ROOT / "src/runtime_security").rglob("*.py")
        }
        if any(contents.get(name) != data for name, data in modules.items()):
            raise ValueError("Package module missing or changed")
        for name in contents:
            if ".dist-info/" not in name and name not in modules and name not in allowed_data:
                raise ValueError("Unexpected wheel content")
    else:
        if "docs/viva-guide-hinglish.md" in contents:
            raise ValueError("Unrelated local notes included")
        for required in (
            "pyproject.toml",
            "setup.py",
            "scripts/offline_demo.py",
            "docs/phase-11-handoff.md",
            "tests/test_release_readiness.py",
        ):
            if required not in contents:
                raise ValueError("Source handoff content missing")
    policy = contents[prefix + "configs/policy_metadata.toml"]
    checkpoint = json.loads(contents[prefix + "research/phase7-scripted-v1/enriched.json"])
    if hashlib.sha256(policy).hexdigest() != checkpoint["policy_hash"]:
        raise ValueError("Checkpoint policy bytes differ; inspect Git line-ending attributes")
    return {
        "archive": path.name,
        "files": len(contents),
        "resources": len(RESOURCES),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args(argv)
    for path in args.archives:
        print(json.dumps(inspect(path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
