"""Reviewed runtime resources for editable checkouts and installed wheels.

Paths are relative to this module, never to the caller's working directory.
The build copies this explicit allowlist without changing resource bytes.
"""

from pathlib import Path

RESOURCE_FILES = (
    "sandbox/synthetic/documents/summary.txt",
    "sandbox/synthetic/documents/injection.txt",
    "sandbox/synthetic/files/public.txt",
    "sandbox/synthetic/files/sensitive.txt",
    "configs/policy_metadata.toml",
    "configs/dataset.toml",
    "configs/model.toml",
    "configs/experiments/phase9.toml",
    "research/phase7-scripted-v1/enriched.json",
)

_PACKAGE = Path(__file__).resolve().parent
_CHECKOUT = _PACKAGE.parent.parent
RESOURCE_ROOT = (
    _CHECKOUT
    if _PACKAGE.parent.name == "src" and (_CHECKOUT / "pyproject.toml").is_file()
    else _PACKAGE / "_data"
)


def resource_path(relative: str) -> Path:
    if relative not in RESOURCE_FILES:
        raise ValueError("Unknown packaged resource")
    return RESOURCE_ROOT / relative
