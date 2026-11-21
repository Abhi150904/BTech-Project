import hashlib
import json
import runpy
import zipfile
from pathlib import Path

import pytest

from runtime_security.assets import RESOURCE_FILES, RESOURCE_ROOT, resource_path
from runtime_security.gateway.decisions import CHECKPOINT, CHECKPOINT_VERSION
from runtime_security.policies.engine import DEFAULT_POLICY
from runtime_security.tools import SandboxToolRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_resource_defaults_ignore_working_directory(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/policy_metadata.toml").write_text("untrusted configuration")
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_POLICY == resource_path("configs/policy_metadata.toml")
    assert CHECKPOINT == resource_path("research/phase7-scripted-v1/enriched.json")
    registry = SandboxToolRegistry()
    assert registry.root == RESOURCE_ROOT / "sandbox/synthetic"
    assert "Atlas" in registry.read_document("summary.txt")
    assert all(resource_path(name).is_file() for name in RESOURCE_FILES)
    with pytest.raises(ValueError):
        resource_path("../.env")


def test_frozen_policy_bytes_and_checkpoint_identity():
    from runtime_security.datasets.schema import fingerprint

    checkpoint = json.loads(CHECKPOINT.read_bytes())
    assert hashlib.sha256(DEFAULT_POLICY.read_bytes()).hexdigest() == checkpoint["policy_hash"]
    assert fingerprint(checkpoint) == CHECKPOINT_VERSION


@pytest.mark.parametrize(
    "unwanted",
    [
        ".env",
        "runtime_security/.env.local",
        "presentation/deck.pptx",
        ".aws/credentials",
        "runtime_security/stale.py",
    ],
)
def test_archive_inspector_rejects_local_or_stale_files(tmp_path, unwanted):
    inspect = runpy.run_path(str(ROOT / "scripts/verify_distribution.py"))["inspect"]
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for source in (ROOT / "src/runtime_security").rglob("*.py"):
            archive.write(source, source.relative_to(ROOT / "src").as_posix())
        for relative in RESOURCE_FILES:
            archive.write(resource_path(relative), "runtime_security/_data/" + relative)
    assert inspect(wheel)["resources"] == len(RESOURCE_FILES)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(unwanted, "local-only sentinel")
    with pytest.raises(ValueError):
        inspect(wheel)


def test_offline_demo_uses_mock_effect_evidence(monkeypatch, capsys):
    import socket

    def forbidden(*args, **kwargs):
        pytest.fail("Network call in offline handoff demo")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    main = runpy.run_path(str(ROOT / "scripts/offline_demo.py"))["main"]
    assert main(["--unprotected-only"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["benign_task_completed"]
    assert rows[1]["attack_completed"] and rows[1]["recorded_mock_requests"] == 1
    assert "payload" not in str(rows) and "SYNTHETIC_SECRET" not in str(rows)


def test_empty_environment_template_has_no_key():
    values = dict(line.split("=", 1) for line in (ROOT / ".env.example").read_text().splitlines())
    assert values["OPENAI_API_KEY"] == ""
