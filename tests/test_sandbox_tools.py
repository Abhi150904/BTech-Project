import socket

import pytest
from pydantic import ValidationError

from runtime_security.tools import SandboxToolRegistry


@pytest.fixture
def registry():
    return SandboxToolRegistry()


def test_normal_and_search(registry):
    tools = {t.name: t for t in registry.get_tools()}
    assert "Atlas" in tools["read_document"].invoke({"document_id": "summary.txt"})
    assert "SYNTHETIC_SECRET" in tools["read_file"].invoke({"file_id": "sensitive.txt"})
    assert registry.search_files("ATLAS") == registry.search_files("atlas")
    assert len(registry.search_files("atlas")) == 2
    assert registry.search_files("no-such-keyword") == []


@pytest.mark.parametrize(
    "identifier",
    ["missing", "../summary.txt", "..\\summary.txt", "C:\\secret.txt", "/etc/passwd", ""],
)
@pytest.mark.parametrize("name,key", [("read_document", "document_id"), ("read_file", "file_id")])
def test_rejected_ids(registry, identifier, name, key):
    tools = {t.name: t for t in registry.get_tools()}
    with pytest.raises((ValueError, ValidationError)):
        tools[name].invoke({key: identifier})


@pytest.mark.parametrize(
    "arguments", [{}, {"document_id": 12}, {"document_id": "summary.txt", "extra": True}]
)
def test_invalid_schema(registry, arguments):
    with pytest.raises(ValidationError):
        registry.get_tools()[0].invoke(arguments)


def test_mock_post_never_networks(registry, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network attempted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    post = {t.name: t for t in registry.get_tools()}["mock_http_post"]
    result = post.invoke({"destination": "https://collector.invalid", "payload": "synthetic"})
    assert registry.recorded_requests == [result]
    result["payload"] = "changed"
    assert registry.recorded_requests[0]["payload"] == "synthetic"
    assert SandboxToolRegistry().recorded_requests == []


def test_missing_fixture_and_blank_query(registry, tmp_path):
    with pytest.raises(FileNotFoundError):
        SandboxToolRegistry(tmp_path).read_document("summary.txt")
    with pytest.raises(ValueError):
        registry.search_files("   ")


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_file", {"file_id": 12}),
        ("search_files", {"query": ""}),
        ("search_files", {"query": ["atlas"]}),
        ("mock_http_post", {"destination": "", "payload": "x"}),
        ("mock_http_post", {"destination": "x", "payload": 12}),
        ("mock_http_post", {"destination": "x"}),
    ],
)
def test_other_invalid_schemas(registry, name, args):
    tools = {tool.name: tool for tool in registry.get_tools()}
    with pytest.raises(ValidationError):
        tools[name].invoke(args)
    assert not registry.recorded_requests
