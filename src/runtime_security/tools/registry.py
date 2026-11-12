"""Deterministic synthetic tools. No network clients are used here."""

from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DocumentInput(Input):
    document_id: str = Field(min_length=1)


class FileInput(Input):
    file_id: str = Field(min_length=1)


class SearchInput(Input):
    query: str = Field(min_length=1)


class PostInput(Input):
    destination: str = Field(min_length=1)
    payload: str


class SandboxToolRegistry:
    """Construct per run to isolate the in-memory mock request ledger."""

    def __init__(self, root: Path | None = None):
        self.root = (
            root or Path(__file__).resolve().parents[3] / "sandbox" / "synthetic"
        ).resolve()
        self.recorded_requests: list[dict[str, str]] = []

    def _read(self, category: str, identifier: str, allowed: set[str]) -> str:
        if identifier not in allowed:
            raise ValueError("Unknown or invalid sandbox ID")
        directory = (self.root / category).resolve()
        path = (directory / identifier).resolve()
        if not directory.is_relative_to(self.root) or not path.is_relative_to(directory):
            raise ValueError("Sandbox path escape")
        return path.read_text(encoding="utf-8")

    def read_document(self, document_id: str) -> str:
        return self._read("documents", document_id, {"summary.txt", "injection.txt"})

    def read_file(self, file_id: str) -> str:
        return self._read("files", file_id, {"public.txt", "sensitive.txt"})

    def search_files(self, query: str) -> list[dict[str, str]]:
        if not query.strip():
            raise ValueError("Query must contain a keyword")
        return [
            {"document_id": name, "content": content}
            for name in ("injection.txt", "summary.txt")
            if query.casefold() in (content := self.read_document(name)).casefold()
        ]

    def mock_http_post(self, destination: str, payload: str) -> dict[str, str]:
        record = {"destination": destination, "payload": payload}
        self.recorded_requests.append(record.copy())
        return record.copy()

    def get_tools(self):
        return [
            StructuredTool.from_function(fn, name=name, description=description, args_schema=schema)
            for name, fn, schema, description in (
                (
                    "read_document",
                    self.read_document,
                    DocumentInput,
                    "Read an allowlisted document.",
                ),
                ("read_file", self.read_file, FileInput, "Read an allowlisted synthetic file."),
                ("search_files", self.search_files, SearchInput, "Search synthetic documents."),
                ("mock_http_post", self.mock_http_post, PostInput, "Record a local mock POST."),
            )
        ]
