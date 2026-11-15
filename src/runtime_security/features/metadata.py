"""Trusted, version-specific sandbox authorization facts."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from runtime_security.contracts import Contract

Sensitivity = Literal["public", "synthetic_sensitive", "unknown"]


class Grants(Contract):
    read_document: set[str] = Field(default_factory=set)
    read_file: set[str] = Field(default_factory=set)
    mock_http_post: set[str] = Field(default_factory=set)
    search_files: bool = False


class Policy(Contract):
    documents: dict[str, Sensitivity]
    files: dict[str, Sensitivity]
    destinations: dict[str, Literal["approved", "unapproved", "unknown"]]
    tasks: dict[str, dict[str, Grants]]

    @model_validator(mode="after")
    def validate_grants(self):
        for versions in self.tasks.values():
            for grants in versions.values():
                for targets, catalog in (
                    (grants.read_document, self.documents),
                    (grants.read_file, self.files),
                    (grants.mock_http_post, self.destinations),
                ):
                    if targets.difference(catalog):
                        raise ValueError("Authorization references an unmapped target")
        return self


def load_policy(path: Path) -> Policy:
    with Path(path).open("rb") as stream:
        return Policy.model_validate(tomllib.load(stream))
