"""Immutable block, region, and module containers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .operations import Operation
from .values import ValueRef

IDENTITY_SCHEMA_VERSION = "flagquantum.quantum_ir.program.v1alpha1"


@dataclass(frozen=True)
class Block:
    arguments: tuple[ValueRef, ...] = ()
    operations: tuple[Operation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "operations", tuple(self.operations))

    def canonical(self) -> dict[str, object]:
        return {
            "arguments": [item.canonical() for item in self.arguments],
            "operations": [item.canonical() for item in self.operations],
        }


@dataclass(frozen=True)
class Region:
    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if not blocks:
            raise ValueError("region requires at least one block")
        object.__setattr__(self, "blocks", blocks)

    def canonical(self) -> dict[str, object]:
        return {"blocks": [block.canonical() for block in self.blocks]}


@dataclass(frozen=True)
class QuantumModule:
    """The private top-level program container used in Phase 1."""

    body: Region
    revision: int = 0
    identity_schema_version: str = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        revision = int(self.revision)
        if revision < 0:
            raise ValueError("module revision cannot be negative")
        if self.identity_schema_version != IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported internal identity schema version")
        object.__setattr__(self, "revision", revision)

    def canonical(self) -> dict[str, object]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "body": self.body.canonical(),
        }

    @property
    def program_identity(self) -> str:
        payload = json.dumps(
            self.canonical(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["Block", "IDENTITY_SCHEMA_VERSION", "QuantumModule", "Region"]
