"""Internal candidate envelope for versioned multi-stage program artifacts.

`CircuitIR` remains the canonical public circuit representation.  This module
adds a typed envelope used to prove that richer future compilation products can
coexist without changing the current stable API.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

ARTIFACT_ENVELOPE_VERSION = "1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ArtifactKind(str, Enum):
    SOURCE = "source"
    CIRCUIT = "circuit"
    LOGICAL = "logical"
    PHYSICAL = "physical"
    PULSE = "pulse"
    NETWORK = "network"
    SIMULATION_PLAN = "simulation_plan"
    EXECUTABLE = "executable"


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    raise TypeError(f"artifact value {type(value).__name__} is not serializable")


def _freeze(value: Any) -> Any:
    normalized = _json_value(value)
    if isinstance(normalized, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in normalized.items()}
        )
    if isinstance(normalized, list):
        return tuple(_freeze(item) for item in normalized)
    return normalized


@dataclass(frozen=True)
class ProgramArtifact:
    """Content-addressed envelope around one compilation-stage payload."""

    kind: ArtifactKind
    payload: Mapping[str, Any]
    producer: str
    version: str = ARTIFACT_ENVELOPE_VERSION
    required_capabilities: tuple[str, ...] = ()
    parent_hashes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version != ARTIFACT_ENVELOPE_VERSION:
            raise ValueError(f"unsupported artifact envelope version {self.version!r}")
        if not self.producer.strip():
            raise ValueError("artifact producer must be non-empty")
        required = tuple(sorted(set(self.required_capabilities)))
        parents = tuple(self.parent_hashes)
        if any(not item for item in required):
            raise ValueError("required capability names must be non-empty")
        if any(_SHA256.fullmatch(item) is None for item in parents):
            raise ValueError("parent hashes must be SHA-256 hex digests")
        object.__setattr__(self, "required_capabilities", required)
        object.__setattr__(self, "parent_hashes", parents)
        object.__setattr__(self, "payload", _freeze(self.payload))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "flagquantum.program_artifact",
            "version": self.version,
            "kind": self.kind.value,
            "producer": self.producer,
            "required_capabilities": list(self.required_capabilities),
            "parent_hashes": list(self.parent_hashes),
            "payload": _json_value(self.payload),
            "metadata": _json_value(self.metadata),
        }

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramArtifact":
        if not isinstance(payload, Mapping):
            raise TypeError("program artifact payload must be a mapping")
        expected = {
            "schema",
            "version",
            "kind",
            "producer",
            "required_capabilities",
            "parent_hashes",
            "payload",
            "metadata",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown:
            raise ValueError(
                "unknown program artifact field(s): " + ", ".join(sorted(unknown))
            )
        if missing:
            raise ValueError(
                "missing program artifact field(s): " + ", ".join(sorted(missing))
            )
        if payload["schema"] != "flagquantum.program_artifact":
            raise ValueError("invalid program artifact schema")
        return cls(
            version=str(payload["version"]),
            kind=ArtifactKind(payload["kind"]),
            producer=str(payload["producer"]),
            required_capabilities=tuple(payload["required_capabilities"]),
            parent_hashes=tuple(payload["parent_hashes"]),
            payload=payload["payload"],
            metadata=payload["metadata"],
        )

    @classmethod
    def from_circuit_ir(cls, circuit_ir: Any, *, producer: str) -> "ProgramArtifact":
        to_dict = getattr(circuit_ir, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("circuit_ir must provide to_dict()")
        return cls(kind=ArtifactKind.CIRCUIT, payload=to_dict(), producer=producer)


__all__ = ("ARTIFACT_ENVELOPE_VERSION", "ArtifactKind", "ProgramArtifact")
