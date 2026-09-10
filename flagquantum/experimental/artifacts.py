"""Experimental read-only views over versioned artifact envelopes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core import _artifacts
from ..core import _compilation_evidence as _evidence_v1
from ..core import _compilation_evidence_v2 as _evidence_v2
from ..core import _compilation_evidence_v3 as _evidence_v3

_MAX_JSON_BYTES = 18 * 1024 * 1024
_PROGRAM_TYPES = (
    _artifacts.ProgramArtifact,
    _artifacts.ProgramArtifactV2,
    _artifacts.ProgramArtifactV3,
)
_EVIDENCE_TYPES = (
    _evidence_v1.CompilationEvidenceBundle,
    _evidence_v2.CompilationEvidenceBundleV2,
    _evidence_v3.CompilationEvidenceBundleV3,
)


def _input_json(payload: object, owner: str) -> str:
    if type(payload) is not str:
        raise TypeError(f"{owner} must be JSON text")
    if len(payload.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{owner} exceeds maximum UTF-8 bytes {_MAX_JSON_BYTES}")
    return payload


def _bounded_output(payload: str, owner: str) -> str:
    if len(payload.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{owner} exceeds maximum UTF-8 bytes {_MAX_JSON_BYTES}")
    return payload


@dataclass(frozen=True, slots=True)
class ProgramArtifact:
    """Frozen role view over a supported Core program artifact."""

    _value: (
        _artifacts.ProgramArtifact
        | _artifacts.ProgramArtifactV2
        | _artifacts.ProgramArtifactV3
    ) = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._value, _PROGRAM_TYPES):
            raise TypeError("ProgramArtifact views require a loaded Core artifact")

    @property
    def version(self) -> str:
        return self._value.version

    @property
    def kind(self) -> str:
        return self._value.kind.value

    @property
    def producer(self) -> str:
        return self._value.producer

    @property
    def payload_sha256(self) -> str | None:
        if isinstance(self._value, _artifacts.ProgramArtifact):
            return None
        return self._value.payload_sha256

    @property
    def identity(self) -> str:
        if isinstance(self._value, _artifacts.ProgramArtifact):
            return self._value.content_hash
        return self._value.artifact_identity

    def to_dict(self) -> dict[str, Any]:
        return self._value.to_dict()

    def to_json(self) -> str:
        return dump_program_artifact(self)


@dataclass(frozen=True, slots=True)
class CompilationEvidence:
    """Frozen role view over a supported Core compilation-evidence bundle."""

    _value: (
        _evidence_v1.CompilationEvidenceBundle
        | _evidence_v2.CompilationEvidenceBundleV2
        | _evidence_v3.CompilationEvidenceBundleV3
    ) = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._value, _EVIDENCE_TYPES):
            raise TypeError("CompilationEvidence views require loaded Core evidence")

    @property
    def version(self) -> str:
        return self._value.version

    @property
    def producer(self) -> str:
        return self._value.producer

    @property
    def identity(self) -> str:
        return self._value.bundle_identity

    def to_dict(self) -> dict[str, Any]:
        return self._value.to_dict()

    def to_json(self) -> str:
        return dump_compilation_evidence(self)


def load_program_artifact(payload: str) -> ProgramArtifact:
    """Load supported artifact JSON into a version-neutral frozen role view."""

    view = ProgramArtifact(
        _artifacts.read_program_artifact_json(_input_json(payload, "artifact"))
    )
    _ = view.identity
    dump_program_artifact(view)
    return view


def dump_program_artifact(artifact: ProgramArtifact) -> str:
    """Return the loaded artifact's exact canonical Core JSON."""

    if type(artifact) is not ProgramArtifact:
        raise TypeError("artifact must be a ProgramArtifact view")
    value = artifact._value
    if isinstance(value, _artifacts.ProgramArtifact):
        payload = json.dumps(
            value.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    else:
        payload = value.to_json()
    return _bounded_output(payload, "artifact")


def load_compilation_evidence(payload: str) -> CompilationEvidence:
    """Load supported evidence JSON into a version-neutral frozen role view."""

    view = CompilationEvidence(
        _evidence_v1.read_compilation_evidence_bundle_json(
            _input_json(payload, "compilation evidence")
        )
    )
    dump_compilation_evidence(view)
    return view


def dump_compilation_evidence(evidence: CompilationEvidence) -> str:
    """Return the loaded evidence's exact canonical Core JSON."""

    if type(evidence) is not CompilationEvidence:
        raise TypeError("evidence must be a CompilationEvidence view")
    return _bounded_output(
        evidence._value.to_json(),
        "compilation evidence",
    )


__all__ = (
    "CompilationEvidence",
    "ProgramArtifact",
    "dump_compilation_evidence",
    "dump_program_artifact",
    "load_compilation_evidence",
    "load_program_artifact",
)
