"""Internal candidate envelope for versioned multi-stage program artifacts.

`CircuitIR` remains the canonical public circuit representation.  This module
adds a typed envelope used to prove that richer future compilation products can
coexist without changing the current stable API.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

ARTIFACT_ENVELOPE_VERSION = "1.0"
PROGRAM_ARTIFACT_V2_VERSION = "2.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")

_V2_MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
_V2_MAX_ENVELOPE_BYTES = 18 * 1024 * 1024
_V2_MAX_DEPTH = 8
_V2_MAX_ENTRIES = 4096
_V2_MAX_STRING_BYTES = 4096
_V2_FIELDS = {
    "schema",
    "version",
    "kind",
    "producer",
    "profile",
    "payload",
    "payload_sha256",
    "circuit_content_hash",
    "requirements",
    "target",
    "compilation",
    "parameter_schema",
    "result_schema",
    "artifact_identity",
}
_V2_PROFILES = {
    "circuit-ir-1.0": {
        "name": "circuit-ir-1.0",
        "media_type": "application/vnd.flagquantum.circuit-ir+json;version=1.0",
        "encoding": "canonical-json",
    },
    "openqasm-2.0": {
        "name": "openqasm-2.0",
        "media_type": "text/x-openqasm;version=2.0;charset=utf-8",
        "encoding": "utf-8",
    },
    "openqasm-3.0": {
        "name": "openqasm-3.0",
        "media_type": "text/x-openqasm;version=3.0;charset=utf-8",
        "encoding": "utf-8",
    },
    "qcis-1.0": {
        "name": "qcis-1.0",
        "media_type": "text/x-qcis;version=1.0;charset=utf-8",
        "encoding": "utf-8",
    },
}
_V2_SENSITIVE_KEY_FRAGMENTS = (
    "account",
    "callback",
    "credential",
    "handle",
    "job_id",
    "path",
    "private_key",
    "queue",
    "secret",
    "tenant",
    "token",
    "url",
)


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_v2_value(
    value: Any,
    *,
    owner: str,
    depth: int = 0,
    counter: list[int] | None = None,
    limit_strings: bool = True,
) -> Any:
    if depth > _V2_MAX_DEPTH:
        raise ValueError(f"{owner} exceeds maximum nesting depth {_V2_MAX_DEPTH}")
    if counter is None:
        counter = [0]
    if isinstance(value, Mapping):
        counter[0] += len(value)
        if counter[0] > _V2_MAX_ENTRIES:
            raise ValueError(f"{owner} exceeds maximum entry count {_V2_MAX_ENTRIES}")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{owner} keys must be non-empty strings")
            normalized[key] = _strict_v2_value(
                item,
                owner=owner,
                depth=depth + 1,
                counter=counter,
                limit_strings=limit_strings,
            )
        return MappingProxyType(normalized)
    if isinstance(value, (tuple, list)):
        counter[0] += len(value)
        if counter[0] > _V2_MAX_ENTRIES:
            raise ValueError(f"{owner} exceeds maximum entry count {_V2_MAX_ENTRIES}")
        return tuple(
            _strict_v2_value(
                item,
                owner=owner,
                depth=depth + 1,
                counter=counter,
                limit_strings=limit_strings,
            )
            for item in value
        )
    if value is None or type(value) in {bool, int, str}:
        if (
            limit_strings
            and isinstance(value, str)
            and len(value.encode("utf-8")) > _V2_MAX_STRING_BYTES
        ):
            raise ValueError(
                f"{owner} string exceeds maximum UTF-8 bytes {_V2_MAX_STRING_BYTES}"
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{owner} cannot contain non-finite numbers")
        return value
    raise TypeError(f"{owner} contains unsupported value type {type(value).__name__}")


def _strict_fields(
    value: Any,
    *,
    expected: set[str],
    owner: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a mapping")
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(f"unknown {owner} field(s): " + ", ".join(sorted(unknown)))
    if missing:
        raise ValueError(f"missing {owner} field(s): " + ", ".join(sorted(missing)))
    return dict(value)


def _thaw_v2(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_v2(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_v2(item) for item in value]
    return value


def _require_sha256(value: Any, *, owner: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{owner} must be a lowercase SHA-256 hex digest")
    return value


def _reject_sensitive_v2(value: Any, *, owner: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if any(fragment in normalized for fragment in _V2_SENSITIVE_KEY_FRAGMENTS):
                raise ValueError(f"{owner} contains prohibited field {key!r}")
            _reject_sensitive_v2(item, owner=owner)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_sensitive_v2(item, owner=owner)
    elif isinstance(value, str) and (
        value.startswith(("/", "~/", "file://", "http://", "https://"))
    ):
        raise ValueError(f"{owner} contains a prohibited locator")


def _profile(value: Any) -> Mapping[str, Any]:
    profile = _strict_fields(
        value,
        expected={"name", "media_type", "encoding"},
        owner="artifact profile",
    )
    if any(type(profile[key]) is not str for key in profile):
        raise TypeError("artifact profile values must be strings")
    expected = _V2_PROFILES.get(profile["name"])
    if expected is None or profile != expected:
        raise ValueError("unsupported or inconsistent artifact profile")
    return _strict_v2_value(profile, owner="artifact profile")


def _parameter_names(circuit_ir: Any) -> tuple[str, ...]:
    from .parameters import parameter_names_in_value

    names: set[str] = set()
    for instruction in circuit_ir.instructions:
        names.update(parameter_names_in_value(instruction.params))
        names.update(parameter_names_in_value(instruction.matrix))
    for observable in circuit_ir.observables:
        names.update(parameter_names_in_value(observable.coefficient))
    return tuple(sorted(names))


@dataclass(frozen=True)
class ProgramArtifactV2:
    """Approved v2 artifact envelope for circuits and static target text."""

    kind: ArtifactKind
    producer: str
    profile: Mapping[str, Any]
    payload: Mapping[str, Any] | str
    circuit_content_hash: str
    requirements: Any | None
    target: Mapping[str, Any] | None
    compilation: Mapping[str, Any] | None
    parameter_schema: Mapping[str, Any]
    result_schema: Mapping[str, Any] | None
    payload_sha256: str = ""
    artifact_identity: str = ""
    version: str = PROGRAM_ARTIFACT_V2_VERSION

    def __post_init__(self) -> None:
        from .ir import CircuitIR
        from .target_capabilities import RequirementSet

        if self.version != PROGRAM_ARTIFACT_V2_VERSION:
            raise ValueError(f"unsupported artifact envelope version {self.version!r}")
        if not isinstance(self.kind, ArtifactKind):
            raise TypeError("v2 artifact kind must be an ArtifactKind")
        if type(self.producer) is not str or not self.producer.strip():
            raise ValueError("v2 artifact producer must be a non-empty string")
        if len(self.producer.encode("utf-8")) > _V2_MAX_STRING_BYTES:
            raise ValueError("v2 artifact producer exceeds maximum string bytes")

        profile = _profile(self.profile)
        profile_name = str(profile["name"])
        is_circuit = profile_name == "circuit-ir-1.0"
        expected_kind = ArtifactKind.CIRCUIT if is_circuit else ArtifactKind.EXECUTABLE
        if self.kind is not expected_kind:
            raise ValueError(
                f"profile {profile_name!r} requires kind {expected_kind.value!r}"
            )

        circuit = None
        if is_circuit:
            if not isinstance(self.payload, Mapping):
                raise TypeError("circuit-ir-1.0 payload must be a mapping")
            payload = _strict_v2_value(
                self.payload,
                owner="circuit payload",
                limit_strings=False,
            )
            circuit = CircuitIR.from_dict(_thaw_v2(payload))
            if circuit.to_dict() != _thaw_v2(payload):
                raise ValueError("circuit-ir-1.0 payload is not canonical CircuitIR")
            payload_bytes = _canonical_json_bytes(_thaw_v2(payload))
        else:
            if type(self.payload) is not str or not self.payload:
                raise TypeError("executable artifact payload must be non-empty text")
            payload_bytes = self.payload.encode("utf-8")
            payload = self.payload
        if len(payload_bytes) > _V2_MAX_PAYLOAD_BYTES:
            raise ValueError("artifact payload exceeds maximum UTF-8 bytes")
        payload_digest = hashlib.sha256(payload_bytes).hexdigest()
        if self.payload_sha256 and self.payload_sha256 != payload_digest:
            raise ValueError("payload_sha256 does not match artifact payload")

        circuit_hash = _require_sha256(
            self.circuit_content_hash,
            owner="circuit_content_hash",
        )
        if circuit is not None and circuit_hash != circuit.content_hash:
            raise ValueError("circuit_content_hash does not match CircuitIR payload")

        parameter_schema = _strict_v2_value(
            self.parameter_schema,
            owner="parameter schema",
        )
        parameter_values = _strict_fields(
            _thaw_v2(parameter_schema),
            expected={"binding", "parameters"},
            owner="parameter schema",
        )
        binding = parameter_values["binding"]
        parameters = parameter_values["parameters"]
        if binding not in {"fully_bound", "symbolic"}:
            raise ValueError("parameter binding must be fully_bound or symbolic")
        if not isinstance(parameters, list) or any(
            type(name) is not str or not name for name in parameters
        ):
            raise TypeError("parameter schema parameters must be non-empty strings")
        if parameters != sorted(set(parameters)):
            raise ValueError("parameter schema parameters must be sorted and unique")

        if is_circuit:
            assert circuit is not None
            names = list(_parameter_names(circuit))
            expected_binding = "symbolic" if names else "fully_bound"
            if parameters != names or binding != expected_binding:
                raise ValueError("parameter schema does not match CircuitIR payload")
            if any(
                item is not None
                for item in (
                    self.requirements,
                    self.target,
                    self.compilation,
                    self.result_schema,
                )
            ):
                raise ValueError(
                    "circuit-ir-1.0 requires null requirements, target, compilation, "
                    "and result_schema"
                )
            requirements = target = compilation = result_schema = None
        else:
            if binding != "fully_bound" or parameters:
                raise ValueError("executable artifacts must be fully bound")
            if not isinstance(self.requirements, RequirementSet):
                raise TypeError("executable artifacts require a Core RequirementSet")
            requirements = self.requirements
            target_values = _strict_fields(
                self.target,
                expected={"snapshot_id"},
                owner="artifact target",
            )
            _require_sha256(target_values["snapshot_id"], owner="target.snapshot_id")
            target = _strict_v2_value(target_values, owner="artifact target")
            compilation_values = _strict_fields(
                self.compilation,
                expected={
                    "target_legalization_identity",
                    "schedule_identity",
                    "emission_identity",
                    "conformance_identity",
                },
                owner="compilation identity",
            )
            for name, value in compilation_values.items():
                _require_sha256(value, owner=f"compilation.{name}")
            compilation = _strict_v2_value(
                compilation_values,
                owner="compilation identity",
            )
            result_values = _strict_fields(
                self.result_schema,
                expected={"kind", "wires", "shots_source"},
                owner="result schema",
            )
            wires = result_values["wires"]
            if (
                result_values["kind"] != "samples"
                or result_values["shots_source"] != "execution_request"
                or not isinstance(wires, list)
                or not wires
                or any(type(wire) is not int for wire in wires)
                or wires != list(range(len(wires)))
            ):
                raise ValueError(
                    "executable result schema requires dense full-register samples"
                )
            result_schema = _strict_v2_value(result_values, owner="result schema")

        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "payload_sha256", payload_digest)
        object.__setattr__(self, "circuit_content_hash", circuit_hash)
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "compilation", compilation)
        object.__setattr__(self, "parameter_schema", parameter_schema)
        object.__setattr__(self, "result_schema", result_schema)

        body = self._identity_payload()
        structured = {key: value for key, value in body.items() if key != "payload"}
        entry_counter = [1]
        _strict_v2_value(
            structured,
            owner="v2 artifact",
            counter=entry_counter,
        )
        if isinstance(body["payload"], Mapping):
            _strict_v2_value(
                body["payload"],
                owner="v2 artifact payload",
                depth=1,
                counter=entry_counter,
                limit_strings=False,
            )
        _reject_sensitive_v2(structured, owner="v2 artifact")
        expected_identity = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        if self.artifact_identity and self.artifact_identity != expected_identity:
            raise ValueError("artifact_identity does not match canonical envelope")
        object.__setattr__(self, "artifact_identity", expected_identity)
        if len(_canonical_json_bytes(self.to_dict())) > _V2_MAX_ENVELOPE_BYTES:
            raise ValueError("artifact envelope exceeds maximum UTF-8 bytes")

    def _identity_payload(self) -> dict[str, Any]:
        requirements = self.requirements
        return {
            "schema": "flagquantum.program_artifact",
            "version": self.version,
            "kind": self.kind.value,
            "producer": self.producer,
            "profile": _thaw_v2(self.profile),
            "payload": _thaw_v2(self.payload),
            "payload_sha256": self.payload_sha256,
            "circuit_content_hash": self.circuit_content_hash,
            "requirements": (None if requirements is None else requirements.to_dict()),
            "target": _thaw_v2(self.target),
            "compilation": _thaw_v2(self.compilation),
            "parameter_schema": _thaw_v2(self.parameter_schema),
            "result_schema": _thaw_v2(self.result_schema),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "artifact_identity": self.artifact_identity}

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramArtifactV2":
        from .target_capabilities import RequirementSet

        values = _strict_fields(
            payload,
            expected=_V2_FIELDS,
            owner="v2 program artifact",
        )
        if values["schema"] != "flagquantum.program_artifact":
            raise ValueError("invalid program artifact schema")
        if type(values["version"]) is not str:
            raise TypeError("v2 artifact version must be a string")
        if type(values["producer"]) is not str:
            raise TypeError("v2 artifact producer must be a string")
        requirements = values["requirements"]
        if requirements is not None:
            requirements = RequirementSet.from_dict(requirements)
        return cls(
            version=values["version"],
            kind=ArtifactKind(values["kind"]),
            producer=values["producer"],
            profile=values["profile"],
            payload=values["payload"],
            payload_sha256=values["payload_sha256"],
            circuit_content_hash=values["circuit_content_hash"],
            requirements=requirements,
            target=values["target"],
            compilation=values["compilation"],
            parameter_schema=values["parameter_schema"],
            result_schema=values["result_schema"],
            artifact_identity=values["artifact_identity"],
        )

    @classmethod
    def from_circuit_ir(cls, circuit_ir: Any, *, producer: str) -> "ProgramArtifactV2":
        from .ir import CircuitIR

        if not isinstance(circuit_ir, CircuitIR):
            raise TypeError("circuit_ir must be a CircuitIR")
        names = _parameter_names(circuit_ir)
        return cls(
            kind=ArtifactKind.CIRCUIT,
            producer=producer,
            profile=_V2_PROFILES["circuit-ir-1.0"],
            payload=circuit_ir.to_dict(),
            circuit_content_hash=circuit_ir.content_hash,
            requirements=None,
            target=None,
            compilation=None,
            parameter_schema={
                "binding": "symbolic" if names else "fully_bound",
                "parameters": list(names),
            },
            result_schema=None,
        )


def read_program_artifact(
    payload: Mapping[str, Any],
) -> ProgramArtifact | ProgramArtifactV2:
    """Dispatch a serialized artifact without changing the v1 reader."""

    if not isinstance(payload, Mapping):
        raise TypeError("program artifact payload must be a mapping")
    if payload.get("version") == PROGRAM_ARTIFACT_V2_VERSION:
        return ProgramArtifactV2.from_dict(payload)
    return ProgramArtifact.from_dict(payload)


def read_program_artifact_json(payload: str) -> ProgramArtifact | ProgramArtifactV2:
    """Decode an artifact while rejecting duplicate JSON object keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate program artifact field {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError("program artifact must be valid JSON") from error
    return read_program_artifact(decoded)


def migrate_v1_circuit_artifact(artifact: ProgramArtifact) -> ProgramArtifactV2:
    """Migrate only the approved unambiguous v1 circuit subset."""

    from .ir import CircuitIR

    if not isinstance(artifact, ProgramArtifact):
        raise TypeError("v1 circuit migration requires a ProgramArtifact")
    blockers = []
    if artifact.kind is not ArtifactKind.CIRCUIT:
        blockers.append("kind_is_not_circuit")
    if artifact.metadata:
        blockers.append("metadata_is_not_empty")
    if artifact.required_capabilities:
        blockers.append("required_capabilities_are_not_empty")
    if artifact.parent_hashes:
        blockers.append("parent_hashes_are_not_empty")
    if blockers:
        raise ValueError("v1 artifact cannot migrate: " + ", ".join(blockers))
    try:
        circuit = CircuitIR.from_dict(artifact.payload)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "v1 artifact cannot migrate: invalid_circuit_payload"
        ) from error
    return ProgramArtifactV2.from_circuit_ir(circuit, producer=artifact.producer)


__all__ = (
    "ARTIFACT_ENVELOPE_VERSION",
    "PROGRAM_ARTIFACT_V2_VERSION",
    "ArtifactKind",
    "ProgramArtifact",
    "ProgramArtifactV2",
    "migrate_v1_circuit_artifact",
    "read_program_artifact",
    "read_program_artifact_json",
)
