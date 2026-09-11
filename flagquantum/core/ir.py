"""Versioned, validated intermediate representation for FlagQuantum."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from ..errors import SerializationError, ValidationError
from .operator_schema import (
    OPERATOR_SCHEMAS,
    OperatorSchema,
    canonical_opcode,
    get_operator_schema,
)
from .parameters import Parameter, ParameterExpression

IR_VERSION = "1.0"


class IRValidationError(ValidationError):
    """Raised when an IR object violates the versioned schema."""


class IRSerializationError(SerializationError):
    """Raised when an IR value cannot be represented deterministically."""


OpcodeSchema = OperatorSchema
OPCODE_SCHEMAS = OPERATOR_SCHEMAS


def _normalize_wires(wires: Sequence[int], *, owner: str) -> tuple[int, ...]:
    normalized = tuple(int(wire) for wire in wires)
    if not normalized:
        raise IRValidationError(f"{owner} requires at least one wire")
    if any(wire < 0 for wire in normalized):
        raise IRValidationError(f"{owner} wires must be non-negative: {normalized}")
    if len(set(normalized)) != len(normalized):
        raise IRValidationError(f"{owner} cannot repeat a wire: {normalized}")
    return normalized


@dataclass(frozen=True)
class Instruction:
    """Backend-neutral operation with opcode and parameter contracts."""

    name: str
    wires: tuple[int, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    matrix: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = canonical_opcode(self.name)
        if not name:
            raise IRValidationError("instruction opcode cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "wires", _normalize_wires(self.wires, owner=f"instruction {name!r}")
        )
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metadata", dict(self.metadata))

        schema = get_operator_schema(name)
        is_channel = bool(self.metadata.get("is_channel"))
        is_dynamic = bool(self.metadata.get("is_dynamic"))
        if schema is None and self.matrix is None and not is_channel and not is_dynamic:
            raise IRValidationError(
                f"unknown opcode {name!r}; custom operations require an explicit matrix"
            )
        if schema is not None:
            if len(self.wires) != schema.arity:
                raise IRValidationError(
                    f"opcode {name!r} requires {schema.arity} wire(s), got {len(self.wires)}"
                )
            missing = tuple(key for key in schema.parameters if key not in self.params)
            if missing:
                raise IRValidationError(
                    f"opcode {name!r} is missing parameter(s): {', '.join(missing)}"
                )


@dataclass(frozen=True)
class ObservableNode:
    """Observable requested from the circuit state."""

    name: str
    wires: tuple[int, ...]
    coefficient: Any = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if not name:
            raise IRValidationError("observable name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "wires", _normalize_wires(self.wires, owner=f"observable {name!r}")
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MeasurementNode:
    """Measurement requested from an executor."""

    kind: str
    wires: tuple[int, ...]
    shots: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if not kind:
            raise IRValidationError("measurement kind cannot be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "wires", _normalize_wires(self.wires, owner=f"measurement {kind!r}")
        )
        if self.shots is not None and int(self.shots) <= 0:
            raise IRValidationError("measurement shots must be a positive integer")
        if self.shots is not None:
            object.__setattr__(self, "shots", int(self.shots))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class CircuitIR:
    """Canonical FlagQuantum IR v1 consumed by compiler and executors."""

    n_wires: int
    instructions: tuple[Instruction, ...]
    version: str = IR_VERSION
    dtype: str = "complex64"
    shape: tuple[int, ...] = ()
    observables: tuple[ObservableNode, ...] = ()
    measurements: tuple[MeasurementNode, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_wires = int(self.n_wires)
        if n_wires <= 0:
            raise IRValidationError(f"n_wires must be positive, got {n_wires}")
        if str(self.version) != IR_VERSION:
            raise IRValidationError(
                f"unsupported IR version {self.version!r}; supported version is {IR_VERSION!r}"
            )
        object.__setattr__(self, "n_wires", n_wires)
        object.__setattr__(self, "version", IR_VERSION)
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "observables", tuple(self.observables))
        object.__setattr__(self, "measurements", tuple(self.measurements))
        object.__setattr__(self, "metadata", dict(self.metadata))
        dtype = str(self.dtype).removeprefix("torch.")
        if not dtype:
            raise IRValidationError("IR dtype cannot be empty")
        object.__setattr__(self, "dtype", dtype)
        shape = tuple(int(size) for size in self.shape) or (2**n_wires,)
        if any(size <= 0 for size in shape):
            raise IRValidationError(f"IR shape dimensions must be positive: {shape}")
        object.__setattr__(self, "shape", shape)
        self.validate()

    def validate(self) -> "CircuitIR":
        for owner, nodes in (
            ("instruction", self.instructions),
            ("observable", self.observables),
            ("measurement", self.measurements),
        ):
            for index, raw_node in enumerate(nodes):
                node: Any = raw_node
                outside = tuple(wire for wire in node.wires if wire >= self.n_wires)
                if outside:
                    raise IRValidationError(
                        f"{owner} {index} references wire(s) {outside} outside "
                        f"circuit range [0, {self.n_wires - 1}]"
                    )
        return self

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "flagquantum.circuit_ir",
            "version": self.version,
            "n_wires": self.n_wires,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "instructions": [_instruction_to_dict(item) for item in self.instructions],
            "observables": [_observable_to_dict(item) for item in self.observables],
            "measurements": [_measurement_to_dict(item) for item in self.measurements],
            "metadata": _encode_value(self.metadata),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CircuitIR":
        allowed = {
            "kind",
            "version",
            "n_wires",
            "dtype",
            "shape",
            "instructions",
            "observables",
            "measurements",
            "metadata",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise IRSerializationError(
                "serialized CircuitIR contains unknown field(s): " + ", ".join(unknown)
            )
        if payload.get("kind") != "flagquantum.circuit_ir":
            raise IRSerializationError(
                "serialized payload is not a FlagQuantum CircuitIR"
            )
        return cls(
            n_wires=int(payload["n_wires"]),
            instructions=tuple(
                _instruction_from_dict(item) for item in payload.get("instructions", ())
            ),
            version=str(payload.get("version", "")),
            dtype=str(payload.get("dtype", "")),
            shape=tuple(int(item) for item in payload.get("shape", ())),
            observables=tuple(
                _observable_from_dict(item) for item in payload.get("observables", ())
            ),
            measurements=tuple(
                _measurement_from_dict(item) for item in payload.get("measurements", ())
            ),
            metadata=_decode_value(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "CircuitIR":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise IRSerializationError(f"invalid CircuitIR JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise IRSerializationError("serialized CircuitIR must be a JSON object")
        return cls.from_dict(payload)

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)


def _encode_value(value: Any) -> Any:
    if isinstance(value, Parameter):
        return {"$parameter": value.name}
    if isinstance(value, ParameterExpression):
        return {
            "$expression": {
                "op": value.op,
                "args": [_encode_value(item) for item in value.args],
            }
        }
    if isinstance(value, complex):
        return {"$complex": [value.real, value.imag]}
    if isinstance(value, Mapping):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode_value(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        tensor = value.detach().cpu()
        return {
            "$tensor": {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "data": _encode_value(tensor.tolist()),
            }
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise IRSerializationError(
        f"value of type {type(value).__name__} is not deterministically serializable"
    )


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode_value(item) for item in value)
    if not isinstance(value, Mapping):
        return value
    if "$parameter" in value:
        return Parameter(str(value["$parameter"]))
    if "$expression" in value:
        expression = value["$expression"]
        return ParameterExpression(
            str(expression["op"]),
            tuple(_decode_value(item) for item in expression["args"]),
        )
    if "$complex" in value:
        real, imaginary = value["$complex"]
        return complex(real, imaginary)
    if "$tensor" in value:
        tensor = value["$tensor"]
        try:
            import torch

            dtype = getattr(torch, str(tensor["dtype"]))
            return torch.tensor(_decode_value(tensor["data"]), dtype=dtype).reshape(
                tuple(int(item) for item in tensor["shape"])
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise IRSerializationError(f"invalid tensor encoding: {exc}") from exc
    return {str(key): _decode_value(item) for key, item in value.items()}


def _instruction_to_dict(item: Instruction) -> dict[str, Any]:
    return {
        "opcode": item.name,
        "wires": list(item.wires),
        "params": _encode_value(item.params),
        "matrix": _encode_value(item.matrix),
        "metadata": _encode_value(item.metadata),
    }


def _instruction_from_dict(payload: Mapping[str, Any]) -> Instruction:
    return Instruction(
        name=str(payload.get("opcode", "")),
        wires=tuple(int(item) for item in payload.get("wires", ())),
        params=_decode_value(payload.get("params", {})),
        matrix=_decode_value(payload.get("matrix")),
        metadata=_decode_value(payload.get("metadata", {})),
    )


def _observable_to_dict(item: ObservableNode) -> dict[str, Any]:
    return {
        "name": item.name,
        "wires": list(item.wires),
        "coefficient": _encode_value(item.coefficient),
        "metadata": _encode_value(item.metadata),
    }


def _observable_from_dict(payload: Mapping[str, Any]) -> ObservableNode:
    return ObservableNode(
        name=str(payload.get("name", "")),
        wires=tuple(int(item) for item in payload.get("wires", ())),
        coefficient=_decode_value(payload.get("coefficient", 1.0)),
        metadata=_decode_value(payload.get("metadata", {})),
    )


def _measurement_to_dict(item: MeasurementNode) -> dict[str, Any]:
    return {
        "kind": item.kind,
        "wires": list(item.wires),
        "shots": item.shots,
        "metadata": _encode_value(item.metadata),
    }


def _measurement_from_dict(payload: Mapping[str, Any]) -> MeasurementNode:
    shots = payload.get("shots")
    return MeasurementNode(
        kind=str(payload.get("kind", "")),
        wires=tuple(int(item) for item in payload.get("wires", ())),
        shots=None if shots is None else int(shots),
        metadata=_decode_value(payload.get("metadata", {})),
    )


def from_engine_qir(n_wires: int, qir: Sequence[Mapping[str, Any]]) -> CircuitIR:
    """Normalize rich circuit QIR into validated canonical FlagQuantum IR."""

    instructions: list[Instruction] = []
    for item in qir:
        params = dict(item.get("parameters", {}) or {})
        metadata = {
            key: item[key]
            for key in ("split", "mpo", "diagonal", "is_channel")
            if key in item
        }
        instructions.append(
            Instruction(
                name=str(item.get("name", "")),
                wires=tuple(int(wire) for wire in item.get("index", ())),
                params=params,
                matrix=item.get("gate"),
                metadata=metadata,
            )
        )
    return CircuitIR(n_wires=int(n_wires), instructions=tuple(instructions))


def ensure_circuit_ir(program: Any) -> CircuitIR:
    """Return validated canonical IR or reject an unsupported executor input."""

    candidate = program.to_ir() if hasattr(program, "to_ir") else program
    if not isinstance(candidate, CircuitIR):
        raise TypeError(
            "FlagQuantum executors require CircuitIR or an object exposing to_ir()"
        )
    return candidate.validate()


__all__ = [
    "IR_VERSION",
    "IRSerializationError",
    "IRValidationError",
    "OpcodeSchema",
    "OPCODE_SCHEMAS",
    "Instruction",
    "ObservableNode",
    "MeasurementNode",
    "CircuitIR",
    "ensure_circuit_ir",
    "from_engine_qir",
]
