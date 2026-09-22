"""Versioned, validated intermediate representation for FlagQuantum."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

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


def _normalize_count(value: Any, *, what: str) -> int:
    """Return a count as an integer, refusing anything that does not denote one.

    ``n_wires`` is a number of wires and a ``shape`` entry is a dimension, so a value
    that is not exactly an integer is not a count.  ``int(value)`` accepted ``2.7`` as
    ``2``, ``"3"`` as ``3`` and ``True`` as ``1``, which silently rewrote the width of
    the program the caller described and then executed it at that other width.  Only
    values that already denote an integer are accepted, through the interpreter's own
    ``__index__`` protocol, which admits NumPy integers and zero-dimensional torch
    integer tensors while refusing floats and strings.  ``bool`` is refused although it
    satisfies ``operator.index``, for the reason :func:`_normalize_shots` gives: it is
    a flag rather than a count, and the other count-taking entry points in this package
    refuse it too.

    ``TypeError`` reports a wrong Python type, which is what the errors-module boundary
    reserves for it; a magnitude is the caller's to check.
    """

    if isinstance(value, bool):
        raise TypeError(f"{what} must be an integer, not a bool")
    try:
        count = operator.index(value)
    except TypeError:
        raise TypeError(
            f"{what} must be an integer, got {type(value).__name__}"
        ) from None
    return int(count)


def _normalize_shots(shots: Any) -> int | None:
    """Return ``shots`` as a shot count, refusing anything that is not one.

    ``shots`` says how many times a program is run, so a value that is not exactly a
    positive integer is not a shot count.  A float, a string or a ``Decimal`` is refused
    rather than rounded, because rounding it would run a different number of shots than
    the caller wrote while reporting the result as the requested one.  ``bool`` is
    refused although it satisfies ``operator.index``: ``int(True) == 1`` is not a caller
    asking for a single shot, and every other count-taking entry point in this package
    refuses it for the same reason.
    """

    if shots is None:
        return None
    if isinstance(shots, bool):
        raise TypeError("measurement shots must be an integer or None, not a bool")
    try:
        # `operator.index` is the protocol for "I am exactly an integer", so it accepts
        # the scalar integer types other arrays and tensors provide.
        count = operator.index(shots)
    except TypeError:
        raise TypeError(
            "measurement shots must be an integer or None, got "
            f"{type(shots).__name__}"
        ) from None
    if count <= 0:
        raise IRValidationError("measurement shots must be a positive integer")
    return int(count)


# NumPy spells the element kind as a code: complex, unicode, bytes, object,
# datetime, timedelta and void are not real numbers.  Torch has no ``kind``, so a
# bool is recognised by name instead.
_NON_REAL_DTYPE_KINDS = frozenset("cUSOMmV")
_BOOL_DTYPE_NAMES = frozenset({"bool", "torch.bool"})


def _dtype_holds_real_numbers(dtype: Any) -> bool:
    """Return whether an array-like element ``dtype`` describes real numbers."""

    if getattr(dtype, "is_complex", False):
        return False
    if str(dtype) in _BOOL_DTYPE_NAMES:
        return False
    return getattr(dtype, "kind", None) not in _NON_REAL_DTYPE_KINDS


def _is_being_traced() -> bool:
    """Return whether a value's magnitude is being traced rather than known.

    ``Module`` compiles a circuit builder with ``make_fx``, which hands the builder a
    real tensor and refuses to let it be read: ``bool(tensor)`` fails so that a value
    cannot be baked into the traced program as a branch.  An angle written that way
    has no magnitude to check while the program is built -- the value that will be
    used is bound when the program runs, and ``Module`` owns that boundary -- so this
    check defers rather than reading a tracer's sample.

    The tracing mode is asked, instead of a failed read being caught and treated as
    this case, because a read that fails outside tracing is a defect of its own and
    must not be waved through as an unknown magnitude.
    """

    try:
        from torch.fx.experimental.proxy_tensor import get_proxy_mode
    except ImportError:
        # A torch without ``fx`` has no tracing to detect, and refusing every angle
        # over a missing optional module would be a worse failure than this check.
        return False
    return get_proxy_mode() is not None


def _holds_only_finite_values(value: Any) -> bool:
    """Return whether every real number ``value`` carries is finite.

    ``math.isfinite`` answers for a scalar.  A value too large for a float raises
    ``OverflowError`` instead of answering, and that is its own answer: it is finite
    and merely outside float range, so a range check would be reading a different
    question into this one.

    An array-like is asked in whichever of two generic ways it offers rather than by
    type, so no import is needed for any of them.  ``isfinite()`` covers a torch
    tensor and a this-package carrier, and it is also the only way a zero-dimensional
    value is reached, since that one cannot be walked.  Anything else is walked as a
    sequence, which is how a NumPy array is read without importing NumPy -- a lane
    that installs ``torch`` and little else still has to collect this module.

    A traced carrier is not asked at all: see :func:`_is_being_traced`.
    """

    if isinstance(value, Real):
        try:
            return math.isfinite(value)
        except OverflowError:
            return True
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        return True if _is_being_traced() else bool(isfinite().all())
    return all(_holds_only_finite_values(item) for item in value)


def _require_finite_angle(value: Any, *, owner: str) -> None:
    """Refuse a gate angle that is not a finite real number."""

    if _holds_only_finite_values(value):
        return
    if isinstance(value, Real):
        raise IRValidationError(f"{owner} must be a finite real number, got {value!r}")
    raise IRValidationError(
        f"{owner} must be a finite real number, got a non-finite "
        f"{type(value).__name__}"
    )


def _normalize_angle(value: Any, *, opcode: str, parameter: str) -> None:
    """Check that a gate parameter is a real angle, without rewriting it.

    Every gate parameter in the operator registry is a rotation angle, so it has to
    denote a real number.  ``Circuit.ry(0, "0.3")`` stored the string, and the only
    reader was the simulator: the failure surfaced as ``ExecutionError: planned
    execution failed`` wrapping ``TypeError: new(): invalid data type 'str'``, which
    reports neither the gate nor the parameter.  ``ry(0, True)`` was worse than a late
    failure, because ``True`` became the angle one radian and the run succeeded.

    The value is checked and returned unchanged: only the simulator knows the batch
    size that decides how many angles one gate may carry, so length and shape stay
    that side of the boundary, and a bound tensor keeps its autograd identity.
    ``Parameter`` and ``ParameterExpression`` are accepted because they are angles
    whose value arrives later, at bind time, and a sequence of angles is checked
    element by element because that is how a per-batch angle is written.

    A wrong Python type is a ``TypeError``, which is what the errors-module boundary
    reserves for it; ``bool`` is refused although it is a real number, for the reason
    :func:`_normalize_count` gives -- it is a flag rather than an angle.  ``str``,
    ``bytes`` and ``bytearray`` are refused before the sequence branch, so that
    ``b"0"`` cannot iterate into ``48`` and be read as one.

    A real number of the wrong *magnitude* is a different mistake and gets the class
    the boundary reserves for a value: ``nan``, ``inf`` and ``-inf`` are not angles,
    and they used to be stored and executed, so ``ry(0, nan)`` returned a NaN
    distribution as its result while ``ry(0, nan).counts(64)`` failed a layer below
    the mistake in ``torch.multinomial``, naming neither the gate nor the parameter.
    That is an :class:`IRValidationError`, which is what :func:`_normalize_shots`
    raises for a magnitude in this same file.

    The magnitude is the one thing here that can be a fact about a *traced* value, and
    a traced angle has none yet, so :func:`_is_being_traced` defers it.  The type
    rules above are untouched by that: a type is static, so they hold on every path.
    """

    owner = f"gate {opcode!r} parameter {parameter!r}"
    if isinstance(value, (Parameter, ParameterExpression)):
        return
    if isinstance(value, bool):
        raise TypeError(f"{owner} must be a real number, not a bool")
    if isinstance(value, Real):
        _require_finite_angle(value, owner=owner)
        return
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{owner} must be a real number, got {type(value).__name__}")
    # An array, a tensor and this package's own Double-Single carrier all report an
    # element dtype, which is what says whether the numbers inside are real.
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        if not _dtype_holds_real_numbers(dtype):
            raise TypeError(
                f"{owner} must be a real number, got a {dtype} "
                f"{type(value).__name__}"
            )
        _require_finite_angle(value, owner=owner)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _normalize_angle(item, opcode=opcode, parameter=parameter)
        return
    raise TypeError(f"{owner} must be a real number, got {type(value).__name__}")


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
            for parameter in schema.parameters:
                _normalize_angle(
                    self.params[parameter], opcode=name, parameter=parameter
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
        object.__setattr__(self, "shots", _normalize_shots(self.shots))
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
        n_wires = _normalize_count(self.n_wires, what="n_wires")
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
        shape = tuple(
            _normalize_count(size, what="IR shape dimension") for size in self.shape
        ) or (2**n_wires,)
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
            n_wires=payload["n_wires"],
            instructions=tuple(
                _instruction_from_dict(item) for item in payload.get("instructions", ())
            ),
            version=str(payload.get("version", "")),
            dtype=str(payload.get("dtype", "")),
            shape=tuple(payload.get("shape", ())),
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
    return MeasurementNode(
        kind=str(payload.get("kind", "")),
        wires=tuple(int(item) for item in payload.get("wires", ())),
        shots=_normalize_shots(payload.get("shots")),
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
