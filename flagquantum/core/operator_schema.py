"""Single-source backend-neutral operator semantics."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class OperatorSchema:
    opcode: str
    aliases: tuple[str, ...]
    arity: int
    parameters: tuple[str, ...]
    dtype_policy: tuple[str, ...]
    semantic_kind: str
    adjoint: str
    decomposition: tuple[str, ...]
    differentiable: bool
    wire_convention: tuple[str, ...]

    @property
    def unitary(self) -> bool:
        return self.semantic_kind == "unitary"

    @property
    def channel(self) -> bool:
        return self.semantic_kind == "channel"


@dataclass(frozen=True)
class GateInfo:
    """User-facing description of a built-in gate's wires and parameters."""

    name: str
    aliases: tuple[str, ...]
    n_wires: int
    parameters: tuple[str, ...]
    parameter_shapes: Mapping[str, tuple[int, ...]]
    differentiable: bool
    semantic_kind: str

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)


def _unitary(
    opcode: str,
    arity: int,
    *,
    aliases: tuple[str, ...] = (),
    parameters: tuple[str, ...] = (),
    adjoint: str = "matrix_adjoint",
    decomposition: tuple[str, ...] = (),
) -> OperatorSchema:
    wire_names = {
        1: ("target",),
        2: ("control_or_left", "target_or_right"),
        3: ("control_0", "control_1_or_target_0", "target_or_target_1"),
    }[arity]
    return OperatorSchema(
        opcode=opcode,
        aliases=aliases,
        arity=arity,
        parameters=parameters,
        dtype_policy=("complex64", "complex128"),
        semantic_kind="unitary",
        adjoint=adjoint,
        decomposition=decomposition,
        differentiable=bool(parameters),
        wire_convention=wire_names,
    )


def _channel(opcode: str) -> OperatorSchema:
    return OperatorSchema(
        opcode=opcode,
        aliases=(),
        arity=1,
        parameters=(),
        dtype_policy=("complex64", "complex128"),
        semantic_kind="channel",
        adjoint="not_applicable",
        decomposition=(),
        differentiable=False,
        wire_convention=("target",),
    )


_SCHEMAS = (
    _unitary("i", 1, aliases=("id",), adjoint="self_inverse"),
    _unitary("x", 1, adjoint="self_inverse"),
    _unitary("y", 1, adjoint="self_inverse"),
    _unitary("z", 1, adjoint="self_inverse"),
    _unitary("h", 1, aliases=("hadamard",), adjoint="self_inverse"),
    _unitary("s", 1),
    _unitary("sdg", 1, aliases=("sd",)),
    _unitary("t", 1),
    _unitary("tdg", 1, aliases=("td",)),
    _unitary("sx", 1),
    _unitary("sxdg", 1),
    _unitary("rx", 1, parameters=("theta",)),
    _unitary("ry", 1, parameters=("theta",)),
    _unitary("rz", 1, parameters=("theta",)),
    _unitary("phase", 1, aliases=("p",), parameters=("theta",)),
    _unitary("u1", 1, parameters=("theta",)),
    _unitary("u2", 1, parameters=("phi", "lbd")),
    _unitary("u3", 1, aliases=("u",), parameters=("theta", "phi", "lbd")),
    _unitary("cx", 2, aliases=("cnot",), adjoint="self_inverse"),
    _unitary("cy", 2, adjoint="self_inverse"),
    _unitary("cz", 2, adjoint="self_inverse"),
    _unitary("swap", 2, adjoint="self_inverse"),
    _unitary("crx", 2, parameters=("theta",)),
    _unitary("cry", 2, parameters=("theta",)),
    _unitary("crz", 2, parameters=("theta",)),
    _unitary("cphase", 2, parameters=("theta",)),
    _unitary("rxx", 2, parameters=("theta",)),
    _unitary("ryy", 2, parameters=("theta",)),
    _unitary("rzz", 2, parameters=("theta",)),
    _unitary("ccx", 3, aliases=("ccnot", "toffoli"), adjoint="self_inverse"),
    _unitary("cswap", 3, aliases=("fredkin",), adjoint="self_inverse"),
    _channel("bit_flip"),
    _channel("phase_flip"),
    _channel("depolarizing"),
    _channel("amplitude_damping"),
)

OPERATOR_SCHEMAS: Mapping[str, OperatorSchema] = MappingProxyType(
    {schema.opcode: schema for schema in _SCHEMAS}
)
OPERATOR_ALIASES: Mapping[str, str] = MappingProxyType(
    {alias: schema.opcode for schema in _SCHEMAS for alias in schema.aliases}
)


def canonical_opcode(name: str) -> str:
    normalized = str(name).strip().lower()
    return OPERATOR_ALIASES.get(normalized, normalized)


def get_operator_schema(name: str) -> OperatorSchema | None:
    return OPERATOR_SCHEMAS.get(canonical_opcode(name))


def gate_info(name: str) -> GateInfo:
    """Return discoverable parameter and wire requirements for a built-in gate."""

    schema = get_operator_schema(name)
    if schema is None:
        raise ValueError(f"unknown FlagQuantum gate {name!r}")
    return GateInfo(
        name=schema.opcode,
        aliases=schema.aliases,
        n_wires=schema.arity,
        parameters=schema.parameters,
        parameter_shapes=MappingProxyType(
            {parameter: () for parameter in schema.parameters}
        ),
        differentiable=schema.differentiable,
        semantic_kind=schema.semantic_kind,
    )


def operator_manifest() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "opcode": schema.opcode,
            "aliases": schema.aliases,
            "arity": schema.arity,
            "parameters": schema.parameters,
            "dtype_policy": schema.dtype_policy,
            "semantic_kind": schema.semantic_kind,
            "adjoint": schema.adjoint,
            "decomposition": schema.decomposition,
            "differentiable": schema.differentiable,
            "wire_convention": schema.wire_convention,
        }
        for schema in OPERATOR_SCHEMAS.values()
    )


__all__ = [
    "OperatorSchema",
    "GateInfo",
    "OPERATOR_ALIASES",
    "OPERATOR_SCHEMAS",
    "canonical_opcode",
    "get_operator_schema",
    "gate_info",
    "operator_manifest",
]
