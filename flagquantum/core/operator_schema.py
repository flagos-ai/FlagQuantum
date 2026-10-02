"""Single-source backend-neutral operator semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


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
    wire_convention: tuple[str, ...]
    parameter_frequencies: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.parameter_frequencies:
            return
        if self.semantic_kind != "unitary":
            raise ValueError(
                f"opcode {self.opcode!r} is a {self.semantic_kind} and cannot "
                "declare parameter frequencies"
            )
        if len(self.parameter_frequencies) != len(self.parameters):
            raise ValueError(
                f"opcode {self.opcode!r} declares {len(self.parameters)} "
                f"parameter(s) {self.parameters} but "
                f"{len(self.parameter_frequencies)} frequency set(s)"
            )
        for name, frequencies in zip(
            self.parameters, self.parameter_frequencies, strict=True
        ):
            try:
                parameter_shift_rule(frequencies)
            except ValueError as error:
                raise ValueError(
                    f"opcode {self.opcode!r} parameter {name!r}: {error}"
                ) from error

    @property
    def unitary(self) -> bool:
        return self.semantic_kind == "unitary"

    @property
    def channel(self) -> bool:
        return self.semantic_kind == "channel"

    @property
    def differentiable(self) -> bool:
        """Whether a derivative can be returned for this opcode.

        This is read from the declared frequency sets rather than stored, so
        "the executor can differentiate this" and "the derivative rule is known"
        cannot drift apart.
        """

        return bool(self.parameter_frequencies)

    def shift_rule(self, parameter: str) -> tuple[tuple[float, float], ...]:
        """Coefficients and shifts that differentiate ``parameter``.

        The rule follows from the gate's declared frequencies, so an executor
        that can already compute an expectation value can also differentiate the
        gate, with no second place to record which gates are differentiable.

        Raises:
            ValueError: If the opcode is a channel, does not declare this
                parameter, or declares no frequencies for it.
        """

        if self.channel or parameter not in self.parameters:
            raise ValueError(
                f"opcode {self.opcode!r} does not declare a differentiable "
                f"parameter {parameter!r}"
            )
        return parameter_shift_rule(
            self.parameter_frequencies[self.parameters.index(parameter)]
        )


def parameter_shift_rule(
    frequencies: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """Coefficients and shifts that differentiate one declared frequency set.

    A parameterized gate whose generator has frequencies
    ``{f_1, ..., f_n}`` has an expectation value that is a trigonometric
    polynomial in those frequencies, so its derivative is a fixed linear
    combination of ``2n`` evaluations at shifted parameter values. This returns
    those ``(coefficient, shift)`` pairs.

    Only equidistant frequencies are supported, which is what every built-in
    opcode declares. A non-equidistant set is rejected rather than silently
    differentiated with the wrong coefficients.

    Raises:
        ValueError: If ``frequencies`` is empty, holds non-positive or repeated
            values, or is not equidistant.
    """

    declared = tuple(float(value) for value in frequencies)
    if not declared:
        raise ValueError("parameter shift requires at least one frequency")
    if any(value <= 0 for value in declared):
        raise ValueError(f"frequencies must be positive, got {declared}")
    ordered = tuple(sorted(declared))
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"frequencies must be unique, got {declared}")
    if len(ordered) > 1:
        gaps = {round(b - a, 10) for a, b in zip(ordered, ordered[1:], strict=False)}
        if len(gaps) != 1:
            raise ValueError(
                f"only equidistant frequencies are supported, got {declared}"
            )
    count = len(ordered)
    lowest = ordered[0]
    rule = []
    for index in range(1, count + 1):
        shift = (2 * index - 1) * math.pi / (2 * count * lowest)
        coefficient = (
            lowest
            * (-1) ** (index - 1)
            / (4 * count * math.sin(math.pi * (2 * index - 1) / (4 * count)) ** 2)
        )
        rule.append((coefficient, shift))
        rule.append((-coefficient, -shift))
    return tuple(rule)


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
    parameter_frequencies: tuple[tuple[float, ...], ...] = ()

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)


def _unitary(
    opcode: str,
    arity: int,
    *,
    aliases: tuple[str, ...] = (),
    parameters: tuple[str, ...] = (),
    frequencies: tuple[tuple[float, ...], ...] = (),
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
        wire_convention=wire_names,
        parameter_frequencies=frequencies,
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
    _unitary("rx", 1, parameters=("theta",), frequencies=((1.0,),)),
    _unitary("ry", 1, parameters=("theta",), frequencies=((1.0,),)),
    _unitary("rz", 1, parameters=("theta",), frequencies=((1.0,),)),
    _unitary(
        "phase",
        1,
        aliases=("p",),
        parameters=("theta",),
        frequencies=((1.0,),),
    ),
    _unitary("u1", 1, parameters=("theta",), frequencies=((1.0,),)),
    _unitary("u2", 1, parameters=("phi", "lbd"), frequencies=((1.0,), (1.0,))),
    _unitary(
        "u3",
        1,
        aliases=("u",),
        parameters=("theta", "phi", "lbd"),
        frequencies=((1.0,), (1.0,), (1.0,)),
    ),
    _unitary("cx", 2, aliases=("cnot",), adjoint="self_inverse"),
    _unitary("cy", 2, adjoint="self_inverse"),
    _unitary("cz", 2, adjoint="self_inverse"),
    _unitary("swap", 2, adjoint="self_inverse"),
    _unitary("crx", 2, parameters=("theta",), frequencies=((0.5, 1.0),)),
    _unitary("cry", 2, parameters=("theta",), frequencies=((0.5, 1.0),)),
    _unitary("crz", 2, parameters=("theta",), frequencies=((0.5, 1.0),)),
    _unitary("cphase", 2, parameters=("theta",), frequencies=((1.0,),)),
    _unitary("rxx", 2, parameters=("theta",), frequencies=((1.0,),)),
    _unitary("ryy", 2, parameters=("theta",), frequencies=((1.0,),)),
    _unitary("rzz", 2, parameters=("theta",), frequencies=((1.0,),)),
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
        parameter_shapes=MappingProxyType(dict.fromkeys(schema.parameters, ())),
        differentiable=schema.differentiable,
        semantic_kind=schema.semantic_kind,
        parameter_frequencies=schema.parameter_frequencies,
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
            "parameter_frequencies": schema.parameter_frequencies,
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
    "parameter_shift_rule",
]
