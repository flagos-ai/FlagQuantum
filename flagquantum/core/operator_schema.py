"""Single-source backend-neutral operator semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

#: Rule names ``OperatorSchema.adjoint`` may use, and what each one computes.
#:
#: ``self_inverse``         the opcode is its own inverse; parameters are unchanged.
#: ``negate_parameters``    every angle is negated in place.
#: ``adjoint_u2_angles``    ``phi -> -lbd - pi`` and ``lbd -> -phi - pi``, which is the
#:                          inverse of ``U2(phi, lbd)`` in this gate's angle convention.
#: ``adjoint_u3_angles``    ``theta -> -theta``, ``phi -> -lbd`` and ``lbd -> -phi``,
#:                          which is the inverse of ``U3(theta, phi, lbd)``.
#: ``matrix_adjoint``       the declaration alone does not determine the inverse; the
#:                          concrete matrix is required.
#: ``not_applicable``       the opcode is not invertible (a noise channel).
#:
#: An ``adjoint`` value that is none of these names the inverse opcode directly, which is
#: how ``s`` and ``sdg`` declare each other. Such a partner must be a parameter-free
#: unitary of the same arity that declares this opcode back, so the pairing is symmetric.
ADJOINT_RULES: tuple[str, ...] = (
    "self_inverse",
    "negate_parameters",
    "adjoint_u2_angles",
    "adjoint_u3_angles",
    "matrix_adjoint",
    "not_applicable",
)


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
    qubit_convention: tuple[str, ...]
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
    """User-facing description of a built-in gate's qubits and parameters."""

    name: str
    aliases: tuple[str, ...]
    n_qubits: int
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
    qubit_names = {
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
        qubit_convention=qubit_names,
        parameter_frequencies=frequencies,
    )


def _channel(opcode: str, *, parameters: tuple[str, ...]) -> OperatorSchema:
    """Declare a channel and the probability or rate names it accepts.

    ``parameters`` is keyword-only so that a channel cannot be added without
    stating them: an opcode whose probability exists only inside its Kraus
    operators is one a reader of the lowered program cannot map, which is what
    kept the four channels out of every interop contract.

    ``differentiable`` ends up ``False`` for every channel even when
    ``parameters`` is non-empty, because a channel declares no parameter
    frequencies: the derived flag states whether a derivative rule is known for
    the opcode, and a channel is sampled rather than differentiated. Declaring a
    number and being able to differentiate it are two separate claims.
    """

    return OperatorSchema(
        opcode=opcode,
        aliases=(),
        arity=1,
        parameters=parameters,
        dtype_policy=("complex64", "complex128"),
        semantic_kind="channel",
        adjoint="not_applicable",
        decomposition=(),
        qubit_convention=("target",),
    )


_SCHEMAS = (
    _unitary("i", 1, aliases=("id",), adjoint="self_inverse"),
    _unitary("x", 1, adjoint="self_inverse"),
    _unitary("y", 1, adjoint="self_inverse"),
    _unitary("z", 1, adjoint="self_inverse"),
    _unitary("h", 1, aliases=("hadamard",), adjoint="self_inverse"),
    _unitary("s", 1, adjoint="sdg"),
    _unitary("sdg", 1, aliases=("sd",), adjoint="s"),
    _unitary("t", 1, adjoint="tdg"),
    _unitary("tdg", 1, aliases=("td",), adjoint="t"),
    _unitary("sx", 1, adjoint="sxdg"),
    _unitary("sxdg", 1, adjoint="sx"),
    _unitary(
        "rx",
        1,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "ry",
        1,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "rz",
        1,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "phase",
        1,
        aliases=("p",),
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "u1",
        1,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "u2",
        1,
        parameters=("phi", "lbd"),
        frequencies=((1.0,), (1.0,)),
        adjoint="adjoint_u2_angles",
    ),
    _unitary(
        "u3",
        1,
        aliases=("u",),
        parameters=("theta", "phi", "lbd"),
        frequencies=((1.0,), (1.0,), (1.0,)),
        adjoint="adjoint_u3_angles",
    ),
    _unitary("cx", 2, aliases=("cnot",), adjoint="self_inverse"),
    _unitary("cy", 2, adjoint="self_inverse"),
    _unitary("cz", 2, adjoint="self_inverse"),
    _unitary("swap", 2, adjoint="self_inverse"),
    _unitary(
        "crx",
        2,
        parameters=("theta",),
        frequencies=((0.5, 1.0),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "cry",
        2,
        parameters=("theta",),
        frequencies=((0.5, 1.0),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "crz",
        2,
        parameters=("theta",),
        frequencies=((0.5, 1.0),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "cphase",
        2,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "rxx",
        2,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "ryy",
        2,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary(
        "rzz",
        2,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
    ),
    _unitary("ccx", 3, aliases=("ccnot", "toffoli"), adjoint="self_inverse"),
    _unitary("cswap", 3, aliases=("fredkin",), adjoint="self_inverse"),
    _channel("bit_flip", parameters=("probability",)),
    _channel("phase_flip", parameters=("probability",)),
    _channel("depolarizing", parameters=("probability",)),
    _channel("amplitude_damping", parameters=("gamma",)),
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


def _negate_angle(value: Any) -> Any:
    """Negate one angle, or every angle of a per-batch sequence of angles."""

    if isinstance(value, (list, tuple)):
        return type(value)(_negate_angle(item) for item in value)
    return -value


def _shift_angle(value: Any, delta: float) -> Any:
    """Subtract ``delta`` from one angle, or from every angle of a sequence."""

    if isinstance(value, (list, tuple)):
        return type(value)(_shift_angle(item, delta) for item in value)
    return value - delta


def inverse_operator(
    schema: OperatorSchema, params: Mapping[str, Any]
) -> tuple[str, dict[str, Any]] | None:
    """Return the opcode and parameters that invert one use of ``schema``.

    The inverse replaces one gate inside a larger product, so it is a single gate of
    the same arity applied to the same qubits, never a decomposition.

    ``None`` means the schema's ``adjoint`` declaration does not determine an inverse
    on its own: ``matrix_adjoint`` needs the concrete matrix, and ``not_applicable``
    says no inverse exists. The caller is expected to refuse those cases and report
    which declaration it met, so that a caller never silently drops a gate.

    A parameter that carries one angle per batch entry is a sequence of angles, and it
    is transformed element by element. ``Parameter`` and ``ParameterExpression`` values
    are transformed symbolically, so the inverse of a template stays a template.

    Args:
        schema: The declaration of the gate being inverted.
        params: That gate's parameter mapping.

    Returns:
        The inverse opcode with its parameters, or ``None`` when the declaration is
        not enough.
    """

    if schema.adjoint == "self_inverse":
        return schema.opcode, dict(params)
    if schema.adjoint == "negate_parameters":
        return schema.opcode, {
            name: _negate_angle(params[name]) if name in schema.parameters else value
            for name, value in params.items()
        }
    if schema.adjoint == "adjoint_u2_angles":
        return schema.opcode, {
            **params,
            "phi": _shift_angle(_negate_angle(params["lbd"]), math.pi),
            "lbd": _shift_angle(_negate_angle(params["phi"]), math.pi),
        }
    if schema.adjoint == "adjoint_u3_angles":
        return schema.opcode, {
            **params,
            "theta": _negate_angle(params["theta"]),
            "phi": _negate_angle(params["lbd"]),
            "lbd": _negate_angle(params["phi"]),
        }
    partner = OPERATOR_SCHEMAS.get(schema.adjoint)
    if (
        partner is not None
        and partner.unitary
        and not partner.parameters
        and partner.arity == schema.arity
    ):
        return partner.opcode, {}
    return None


def gate_info(name: str) -> GateInfo:
    """Return discoverable parameter and qubit requirements for a built-in gate."""

    schema = get_operator_schema(name)
    if schema is None:
        raise ValueError(f"unknown FlagQuantum gate {name!r}")
    return GateInfo(
        name=schema.opcode,
        aliases=schema.aliases,
        n_qubits=schema.arity,
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
            "qubit_convention": schema.qubit_convention,
            "parameter_frequencies": schema.parameter_frequencies,
        }
        for schema in OPERATOR_SCHEMAS.values()
    )


__all__ = [
    "OperatorSchema",
    "GateInfo",
    "ADJOINT_RULES",
    "OPERATOR_ALIASES",
    "OPERATOR_SCHEMAS",
    "canonical_opcode",
    "get_operator_schema",
    "gate_info",
    "inverse_operator",
    "operator_manifest",
    "parameter_shift_rule",
]
