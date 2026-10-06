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

#: Rule names :attr:`OperatorSchema.power_rule` may return, and what each one means.
#:
#: ``scale_single_parameter``  the opcode declares exactly one parameter and its
#:                             operator is the exponential of that parameter, so the
#:                             gate applied ``k`` times is the same gate with that one
#:                             parameter multiplied by ``k``. This is the only rewrite
#:                             a power performs, and it is measured rather than
#:                             assumed: the circuit composition contract gate
#:                             compares the two forms through the dense operator of
#:                             every opcode this rule names, so an opcode whose angle
#:                             does not scale linearly is caught by the gate instead
#:                             of being mis-announced here.
#: ``repeat_instruction``      the declaration fixes no single-gate form, so the power
#:                             is written out as further copies of the program. This
#:                             is exact for every instruction, unitary or not, which
#:                             is why it is the rule for everything else: a
#:                             multi-angle opcode such as ``u3``, a gate with no
#:                             parameters at all, a custom operation carrying its own
#:                             matrix, and a noise channel all get it.
POWER_RULES: tuple[str, ...] = (
    "scale_single_parameter",
    "repeat_instruction",
)

#: Rule names ``OperatorSchema.control`` may use, and what each one emits.
#:
#: A controlled form is not a corner of the matrix library but a closed identity over
#: the same registered opcodes, so it is declared here with the standing ``adjoint``
#: has. Two facts are declared: which registered opcode already is the single-control
#: form (``CONTROL_PARTNERS``, consulted before any rule below is read), and which
#: identity carries the remaining control count.
#:
#: Every identity is exact and uses no ancilla, which is what makes it safe inside a
#: program builder. A builder has no way to learn which qubit starts in ``|0>``, so a
#: construction that borrowed an ancilla would be right on a clean one and silently
#: wrong on a dirty one. The price is depth: the ladder for ``k`` controls emits
#: ``4 * 3 ** (k - 1) - 3`` gates, so ``MAX_LADDER_LEVEL`` bounds it rather than
#: letting an arithmetic request produce a program nobody can inspect.
#:
#: ``identity``              the controlled form of the gate is the empty program.
#: ``diagonal_ladder``       the gate is a phase ``P(phi)``, so ``C^k`` is the phase
#:                           ladder itself.
#: ``x_conjugated_ladder``   the gate is ``H P(phi) H``.
#: ``y_conjugated_ladder``   the gate is ``(S H) P(phi) (S H)^dagger``.
#: ``ry_conjugated_ladder``  the gate is ``RY(pi/4) P(pi) RY(-pi/4)``.
#: ``rz_ladder``             the gate is ``P(theta)`` up to one eigenvalue phase
#:                           ``-theta/2``, which stops being global once controlled.
#: ``rx_ladder``             ``H P(theta) H`` up to the same eigenvalue phase.
#: ``ry_ladder``             ``(S H) P(theta) (S H)^dagger`` up to the same phase.
#: ``u_angle_ladder``        a gate whose two or three angles are two diagonal phases
#:                           around one conjugated rotation.
#: ``swap_ladder``           ``SWAP`` as three ``CX``.
#: ``cswap_ladder``          ``CSWAP`` as three ``CCX``.
#: ``rzz_ladder``            ``RZZ`` as ``CX RZ CX``.
#: ``rxx_ladder``            ``RXX`` as ``(H x H) RZZ (H x H)``.
#: ``ryy_ladder``            ``RYY`` as ``(S H x S H) RZZ (S H x S H)^dagger``.
#: ``not_available``         the opcode has no controlled form this IR can express.
CONTROL_RULES: tuple[str, ...] = (
    "identity",
    "diagonal_ladder",
    "x_conjugated_ladder",
    "y_conjugated_ladder",
    "ry_conjugated_ladder",
    "rz_ladder",
    "rx_ladder",
    "ry_ladder",
    "u_angle_ladder",
    "swap_ladder",
    "cswap_ladder",
    "rzz_ladder",
    "rxx_ladder",
    "ryy_ladder",
    "not_available",
)

#: The deepest phase ladder one controlled instruction may need.
#:
#: A ladder of level ``level`` emits ``4 * 3 ** (level - 1) - 3`` instructions, so the
#: cost is exponential and this bound is what keeps a control count from producing a
#: program no one can inspect. It is a bound on the *ladder*, not on the control count,
#: because a wider gate reaches a deeper ladder for the same count:
#: ``control_ladder_level`` returns ``n_controls + arity - 1``. At this level the
#: widest route -- ``CSWAP``, whose three Toffolis each reach one level further -- emits
#: 236193 instructions, which the circuit composition contract records as measured
#: evidence rather than as an estimate.
MAX_LADDER_LEVEL = 10


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
    #: How one more control is added to this opcode; see ``CONTROL_RULES``. The
    #: default is the refusal, so an opcode added without a controlled form is
    #: uncontrollable rather than silently controlled by the wrong identity.
    control: str = "not_available"

    def __post_init__(self) -> None:
        if self.control not in CONTROL_RULES:
            raise ValueError(
                f"opcode {self.opcode!r} declares control rule {self.control!r}, "
                f"which is not one of {CONTROL_RULES}"
            )
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

    @property
    def power_rule(self) -> str:
        """Whether this opcode's angles scale, or its power is a repetition.

        This is read from the declaration rather than stored beside it, so
        "the gate's angle scales linearly" and "the powered gate is one
        instruction" cannot drift apart. The derivation is a measurement of the
        gates:

        * a unitary that declares exactly one parameter is the exponential of that
          parameter, so the gate applied ``k`` times is that gate with the parameter
          multiplied by ``k``, which is one instruction and an exact result;
        * every other instruction -- no parameter at all, several whose joint action
          is not linear in them, an unknown opcode carrying its own matrix, or a noise
          channel -- has no single-gate form, so the power is written as further
          copies of the program, which is exact for any instruction and needs no
          property of the gate beyond being executable.

        The first claim is a property of the gates and not of the field, so the circuit
        composition contract gate measures it through each opcode's dense operator and
        refuses a gate whose angle turns out not to scale. A prediction read off a
        declaration is only as good as the measurement behind it.
        """

        if self.unitary and len(self.parameters) == 1:
            return "scale_single_parameter"
        return "repeat_instruction"

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
    control: str = "not_available",
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
        control=control,
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
    _unitary("i", 1, aliases=("id",), adjoint="self_inverse", control="identity"),
    _unitary("x", 1, adjoint="self_inverse", control="x_conjugated_ladder"),
    _unitary("y", 1, adjoint="self_inverse", control="y_conjugated_ladder"),
    _unitary("z", 1, adjoint="self_inverse", control="diagonal_ladder"),
    _unitary(
        "h",
        1,
        aliases=("hadamard",),
        adjoint="self_inverse",
        control="ry_conjugated_ladder",
    ),
    _unitary("s", 1, adjoint="sdg", control="diagonal_ladder"),
    _unitary("sdg", 1, aliases=("sd",), adjoint="s", control="diagonal_ladder"),
    _unitary("t", 1, adjoint="tdg", control="diagonal_ladder"),
    _unitary("tdg", 1, aliases=("td",), adjoint="t", control="diagonal_ladder"),
    _unitary("sx", 1, adjoint="sxdg", control="x_conjugated_ladder"),
    _unitary("sxdg", 1, adjoint="sx", control="x_conjugated_ladder"),
    _unitary(
        "rx",
        1,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="rx_ladder",
    ),
    _unitary(
        "ry",
        1,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="ry_ladder",
    ),
    _unitary(
        "rz",
        1,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="rz_ladder",
    ),
    _unitary(
        "phase",
        1,
        aliases=("p",),
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="diagonal_ladder",
    ),
    _unitary(
        "u1",
        1,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="diagonal_ladder",
    ),
    _unitary(
        "u2",
        1,
        parameters=("phi", "lbd"),
        frequencies=((1.0,), (1.0,)),
        adjoint="adjoint_u2_angles",
        control="u_angle_ladder",
    ),
    _unitary(
        "u3",
        1,
        aliases=("u",),
        parameters=("theta", "phi", "lbd"),
        frequencies=((1.0,), (1.0,), (1.0,)),
        adjoint="adjoint_u3_angles",
        control="u_angle_ladder",
    ),
    _unitary(
        "cx",
        2,
        aliases=("cnot",),
        adjoint="self_inverse",
        control="x_conjugated_ladder",
    ),
    _unitary("cy", 2, adjoint="self_inverse", control="y_conjugated_ladder"),
    _unitary("cz", 2, adjoint="self_inverse", control="diagonal_ladder"),
    _unitary("swap", 2, adjoint="self_inverse", control="swap_ladder"),
    _unitary(
        "crx",
        2,
        parameters=("theta",),
        frequencies=((0.5, 1.0),),
        adjoint="negate_parameters",
        control="rx_ladder",
    ),
    _unitary(
        "cry",
        2,
        parameters=("theta",),
        frequencies=((0.5, 1.0),),
        adjoint="negate_parameters",
        control="ry_ladder",
    ),
    _unitary(
        "crz",
        2,
        parameters=("theta",),
        frequencies=((0.5, 1.0),),
        adjoint="negate_parameters",
        control="rz_ladder",
    ),
    _unitary(
        "cphase",
        2,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="diagonal_ladder",
    ),
    _unitary(
        "rxx",
        2,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="rxx_ladder",
    ),
    _unitary(
        "ryy",
        2,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="ryy_ladder",
    ),
    _unitary(
        "rzz",
        2,
        parameters=("theta",),
        frequencies=((1.0,),),
        adjoint="negate_parameters",
        control="rzz_ladder",
    ),
    _unitary(
        "ccx",
        3,
        aliases=("ccnot", "toffoli"),
        adjoint="self_inverse",
        control="x_conjugated_ladder",
    ),
    _unitary(
        "cswap",
        3,
        aliases=("fredkin",),
        adjoint="self_inverse",
        control="cswap_ladder",
    ),
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

#: Opcodes that already are the one-control form of another opcode.
#:
#: These are read before the ``control`` rule of the *controlling* opcode, so that
#: ``control.z`` at one control emits ``cz`` rather than rebuilding it, and
#: ``control.cx`` at one control emits ``ccx``. A gate's own rule describes how to add
#: controls beyond the ones it already has; this table describes when none need to be
#: added. Each pair is validated against the registry below: the partner has to exist,
#: be a parameter-free unitary of the same arity, and be the canonical opcode.
CONTROL_PARTNERS: Mapping[str, str] = MappingProxyType(
    {
        "x": "cx",
        "y": "cy",
        "z": "cz",
        "rx": "crx",
        "ry": "cry",
        "rz": "crz",
        "phase": "cphase",
        "u1": "cphase",
        "swap": "cswap",
        "cx": "ccx",
    }
)


def canonical_opcode(name: str) -> str:
    normalized = str(name).strip().lower()
    return OPERATOR_ALIASES.get(normalized, normalized)


def get_operator_schema(name: str) -> OperatorSchema | None:
    return OPERATOR_SCHEMAS.get(canonical_opcode(name))


def _control_partner_errors() -> tuple[str, ...]:
    """Return the pairs in ``CONTROL_PARTNERS`` a gate's own rule already covers.

    The pair table is a shortcut, so a wrong entry would be invisible: the emitted
    program would still be a program, just not the controlled gate. Each pair is
    therefore checked against the registry at import time -- the partner has to be a
    registered, parameter-free, unitary of the same arity written under its canonical
    name -- and a pair is refused when the two opcodes declare different rules, which
    is how a copy-paste mistake in this table would first show up.
    """

    errors = []
    for opcode, partner in CONTROL_PARTNERS.items():
        schema = OPERATOR_SCHEMAS.get(opcode)
        partner_schema = OPERATOR_SCHEMAS.get(partner)
        if schema is None or partner_schema is None:
            errors.append(
                f"controlled pair {opcode!r} -> {partner!r} names an unregistered opcode"
            )
            continue
        if (
            not partner_schema.unitary
            or partner_schema.parameters
            or partner_schema.arity != schema.arity
            or canonical_opcode(partner) != partner
        ):
            errors.append(
                f"controlled pair {opcode!r} -> {partner!r} is not a parameter-free "
                "unitary of the same arity"
            )
            continue
        if partner_schema.control != schema.control:
            errors.append(
                f"controlled pair {opcode!r} -> {partner!r} disagrees on the control "
                f"rule: {schema.control!r} against {partner_schema.control!r}"
            )
    return tuple(errors)


def control_ladder_level(arity: int, n_controls: int) -> int:
    """Return the deepest phase ladder a controlled gate of this shape needs.

    A gate of ``arity`` qubits has ``arity - 1`` partners that a control can be
    attached to directly and one place left over where the phase ladder starts, so the
    ladder level is ``n_controls + arity - 1``. One control on a one-qubit gate is the
    shallowest case, level one, and one control on ``CSWAP`` is the deepest, level
    three.
    """

    return n_controls + arity - 1


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


def _scale_angle(value: Any, factor: int) -> Any:
    """Multiply one angle by an integer, or every angle of a per-batch sequence.

    A tensor, a float, and a symbolic ``Parameter`` all multiply on the right, so one
    expression covers the three forms an angle is stored in. ``factor`` is multiplied
    rather than added ``factor`` times, which is what keeps ``power(1)`` the identity
    on the parameters and keeps a batched angle a single instruction.
    """

    if isinstance(value, (list, tuple)):
        return type(value)(_scale_angle(item, factor) for item in value)
    return value * factor


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


#: A power is a rewriting of instructions the IR already describes, so the only limit
#: is readability: beyond this many emitted instructions the arithmetic is still exact
#: but the program is no longer something a user can inspect, and a product is the
#: better way to ask for it.
MAX_POWER_REPEATS = 4096


def power_operator(
    schema: OperatorSchema, params: Mapping[str, Any], exponent: int
) -> tuple[str, dict[str, Any]] | None:
    """Return the single gate that equals ``exponent`` copies of one gate, if any.

    A gate whose declaration is the exponential of one parameter has a one-instruction
    ``exponent``-fold action: the same gate with that parameter multiplied by
    ``exponent``. That is the whole of this function, and it returns the opcode and the
    scaled parameters.

    ``None`` means no single gate equals the repeated one, and the caller must write
    the power out as further copies of the program. This is the common case rather than
    an error: no parameter-free gate has a one-instruction square that is not itself,
    a multi-angle gate's angles do not scale, a custom operation is defined by its
    matrix rather than by a declaration, and a noise channel is not a unitary at all.

    The caller must only reach for this when the power is of a *single-instruction*
    program. For a longer program the copies of one gate are separated by the other
    gates, so they do not compose into one instruction even when each of them does.

    Args:
        schema: The declaration of the gate being raised to a power.
        params: That gate's parameter mapping.
        exponent: A positive integer count of applications.

    Returns:
        The opcode and its scaled parameters, or ``None`` when no single gate equals
        the repetition.

    Raises:
        ValueError: If ``exponent`` is not positive.
    """

    if exponent < 1:
        raise ValueError(f"power requires a positive exponent, got {exponent}")
    if schema.power_rule != "scale_single_parameter":
        return None
    return schema.opcode, {
        name: (
            _scale_angle(params[name], exponent) if name in schema.parameters else value
        )
        for name, value in params.items()
    }


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
            "power": schema.power_rule,
            "control": schema.control,
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
    "MAX_POWER_REPEATS",
    "OPERATOR_ALIASES",
    "OPERATOR_SCHEMAS",
    "POWER_RULES",
    "canonical_opcode",
    "get_operator_schema",
    "gate_info",
    "inverse_operator",
    "operator_manifest",
    "parameter_shift_rule",
    "power_operator",
]
