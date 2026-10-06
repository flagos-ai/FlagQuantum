"""Ancilla-free controlled expansions of the registered gates.

One control is often a gate the registry already has: the controlled ``x`` is ``cx``, the
controlled ``phase`` is ``cphase``. Every further control has to be built, and this module
is the single place that says how. It is backend-neutral -- it reads
:mod:`flagquantum.core.operator_schema` and emits instructions, never a matrix -- so one
expansion is what a Circuit records, what the IR validates, and what any executor runs.

The construction is one primitive: the phase ladder ``C^k(P(theta))``, written
recursively as

    C^k(P(t)) = A . C^{k-1}(P(-t/2)) . A . C^{k-1}(P(t/2)) . C^{k-1}(P(t/2))   with
    A = C^{k-1}(X) applied to the last control,

where ``A`` itself is ``h . C^{k-1}(P(pi)) . h``. Read as an exponent on the basis state
``x_1 ... x_{k-1} r s`` (controls, ladder rung, target) the emitted product contributes
``s * (t/2) * (r + m - (r xor m)) = t * s * r * m`` with ``m = x_1 ... x_{k-1}``, which is
exactly ``C^k(P(t))``. A purely diagonal recursion cannot reach that monomial: the
diagonal blocks of a ``(k-1)``-control ladder can only span ``{x_1..x_{k-1} x_k,
x_1..x_{k-1} x_t, x_k x_t}``, and the flip on the rung supplies the missing degree. That
is why the ladder carries an ``h`` conjugation rather than another phase gate.

No ancilla is used, and that is the point rather than an accident. A program builder
cannot know which qubit of the caller's circuit starts in ``|0>``: the framework's own
multi-controlled-X helper takes caller-supplied ancillas that must be clean and returns a
wrong answer, not a refusal, when they are not. A construction whose correctness depends
on a precondition the builder cannot check would be a silent wrong answer with a plausible
instruction listing. The price is depth, declared by ``MAX_LADDER_LEVEL``: the ladder of
level ``k`` emits ``4 * 3 ** (k - 1) - 3`` instructions, so the cost is exponential in the
control count, and ``Circuit.control`` publishes that count rather than hiding it.

Every rule is exact at any angle, including zero: a zero angle emits the identity gates it
is written as rather than being elided, so an emitted instruction count is a function of
the shape alone. Eliding would need a numeric comparison that a ``Parameter``, a per-batch
sequence, and a bound tensor cannot all answer, and a tolerance that dropped a small
non-zero angle would change the program by more than round-off.

The gates that are not already diagonal are reached by conjugating the ladder with their
own single-qubit eigenvectors, and one detail is easy to get wrong: a factorization
``V P(t) V^dagger`` differs from the gate by a *global* phase on one qubit, and the moment
that qubit becomes a control the global phase turns into a *relative* phase between the
control's zero and one branches. ``_correction`` restores it. The single-control partner
is exempt, because that opcode already is the controlled gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .ir import Instruction
from .operator_schema import (
    CONTROL_PARTNERS,
    MAX_LADDER_LEVEL,
    OperatorSchema,
    control_ladder_level,
)

__all__ = ["controlled_instructions"]

#: A half turn, used as the base angle of the flip and as the factor that halves one.
_PI = math.pi
_HALF = 0.5

#: The basis change ``H``, written in application order. ``H`` is its own inverse.
_BASIS_H: tuple[str, ...] = ("h",)

#: The basis change ``V = S H`` that turns ``Z`` into ``Y``, in application order.
_BASIS_V: tuple[str, ...] = ("h", "s")

#: ``V^dagger``, which is what a conjugation applies before the phase gate. Reading the
#: emitted sequence left to right as gates applied to the state, ``V^dagger`` comes first
#: and ``V`` last so that the product is ``V P V^dagger``.
_BASIS_V_DAGGER: tuple[str, ...] = ("sdg", "h")

#: The phase angle each parameter-free opcode contributes to the ladder.
#:
#: A parameter-free gate is still a phase gate in some basis, and this is the table that
#: says which one. It is the only operator data in this module and it is deliberately
#: here rather than in the operator registry: the registry describes what a gate *is* --
#: its arity, its parameters, how it inverts -- while this describes how one particular
#: expansion is written, which is this module's own business. A wrong entry would emit a
#: plausible program that is not the controlled gate, so the contract's gate measures
#: every entry against the matrices the simulator executes rather than trusting the
#: table.
_LADDER_ANGLE: Mapping[str, float] = {
    "x": _PI,
    "y": _PI,
    "z": _PI,
    "h": _PI,
    "s": _PI / 2,
    "sdg": -_PI / 2,
    "t": _PI / 4,
    "tdg": -_PI / 4,
    "sx": _PI / 2,
    "sxdg": -_PI / 2,
    "cx": _PI,
    "cy": _PI,
    "cz": _PI,
    "ccx": _PI,
}


def controlled_instructions(
    schema: OperatorSchema,
    params: Mapping[str, Any],
    qubits: Sequence[int],
    controls: Sequence[int],
) -> tuple[Instruction, ...] | None:
    """Return the instructions of one gate with ``controls`` added to it.

    ``None`` means the opcode has no controlled form this IR can express, which is the
    same convention :func:`~flagquantum.core.inverse_operator` uses for an inverse the
    declaration does not determine: the caller refuses and reports which declaration it
    met rather than this function inventing a program.

    The returned instructions are new gates on the caller's own qubit labels, in
    application order, and they use only opcodes the registry declares. A parameter object
    is forwarded rather than rebuilt, so a trainable angle stays in the autograd graph,
    and one angle per batch entry scales element by element.

    Args:
        schema: The declaration of the gate being controlled.
        params: That gate's parameter mapping, named as the registry declares it.
        qubits: The gate's own qubit labels, in the registry's order.
        controls: The control labels to add. The caller checks that they are distinct and
            disjoint from ``qubits``; this function only needs them in order.

    Returns:
        The replacing instructions, ``()`` when the controlled gate is the identity, or
        ``None`` when the opcode is not controllable.

    Raises:
        ValueError: If the ladder this request needs is deeper than ``MAX_LADDER_LEVEL``.
            The bound is on the ladder rather than on the control count because a wider
            gate reaches a deeper ladder for the same count: ``control_ladder_level``
            returns ``n_controls + arity - 1``.
    """

    if schema.control == "not_available":
        return None
    # The identity is answered before the bound is consulted: it is controllable at any
    # depth precisely because it emits nothing, so a bound on emitted size cannot refuse
    # it.
    if schema.control == "identity":
        return ()
    level = control_ladder_level(schema.arity, len(controls))
    if level > MAX_LADDER_LEVEL:
        raise ValueError(
            f"{len(controls)} control qubit(s) on a {schema.arity}-qubit gate needs a "
            f"phase ladder of level {level}; the deepest supported ladder is "
            f"{MAX_LADDER_LEVEL}"
        )
    partner = CONTROL_PARTNERS.get(schema.opcode) if len(controls) == 1 else None
    if partner is not None:
        return (Instruction(partner, (*controls, *qubits), dict(params)),)
    # A control attaches to the gate's own last qubit as the ladder's target; the gate's
    # other qubits are partners that a control can be attached to directly, so they join
    # the ladder's control list. For a one-qubit gate this changes nothing.
    ladder = tuple(controls) + tuple(qubits[:-1])
    out: list[Instruction] = []
    _dispatch(schema, level, ladder, qubits[-1], params, out)
    return tuple(out)


def _dispatch(
    schema: OperatorSchema,
    level: int,
    controls: tuple[int, ...],
    target: int,
    params: Mapping[str, Any],
    out: list[Instruction],
) -> None:
    """Append the expansion the opcode's declared control rule names."""

    rule = schema.control
    if rule == "diagonal_ladder":
        _phase_ladder(level, controls, target, _ladder_angle(schema, params), out)
        return
    if rule == "x_conjugated_ladder":
        _flip(level, controls, target, _ladder_angle(schema, params), out)
        return
    if rule == "y_conjugated_ladder":
        _conjugated(
            level,
            controls,
            target,
            _BASIS_V_DAGGER,
            _BASIS_V,
            _ladder_angle(schema, params),
            out,
        )
        return
    if rule == "ry_conjugated_ladder":
        # H = RY(pi/4) Z RY(-pi/4) exactly, so this conjugation is a real rotation on the
        # target rather than a phase on a control, and it needs no correction.
        angle = _ladder_angle(schema, params)
        out.append(_angle_gate("ry", (target,), -_PI / 4))
        _phase_ladder(level, controls, target, angle, out)
        out.append(_angle_gate("ry", (target,), _PI / 4))
        return
    if rule == "rz_ladder":
        # RZ(t) = exp(-i t/2) P(t) and P is already diagonal, so the conjugation is the
        # identity and only the eigenvalue phase has to be restored.
        theta = params["theta"]
        _correction(level, controls, _scale_angle(theta, -_HALF), out)
        _phase_ladder(level, controls, target, theta, out)
        return
    if rule == "rx_ladder":
        theta = params["theta"]
        _correction(level, controls, _scale_angle(theta, -_HALF), out)
        _conjugated(level, controls, target, _BASIS_H, _BASIS_H, theta, out)
        return
    if rule == "ry_ladder":
        theta = params["theta"]
        _correction(level, controls, _scale_angle(theta, -_HALF), out)
        _conjugated(level, controls, target, _BASIS_V_DAGGER, _BASIS_V, theta, out)
        return
    if rule == "u_angle_ladder":
        # U3(theta, phi, lbd) = exp(i(phi+lbd)/2) RZ(phi) RY(theta) RZ(lbd), and with
        # RZ(t) = exp(-i t/2) P(t) and RY(t) = exp(-i t/2) V P(t) V^dagger this collapses
        # to U3 = exp(-i theta/2) P(phi) V P(theta) V^dagger P(lbd) exactly: the four
        # eigenvalue phases cancel pairwise and only one is left. U2(phi, lbd) is
        # U3(pi/2, phi, lbd), so both share this rule.
        theta = params.get("theta", _PI / 2)
        _correction(level, controls, _scale_angle(theta, -_HALF), out)
        _phase_ladder(level, controls, target, params["lbd"], out)
        _conjugated(level, controls, target, _BASIS_V_DAGGER, _BASIS_V, theta, out)
        _phase_ladder(level, controls, target, params["phi"], out)
        return
    if rule == "swap_ladder":
        # SWAP is CX . CX . CX with the pair alternating, so the gate's own two qubits are
        # the two ends of the flip and the added controls stay in place.
        swapped, other = controls[-1], target
        base = controls[:-1]
        for first, second in ((swapped, other), (other, swapped), (swapped, other)):
            _flip(level, (*base, first), second, _PI, out)
        return
    if rule == "cswap_ladder":
        # CSWAP is the same product with a Toffoli, so the last of the gate's own qubits is
        # its control and the pool of ladder controls grows by one.
        receiver, swapped, other = controls[-2], controls[-1], target
        base = controls[:-2]
        for first, second in ((swapped, other), (other, swapped), (swapped, other)):
            _flip(level, (*base, receiver, first), second, _PI, out)
        return
    if rule in {"rzz_ladder", "rxx_ladder", "ryy_ladder"}:
        _rotation_pair(rule, level, controls, target, params["theta"], out)
        return
    raise ValueError(f"opcode {schema.opcode!r} declares unknown control rule {rule!r}")


def _rotation_pair(
    rule: str,
    level: int,
    controls: tuple[int, ...],
    target: int,
    theta: Any,
    out: list[Instruction],
) -> None:
    """Append the controlled ``RZZ``, ``RXX``, or ``RYY`` that ``rule`` names.

    All three are one diagonal interaction in a different single-qubit basis:
    ``RZZ(t) = CX . RZ(t) . CX``, ``RXX`` is that conjugated by ``H`` on both qubits, and
    ``RYY`` the same by ``V = S H``. The interacting pair is the gate's own two qubits, so
    the flip between them costs one more control than the gate already needed, while the
    inner ``RZ`` sits one level lower.
    """

    left, right = controls[-1], target
    base = controls[:-1]
    if rule == "rxx_ladder":
        _basis_change((left, right), _BASIS_H, out)
    elif rule == "ryy_ladder":
        for qubit in (left, right):
            _basis_change((qubit,), _BASIS_V_DAGGER, out)
    _flip(level, (*base, left), right, _PI, out)
    _rz(level - 1, base, right, theta, out)
    _flip(level, (*base, left), right, _PI, out)
    if rule == "rxx_ladder":
        _basis_change((left, right), _BASIS_H, out)
    elif rule == "ryy_ladder":
        for qubit in (left, right):
            _basis_change((qubit,), _BASIS_V, out)


def _phase_ladder(
    level: int,
    controls: Sequence[int],
    target: int,
    theta: Any,
    out: list[Instruction],
) -> None:
    """Append ``C^level(P(theta))``: the one primitive every other rule is built on."""

    if level == 1:
        out.append(_angle_gate("cphase", (controls[0], target), theta))
        return
    rung = controls[-1]
    head = tuple(controls[:-1])
    for factor in (-_HALF, _HALF):
        _flip(level - 1, head, rung, _PI, out)
        out.append(_angle_gate("cphase", (rung, target), _scale_angle(theta, factor)))
    _phase_ladder(level - 1, head, target, _scale_angle(theta, _HALF), out)


def _flip(
    level: int, controls: Sequence[int], target: int, theta: Any, out: list[Instruction]
) -> None:
    """Append ``C^level(H P(theta) H)``, the non-diagonal rung of the ladder."""

    _conjugated(level, controls, target, _BASIS_H, _BASIS_H, theta, out)


def _conjugated(
    level: int,
    controls: Sequence[int],
    target: int,
    before: Sequence[str],
    after: Sequence[str],
    theta: Any,
    out: list[Instruction],
) -> None:
    """Append ``C^level(after . P(theta) . before)`` on one target.

    ``before`` and ``after`` are the two halves of a single-qubit basis change, in
    application order. The phase gate is unconditional on the target, so neither half
    needs a control of its own and the conjugation is exact.
    """

    _basis_change((target,), before, out)
    _phase_ladder(level, controls, target, theta, out)
    _basis_change((target,), after, out)


def _rz(
    level: int, controls: Sequence[int], target: int, theta: Any, out: list[Instruction]
) -> None:
    """Append ``C^level(RZ(theta))``, the diagonal rotation the interaction needs."""

    _correction(level, controls, _scale_angle(theta, -_HALF), out)
    _phase_ladder(level, controls, target, theta, out)


def _correction(
    level: int, controls: Sequence[int], delta: Any, out: list[Instruction]
) -> None:
    """Append the eigenvalue phase that a conjugation turned from global to relative.

    A single-qubit gate written as ``e^{i gamma} V P(t) V^dagger`` differs from
    ``V P(t) V^dagger`` by a phase that is global while the qubit is uncontrolled and
    becomes a relative phase between the branches of the first control once it is not.
    Restoring it is one phase gate when there is a single control and one ladder of the
    next level down otherwise. That ladder commutes with every ladder of the outer level,
    because both are diagonal once the outermost control is fixed, so the correction can
    be emitted anywhere in the sequence; it goes first for readability.
    """

    if level == 1:
        out.append(_angle_gate("phase", (controls[0],), delta))
        return
    _phase_ladder(level - 1, controls[:-1], controls[-1], delta, out)


def _basis_change(
    targets: Sequence[int], basis: Sequence[str], out: list[Instruction]
) -> None:
    """Append one single-qubit basis change to each of ``targets``."""

    for target in targets:
        for opcode in basis:
            out.append(Instruction(opcode, (target,)))


def _ladder_angle(schema: OperatorSchema, params: Mapping[str, Any]) -> Any:
    """The angle of the phase gate the opcode's rule is written around.

    A gate that declares its own angle supplies it -- ``phase(t)`` is ``P(t)`` -- and a
    parameter-free gate supplies the entry ``_LADDER_ANGLE`` records for it.
    """

    if schema.parameters == ("theta",):
        return params["theta"]
    return _LADDER_ANGLE[schema.opcode]


def _angle_gate(opcode: str, qubits: Sequence[int], theta: Any) -> Instruction:
    return Instruction(opcode, tuple(qubits), {"theta": theta})


def _scale_angle(value: Any, factor: float) -> Any:
    """Multiply one angle, or every angle of a per-batch sequence of angles."""

    if isinstance(value, (list, tuple)):
        return type(value)(_scale_angle(item, factor) for item in value)
    return value * factor
