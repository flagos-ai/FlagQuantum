"""One-qubit Euler-angle synthesis into a z-rotation plus a pi/2 pulse basis.

`gates.native` publishes the single-qubit gates a target can execute. Every
single-qubit unitary in FlagQuantum IR is `U3(theta, phi, lam)` up to a global
phase, and

    ``U3(theta, phi, lam) == RZ(lam) SX RZ(theta - pi) SX RZ(phi - pi)``

also up to a global phase. `RX(pi/2)` *is* `SX` exactly, so a basis that
publishes either one reaches the whole one-qubit group with a z-rotation and that
single pulse. `sxdg` is never needed because `SXDG = RZ(-pi) SX RZ(-pi)`.

The angle triples of the parameter-free gates are tabulated rather than solved
numerically, so they stay exact and bit-reproducible. Parameterized gates carry
the caller's own parameter objects through, so a trainable rotation stays
trainable, and a trainable angle never selects a short branch.

**Global phase is dropped.** `RZ` and `SX` generate a determinant group whose
phases are multiples of `i`, so `H`, `S`, `T`, and the rest cannot be matched
exactly, and FlagQuantum IR has no field in which to record the difference. The
caller's exact hand-written rewrites stay ahead of this module, so the phase is
only lost where the target's basis leaves no exact form at all. A global phase
is unobservable in every expectation value, but a consumer that compares raw
statevectors rather than measurement statistics has to know this.

Two entry points emit the same form. `synthesize_one_qubit` starts from an
instruction and reads its angles, so a trainable rotation stays in the autograd
graph. `synthesize_one_qubit_matrix` starts from a 2x2 matrix, which is what a
caller holding a local factor of a larger decomposition has; see
`two_qubit_synthesis` for the two-qubit case. The matrix entry point recovers the
same `U3` triple by closed form, which costs one global phase more precision than
the tabulated fixed gates but needs no table.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from ..core.ir import Instruction
from ..core.operator_schema import canonical_opcode, get_operator_schema
from .pipeline import _is_zero

#: A 2x2 matrix of Python complex numbers, rows first.
Matrix = list[list[complex]]

#: A matrix is accepted as a unitary within this absolute tolerance.
_UNITARY_ATOL = 1.0e-9

_HALF_PI = 0.5 * math.pi
_QUARTER_PI = 0.25 * math.pi
_TWO_PI = 2.0 * math.pi

# The opcodes that can carry the z-rotation of the general form, most preferred
# first. They differ only by a global phase, which this module drops anyway, so
# the order is a preference rather than a semantic choice.
Z_ROTATION_OPCODES: tuple[str, ...] = ("rz", "phase", "u1")

# The opcodes that can carry the pi/2 pulse of the general form, most preferred
# first. `sx` is parameter-free; `rx` is the same pulse spelled as a rotation.
HALF_PI_PULSE_OPCODES: tuple[str, ...] = ("sx", "rx")

# Below this magnitude a matrix entry carries no usable phase.
_PHASE_EPS = 1.0e-12

# ZYZ angle triples `(theta, phi, lam)` for the parameter-free single-qubit
# unitary opcodes of `OPERATOR_SCHEMAS`, read as
# `opcode == exp(1j * phase) * U3(theta, phi, lam)`.
_FIXED_EULER_ANGLES: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "i": (0.0, 0.0, 0.0),
        "x": (math.pi, 0.0, math.pi),
        "y": (math.pi, _HALF_PI, _HALF_PI),
        "z": (0.0, 0.0, math.pi),
        "h": (_HALF_PI, 0.0, math.pi),
        "s": (0.0, 0.0, _HALF_PI),
        "sdg": (0.0, 0.0, -_HALF_PI),
        "t": (0.0, 0.0, _QUARTER_PI),
        "tdg": (0.0, 0.0, -_QUARTER_PI),
        "sx": (_HALF_PI, -_HALF_PI, _HALF_PI),
        "sxdg": (-_HALF_PI, -_HALF_PI, _HALF_PI),
    }
)


def _rx_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    # `RX(theta) = RZ(-pi/2) RY(theta) RZ(pi/2)`.
    return (instruction.params["theta"], -_HALF_PI, _HALF_PI)


def _ry_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    return (instruction.params["theta"], 0.0, 0.0)


def _z_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    return (0.0, 0.0, instruction.params["theta"])


def _u2_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    return (_HALF_PI, instruction.params["phi"], instruction.params["lbd"])


def _u3_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    return (
        instruction.params["theta"],
        instruction.params["phi"],
        instruction.params["lbd"],
    )


# ZYZ angle triples for the opcodes that read their angles from IR parameters.
# `rz`, `phase`, and `u1` differ only by a global phase, so all three delegate to
# whichever of them the target publishes.
_PARAMETERIZED_EULER_ANGLES: Mapping[
    str, Callable[[Instruction], tuple[Any, Any, Any]]
] = MappingProxyType(
    {
        "rx": _rx_angles,
        "ry": _ry_angles,
        "rz": _z_angles,
        "phase": _z_angles,
        "u1": _z_angles,
        "u2": _u2_angles,
        "u3": _u3_angles,
    }
)


def _polar_angle(value: Any) -> float | None:
    """Return `value` as a float for branch selection, or None when it has none.

    A trainable value never selects a branch, for the same reason
    `pipeline._is_zero` refuses to call a trainable angle zero: the short forms
    omit `theta` entirely, which would silently detach a rotation initialized at
    zero from the autograd graph.
    """
    if bool(getattr(value, "requires_grad", False)):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (RuntimeError, ValueError):
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _pulse_leaf(
    opcode: str,
    *,
    wires: tuple[int, ...],
    metadata: Mapping[str, Any],
    angle: Any,
) -> Instruction | None:
    """The pi/2 pulse as `opcode`, or None when the angle has become zero."""
    schema = get_operator_schema(opcode)
    parameters = schema.parameters if schema is not None else ()
    if not parameters:
        return Instruction(opcode, wires, metadata=metadata)
    if _is_zero(angle):
        return None
    return Instruction(
        opcode, wires, params=dict.fromkeys(parameters, angle), metadata=metadata
    )


def _leaves(
    theta: Any,
    phi: Any,
    lam: Any,
    *,
    z_rotation: str,
    pulse_opcode: str,
) -> tuple[tuple[str, Any], ...]:
    """Return the gates of one Euler triple, in application order.

    The general form is `RZ(lam) PULSE RZ(theta - pi) PULSE RZ(phi - pi)`, which
    is `U3(theta, phi, lam)` up to a global phase: the sequence a ZSX decomposer
    emits, with `SX` or an equivalent `RX(pi/2)` pulse. A polar angle of zero,
    `+/-pi/2`, or `+/-pi` collapses to fewer gates. Each entry is
    `(opcode, angle)`, where a `None` angle marks the pulse. The angles are the
    caller's own objects, so a leaf stays trainable when the source rotation was.
    """
    polar = _polar_angle(theta)
    if polar == 0.0:
        return ((z_rotation, phi + lam),)
    if polar == _HALF_PI:
        return (
            (z_rotation, lam - _HALF_PI),
            (pulse_opcode, None),
            (z_rotation, phi + _HALF_PI),
        )
    if polar == -_HALF_PI:
        return (
            (z_rotation, lam + _HALF_PI),
            (pulse_opcode, None),
            (z_rotation, phi - _HALF_PI),
        )
    if polar is not None and abs(polar) == math.pi:
        return (
            (pulse_opcode, None),
            (pulse_opcode, None),
            (z_rotation, phi + math.pi - lam),
        )
    return (
        (z_rotation, lam),
        (pulse_opcode, None),
        (z_rotation, theta - math.pi),
        (pulse_opcode, None),
        (z_rotation, phi - math.pi),
    )


def _principal(angle: float) -> float:
    """Reduce an angle in radians to `(-pi, pi]`."""
    reduced = math.remainder(angle, _TWO_PI)
    return reduced + _TWO_PI if reduced <= -math.pi else reduced


def _zyz_angles(matrix: Matrix) -> tuple[float, float, float]:
    """Return the `U3(theta, phi, lam)` angles of a 2x2 unitary up to a phase.

    `U3(theta, phi, lam)` is
    `[[cos(t/2), -exp(1j*lam) sin(t/2)], [exp(1j*phi) sin(t/2),
    exp(1j*(phi+lam)) cos(t/2)]]`, so the polar angle comes from the two column
    magnitudes and the azimuths from two phase differences. Two shapes are
    degenerate and read their angles off the surviving entries instead: a
    diagonal matrix has no azimuth to read at all, and an anti-diagonal one has
    no diagonal entry left. `phi` and `lam` leave `U3` exactly invariant under a
    `2*pi` shift, so both are reduced to `(-pi, pi]`; that is what lets the short
    forms in `_leaves` collapse a matching matrix to a single pulse rather than
    emitting a redundant `RZ(2*pi)`.
    """
    top_left, top_right = matrix[0][0], matrix[0][1]
    bottom_left, bottom_right = matrix[1][0], matrix[1][1]
    if abs(bottom_left) <= _PHASE_EPS:
        return 0.0, 0.0, _principal(cmath.phase(bottom_right) - cmath.phase(top_left))
    if abs(top_left) <= _PHASE_EPS:
        return (
            math.pi,
            _principal(cmath.phase(bottom_left)),
            _principal(cmath.phase(top_right) - math.pi),
        )
    return (
        2.0 * math.atan2(abs(bottom_left), abs(top_left)),
        _principal(cmath.phase(bottom_left) - cmath.phase(top_left)),
        _principal(cmath.phase(bottom_right) - cmath.phase(bottom_left)),
    )


def _rows(raw: Any, size: int, *, what: str, where: str) -> list[Any]:
    """Return `raw` as a list of `size` rows, or raise ValueError.

    Anything sized and iterable is accepted, and anything exposing `tolist` is
    converted first, so an array or a tensor can be handed in without this
    package importing the library that produced it. `where` names the position
    for the error message, which is how the two levels below share this.
    """
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"{where} must be a sequence of {size} entries")
    try:
        rows = list(raw)
    except TypeError as error:
        raise ValueError(f"{where} must be a sequence of {size} entries") from error
    if len(rows) != size:
        raise ValueError(f"{where} must have exactly {size} entries, got {len(rows)}")
    return rows


def _as_complex_matrix(raw: Any, size: int, *, what: str) -> Matrix:
    """Return `raw` as a `size` x `size` matrix of Python complex numbers."""
    matrix: Matrix = []
    for index, row in enumerate(_rows(raw, size, what=what, where=what)):
        entries: list[complex] = []
        for item in _rows(row, size, what=what, where=f"{what} row {index}"):
            try:
                entries.append(complex(item))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{what} entry {index} is not a number: {item!r}"
                ) from error
        matrix.append(entries)
    return matrix


def _is_unitary(matrix: Matrix) -> bool:
    """Whether a 2x2 matrix is unitary to `_UNITARY_ATOL`."""
    (a, b), (c, d) = matrix
    return all(
        abs(value - expected) <= _UNITARY_ATOL
        for value, expected in (
            (a.conjugate() * a + c.conjugate() * c, 1.0),
            (b.conjugate() * b + d.conjugate() * d, 1.0),
            (a.conjugate() * b + c.conjugate() * d, 0.0),
        )
    )


def _is_supported_basis(*, z_rotation: str, pulse_opcode: str) -> bool:
    """Whether a basis can carry the general form of this module."""
    return z_rotation in Z_ROTATION_OPCODES and pulse_opcode in HALF_PI_PULSE_OPCODES


def _emit_leaves(
    leaves: tuple[tuple[str, Any], ...],
    *,
    wires: tuple[int, ...],
    metadata: Mapping[str, Any],
    z_rotation: str,
    pulse_opcode: str,
) -> tuple[Instruction, ...]:
    """Turn the `(opcode, angle)` leaves of one Euler triple into instructions.

    A zero z-angle is dropped: the short forms in `_leaves` produce them on
    purpose, and `RZ(0)` is the identity.
    """
    replacement: list[Instruction] = []
    for leaf_opcode, angle in leaves:
        if leaf_opcode == pulse_opcode:
            pulse = _pulse_leaf(
                leaf_opcode, wires=wires, metadata=metadata, angle=_HALF_PI
            )
            if pulse is not None:
                replacement.append(pulse)
        elif not _is_zero(angle):
            replacement.append(
                Instruction(
                    z_rotation, wires, params={"theta": angle}, metadata=metadata
                )
            )
    return tuple(replacement)


def canonical_euler_angles(instruction: Instruction) -> tuple[Any, Any, Any] | None:
    """Return the `(theta, phi, lam)` triple this module tabulates for a gate.

    The triple is read from `_FIXED_EULER_ANGLES` or `_PARAMETERIZED_EULER_ANGLES`
    for the instruction's canonical opcode, so `instruction` equals
    `exp(1j * phase) * U3(theta, phi, lam)` for some phase. None means the
    instruction is not a declared single-qubit unitary opcode at all, which is
    the same refusal `synthesize_one_qubit` gives for an unknown opcode, a
    non-unitary one, or one of another arity.
    """

    opcode = canonical_opcode(instruction.name)
    schema = get_operator_schema(opcode)
    if schema is None or not schema.unitary or schema.arity != 1:
        return None
    fixed = _FIXED_EULER_ANGLES.get(opcode)
    if fixed is not None:
        return fixed
    source = _PARAMETERIZED_EULER_ANGLES.get(opcode)
    if source is None:
        return None
    return source(instruction)


def is_diagonal_one_qubit(instruction: Instruction) -> bool:
    """Report whether one instruction is diagonal in the computational basis.

    `U3(theta, phi, lam)` is
    `[[cos(t/2), -exp(1j*lam) sin(t/2)], [exp(1j*phi) sin(t/2),
    exp(1j*(phi+lam)) cos(t/2)]]`, so both off-diagonal entries carry the same
    factor `sin(theta / 2)`: the matrix is diagonal exactly when that factor
    vanishes, whatever `phi` and `lam` are, and it is then
    `diag(1, exp(1j * (phi + lam)))`. The polar angle alone decides it, and the
    tabulated triple is the only place this module states that angle, so no
    second opcode table is introduced here.

    The test is exact `== 0.0` on the polar angle, matching `_leaves`, which
    selects its short forms the same way: `U3` is only diagonal for a polar
    angle of `2*pi*k`, this module has no exactness contract for angle
    arithmetic, and a tolerance would call `rx(1e-13)` diagonal on the strength
    of a rounding.

    A trainable value is handled by `_polar_angle`, which answers None rather
    than a float, and the answer differs by opcode for the right reason. For
    `rx`, `ry` and `u3` the polar angle *is* the parameter, so a trainable one
    never selects the zero branch and a rotation initialized at zero is not
    reported diagonal -- the same refusal `_leaves` makes, for the same reason.
    For `rz`, `phase` and `u1` the tabulated triple's polar angle is the literal
    `0.0` because those opcodes are diagonal for *every* value of their
    parameter, so a trainable one is reported diagonal and that is not a
    detachment: the phase it carries is what a measurement cannot see.
    """

    angles = canonical_euler_angles(instruction)
    if angles is None:
        return False
    return _polar_angle(angles[0]) == 0.0


#: Above this magnitude an angle is not folded at all.
#:
#: A fold is only meaningful while a float64 can sit within the tolerance of a
#: multiple of `4*pi` without *being* that multiple: past `4*pi * 512` the spacing
#: of float64 exceeds the tolerance, so the only angle whose exact remainder
#: vanishes is one already equal to `4*pi * k`, and such an angle is an arbitrary
#: rotation rather than the identity. Folding there would compare rounded integers
#: rather than angles and could only invent a removal.
_MODULAR_FOLD_LIMIT = 2048.0 * math.pi

#: The angle at which the `U3` family returns to the identity *exactly*, rather
#: than to minus the identity.
#:
#: `U3(2*pi, 0, 0)` is `-I` and `U3(4*pi, 0, 0)` is `I`, so a rule that folded its
#: angles modulo `2*pi` would call `rz(2*pi)` the identity and drop a sign the IR
#: has nowhere to record: `CircuitIR` carries no global-phase field, so the
#: statevector *is* the program's full output and a `-I` that disappears is a
#: program that changed. Folding modulo `4*pi` costs the removals that are only
#: correct up to a sign, which are named in `is_identity_one_qubit`, and in
#: exchange every removal this rule makes leaves the statevector identical.
_IDENTITY_PERIOD = 2.0 * _TWO_PI


def _is_identity_angle(value: Any, atol: float = 1e-12) -> bool:
    """Report whether one tabulated angle is an integer multiple of `4*pi`.

    `math.remainder` returns `value - n * 4*pi` exactly, for the `n` nearest
    `value / (4*pi)`, so the residue carries no rounding of its own and the same
    `atol` `pipeline._is_zero` applies to a raw angle applies unchanged to the
    folded one. A trainable value, a non-scalar, and a value outside the fold
    limit all answer False, which keeps the removal fail-closed.
    """

    polar = _polar_angle(value)
    if polar is None or abs(polar) > _MODULAR_FOLD_LIMIT:
        return False
    return _is_zero(math.remainder(polar, _IDENTITY_PERIOD), atol=atol)


def is_identity_one_qubit(instruction: Instruction) -> bool:
    """Report whether one instruction is exactly the identity operator.

    `U3(theta, phi, lam)` is `[[cos(t/2), -exp(1j*lam) sin(t/2)],
    [exp(1j*phi) sin(t/2), exp(1j*(phi+lam)) cos(t/2)]]` for `t = theta`, so its
    off-diagonal entries vanish exactly where `sin(t/2)` does, and at those points
    it is `cos(t/2) * diag(1, exp(1j*(phi+lam)))`. The matrix of `U3` alone is
    therefore the identity exactly when `t` is a multiple of `4*pi` -- so that the
    first diagonal entry is `+1` and not `-1` -- and `phi + lam` is a multiple of
    `2*pi`. The declared triples are read through `canonical_euler_angles`, which
    is the only place this module states those angles, so no opcode table and no
    second angle table live here, an opcode the schema does not describe is never
    removed, and a new opcode that tabulates `(0, 0, 0)` is classified rather than
    declined.

    The fold applied to `phi + lam` is `4*pi` as well, which is deliberately
    stricter than `U3` alone requires, and the reason is that a triple does not
    always name its opcode's matrix. `rz(theta)`, `phase(theta)` and `u1(theta)`
    all tabulate `(0, 0, theta)`, and that triple is the matrix of `phase(theta)`;
    the matrix of `rz(theta)` is `exp(-1j*theta/2)` times it. So `phase(2*pi)` *is*
    the identity while `rz(2*pi)` is minus the identity, and no rule that reads
    only the triple can tell them apart. FlagQuantum IR has no field in which to
    record a global phase, so `optimize` has to leave the statevector identical --
    a property `tests/unit/test_compilation_inverse_cancellation.py` asserts over
    random programs -- and this module resolves the collision by declining all
    three. Folding `phi + lam` by `2*pi` while folding `t` by `4*pi` would remove
    `phase(2*pi)`, `u1(2*pi)` and `rz(2*pi)` alike and change the statevector of
    the last.

    Three reaches follow from the fold, and all three are refusals rather than
    mistakes. `phase(2*pi)` and `u1(2*pi)` are the identity and are declined, for
    the collision above. `u3(4*pi, phi, lam)` is declined when `phi + lam` is an
    even multiple of `pi` that is not a multiple of `4*pi` -- `phi + lam = 2*pi`,
    where the matrix *is* the identity -- even though `u3`'s triple names its own
    matrix, because one fold is applied to the sum. And every two-wire rotation is
    decided by the multi-wire branch of `remove_identity_gates`, not here, so this
    module's whole reach is single-wire.

    Only exact integer multiples are considered, within `_is_zero`'s own `1e-12`,
    rather than a fidelity cutoff of the kind a target error rate would supply: a
    false yes changes the program, and an angle merely near a multiple is a
    rotation a target may well distinguish. `None` from `_polar_angle` on any of
    the three angles -- a trainable one most of all -- refuses, so a rotation
    initialized at zero is never detached from the autograd graph.

    False for an instruction of another arity or one that is not a declared
    unitary, including `measure`, `reset`, `barrier`, and an operation carrying
    its own matrix. The matrix is refused rather than read: a caller who attached
    one has said the opcode's name no longer describes the operator, so the angle
    triple below is not evidence about it and this module declines rather than
    guesses. `diagonal_before_measure._is_diagonal` refuses on the same ground.
    """

    if len(instruction.wires) != 1 or instruction.matrix is not None:
        return False
    angles = canonical_euler_angles(instruction)
    if angles is None:
        return False
    theta, phi, lam = angles
    if not _is_identity_angle(theta):
        return False
    phase_angle = _polar_angle(phi)
    z_angle = _polar_angle(lam)
    if phase_angle is None or z_angle is None:
        return False
    return _is_identity_angle(phase_angle + z_angle)


def synthesize_one_qubit(
    instruction: Instruction,
    *,
    z_rotation: str,
    pulse_opcode: str = "sx",
) -> tuple[Instruction, ...] | None:
    """Return `instruction` as a z-rotation and pi/2 pulses, or None.

    `z_rotation` is the z-rotation opcode the target publishes and `pulse_opcode`
    the opcode it publishes for a pi/2 rotation about x. None is returned when
    nothing here applies: a multi-wire or non-unitary instruction, an instruction
    the operator schema does not describe, or an instruction that is already that
    same z-rotation, whose shortest form only the caller's native gate set can
    choose. None is also returned when `z_rotation` or `pulse_opcode` is not a
    basis this module can synthesize over.
    """
    if not _is_supported_basis(z_rotation=z_rotation, pulse_opcode=pulse_opcode):
        return None
    if canonical_opcode(instruction.name) == z_rotation:
        return None
    angles = canonical_euler_angles(instruction)
    if angles is None:
        return None
    return _emit_leaves(
        _leaves(*angles, z_rotation=z_rotation, pulse_opcode=pulse_opcode),
        wires=instruction.wires,
        metadata=instruction.metadata,
        z_rotation=z_rotation,
        pulse_opcode=pulse_opcode,
    )


def synthesize_one_qubit_matrix(
    matrix: Any,
    *,
    wire: int,
    z_rotation: str,
    pulse_opcode: str = "sx",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Instruction, ...] | None:
    """Return a 2x2 unitary as a z-rotation and pi/2 pulses, or None.

    The matrix entry point behind `synthesize_one_qubit`, for a caller that
    holds a local factor of a larger decomposition rather than an instruction.
    `matrix` is any sequence of two rows of two numbers. The result carries the
    same z-rotation plus pulse form and is equal to `matrix` up to one global
    phase, which FlagQuantum IR has no field to record.

    `metadata` is copied onto every emitted leaf, so a caller that holds the
    source instruction can keep its annotations.

    Returns None when `z_rotation` or `pulse_opcode` is not a basis this module
    can synthesize over. Raises ValueError when `matrix` is not a 2x2 unitary or
    `wire` is not a non-negative integer. An empty tuple, for the identity, is a
    correct answer.
    """
    if not _is_supported_basis(z_rotation=z_rotation, pulse_opcode=pulse_opcode):
        return None
    if isinstance(wire, bool) or not isinstance(wire, int) or wire < 0:
        raise ValueError("wire must be a non-negative integer")
    target = _as_complex_matrix(matrix, 2, what="matrix")
    if not _is_unitary(target):
        raise ValueError("matrix must be a 2x2 unitary")
    return _emit_leaves(
        _leaves(*_zyz_angles(target), z_rotation=z_rotation, pulse_opcode=pulse_opcode),
        wires=(wire,),
        metadata={} if metadata is None else metadata,
        z_rotation=z_rotation,
        pulse_opcode=pulse_opcode,
    )


__all__ = (
    "canonical_euler_angles",
    "is_diagonal_one_qubit",
    "synthesize_one_qubit",
    "synthesize_one_qubit_matrix",
)
