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
    qubits: tuple[int, ...],
    metadata: Mapping[str, Any],
    angle: Any,
) -> Instruction | None:
    """The pi/2 pulse as `opcode`, or None when the angle has become zero."""
    schema = get_operator_schema(opcode)
    parameters = schema.parameters if schema is not None else ()
    if not parameters:
        return Instruction(opcode, qubits, metadata=metadata)
    if _is_zero(angle):
        return None
    return Instruction(
        opcode, qubits, params=dict.fromkeys(parameters, angle), metadata=metadata
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
    qubits: tuple[int, ...],
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
                leaf_opcode, qubits=qubits, metadata=metadata, angle=_HALF_PI
            )
            if pulse is not None:
                replacement.append(pulse)
        elif not _is_zero(angle):
            replacement.append(
                Instruction(
                    z_rotation, qubits, params={"theta": angle}, metadata=metadata
                )
            )
    return tuple(replacement)


def synthesize_one_qubit(
    instruction: Instruction,
    *,
    z_rotation: str,
    pulse_opcode: str = "sx",
) -> tuple[Instruction, ...] | None:
    """Return `instruction` as a z-rotation and pi/2 pulses, or None.

    `z_rotation` is the z-rotation opcode the target publishes and `pulse_opcode`
    the opcode it publishes for a pi/2 rotation about x. None is returned when
    nothing here applies: a multi-qubit or non-unitary instruction, an instruction
    the operator schema does not describe, or an instruction that is already that
    same z-rotation, whose shortest form only the caller's native gate set can
    choose. None is also returned when `z_rotation` or `pulse_opcode` is not a
    basis this module can synthesize over.
    """
    if not _is_supported_basis(z_rotation=z_rotation, pulse_opcode=pulse_opcode):
        return None
    opcode = canonical_opcode(instruction.name)
    if opcode == z_rotation:
        return None
    schema = get_operator_schema(opcode)
    if schema is None or not schema.unitary or schema.arity != 1:
        return None
    angles = _FIXED_EULER_ANGLES.get(opcode)
    if angles is None:
        source = _PARAMETERIZED_EULER_ANGLES.get(opcode)
        if source is None:
            return None
        angles = source(instruction)
    return _emit_leaves(
        _leaves(*angles, z_rotation=z_rotation, pulse_opcode=pulse_opcode),
        qubits=instruction.wires,
        metadata=instruction.metadata,
        z_rotation=z_rotation,
        pulse_opcode=pulse_opcode,
    )


def synthesize_one_qubit_matrix(
    matrix: Any,
    *,
    qubit: int,
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
    `qubit` is not a non-negative integer. An empty tuple, for the identity, is a
    correct answer.
    """
    if not _is_supported_basis(z_rotation=z_rotation, pulse_opcode=pulse_opcode):
        return None
    if isinstance(qubit, bool) or not isinstance(qubit, int) or qubit < 0:
        raise ValueError("qubit must be a non-negative integer")
    target = _as_complex_matrix(matrix, 2, what="matrix")
    if not _is_unitary(target):
        raise ValueError("matrix must be a 2x2 unitary")
    return _emit_leaves(
        _leaves(*_zyz_angles(target), z_rotation=z_rotation, pulse_opcode=pulse_opcode),
        qubits=(qubit,),
        metadata={} if metadata is None else metadata,
        z_rotation=z_rotation,
        pulse_opcode=pulse_opcode,
    )


__all__ = ("synthesize_one_qubit", "synthesize_one_qubit_matrix")
