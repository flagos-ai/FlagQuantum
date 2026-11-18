# Copyright 2020 The Cirq Developers
# SPDX-License-Identifier: Apache-2.0
# Adapted from Cirq controlled_gate_decomposition.py, as called by Qualtran
# v0.7.0 PrepareUniformSuperposition through its Cirq controlled-H dependency.
# Changes: integer wire bindings, local native matrix instructions, validation,
# and NumPy's 2x2 spectral factorization instead of the optional SciPy Schur API.

"""Phase-exact recursive control synthesis with borrowed-wire restoration.

Only the supplied 2x2 local unitary is factored. No whole-block operator is
constructed or simulated. Matrix atoms remain local native IR leaves, as in
source decomposition; their count is not an expanded Clifford+T gate count.
"""

from collections.abc import Sequence
from math import cos, pi, sin

import numpy as np
from numpy.typing import NDArray

from ...core.ir import Instruction
from ..registers.bits.bitwise import _validate_wires

Matrix = NDArray[np.complex128]


def _rotation(axis: str, angle: float) -> Matrix:
    c, s = cos(angle / 2), sin(angle / 2)
    if axis == "y":
        return np.asarray([[c, -s], [s, c]], dtype=np.complex128)
    return np.diag(np.exp(np.asarray([-0.5j, 0.5j]) * angle))


def _power(matrix: Matrix, power: float) -> Matrix:
    values, vectors = np.linalg.eig(matrix)
    # Unitary matrices are normal. QR removes numerical nonorthogonality of
    # nearly repeated eigenvectors; the identity case is handled explicitly.
    if np.allclose(matrix, np.eye(2) * matrix[0, 0], atol=1e-14, rtol=0):
        return np.eye(2, dtype=np.complex128) * complex(matrix[0, 0]) ** power
    vectors, _ = np.linalg.qr(vectors)
    return np.asarray(
        sum(
            (np.outer(v, v.conj()) * complex(value) ** power)
            for value, v in zip(values, vectors.T, strict=True)
        ),
        dtype=np.complex128,
    )


def _abc(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix, float]:
    delta = float(np.angle(np.linalg.det(matrix)) * 0.5)
    alpha = float(np.angle(matrix[0, 0]) + np.angle(matrix[0, 1]) - 2 * delta)
    beta = float(np.angle(matrix[0, 0]) - np.angle(matrix[0, 1]))
    m00 = float(abs(matrix[0, 0]))
    if abs(m00 - 1) < 1e-9:
        m00 = 1
    theta = float(2 * np.arccos(m00))
    a = _rotation("z", -alpha) @ _rotation("y", -theta / 2)
    b = _rotation("y", theta / 2) @ _rotation("z", (alpha + beta) / 2)
    c = _rotation("z", (alpha - beta) / 2)
    return a, b, c, delta


def _phase(wire: int, exponent: float) -> Instruction:
    fixed = {1.0: "z", 0.5: "s", 1.5: "sdg", 0.25: "t", 1.75: "tdg"}
    name = fixed.get(exponent % 2)
    return (
        Instruction(name, (wire,))
        if name is not None
        else Instruction("phase", (wire,), {"theta": exponent * pi})
    )


def _single(matrix: Matrix, control: int, target: int) -> list[Instruction]:
    a, b, c, delta = _abc(matrix)
    candidates = [
        (_phase(control, delta / pi), np.diag([1, np.exp(1j * delta)])),
        (Instruction("any", (target,), matrix=c.tolist()), c),
        (Instruction("cx", (control, target)), None),
        (Instruction("any", (target,), matrix=b.tolist()), b),
        (Instruction("cx", (control, target)), None),
        (Instruction("any", (target,), matrix=a.tolist()), a),
    ]
    return [
        gate
        for gate, local in candidates
        if local is None or not np.allclose(local, np.eye(2))
    ]


def _congruent(c0: int, c1: int, target: int) -> list[Instruction]:
    return [
        Instruction("ry", (target,), {"theta": -pi / 4}),
        Instruction("cx", (c1, target)),
        Instruction("ry", (target,), {"theta": -pi / 4}),
        Instruction("cx", (c0, target)),
        Instruction("ry", (target,), {"theta": pi / 4}),
        Instruction("cx", (c1, target)),
        Instruction("ry", (target,), {"theta": pi / 4}),
    ]


def _multi_x(
    controls: tuple[int, ...], target: int, free: tuple[int, ...]
) -> list[Instruction]:
    m = len(controls)
    if m < 3:
        return [Instruction(("x", "cx", "ccx")[m], (*controls, target))]
    n = m + 1 + len(free)
    if n >= 2 * m - 1:
        blocks = [
            _congruent(controls[m - 2 - i], free[m - 4 - i], free[m - 3 - i])
            for i in range(m - 3)
        ]
        middle = [g for block in blocks for g in block]
        middle += _congruent(controls[0], controls[1], free[0])
        middle += [g for block in reversed(blocks) for g in block]
        first = Instruction("ccx", (controls[m - 1], free[m - 3], target))
        return [first, *middle, first, *middle]
    if free:
        m1 = n // 2
        part1 = _multi_x(controls[:m1], free[0], (*controls[m1:], target, *free[1:]))
        part2 = _multi_x((*controls[m1:], free[0]), target, (*controls[:m1], *free[1:]))
        return [*part1, *part2, *part1, *part2]
    return _recursive(
        np.asarray([[0, 1], [1, 0]], dtype=np.complex128), 1, controls, target, ()
    )


def _recursive(
    matrix: Matrix,
    power: float,
    controls: tuple[int, ...],
    target: int,
    free: tuple[int, ...],
) -> list[Instruction]:
    if len(controls) == 1:
        return _single(_power(matrix, power), controls[0], target)
    cnots = _multi_x(controls[:-1], controls[-1], (*free, target))
    return [
        *_single(_power(matrix, 0.5 * power), controls[-1], target),
        *cnots,
        *_single(_power(matrix, -0.5 * power), controls[-1], target),
        *cnots,
        *_recursive(matrix, 0.5 * power, controls[:-1], target, (controls[-1], *free)),
    ]


def borrowed_controlled_x_ir(
    controls: tuple[int, ...], target: int, borrowed: tuple[int, ...] = ()
) -> tuple[Instruction, ...]:
    """Emit source multi-X recursion, restoring arbitrary borrowed wire states."""
    _validate_wires((*controls, target, *borrowed))
    return tuple(_multi_x(controls, target, borrowed))


def controlled_unitary_ir(
    matrix: Sequence[Sequence[complex]] | Matrix,
    controls: tuple[int, ...],
    target: int,
) -> tuple[Instruction, ...]:
    """Control a concrete local 2x2 unitary, retaining its full scalar phase.

    Controls are positive and ordered. Negative activation wrapping belongs to
    the calling register primitive. This transformation owns no clean work or
    classical slots. Invalid shapes/nonunitary/nonfinite matrices raise ValueError.
    """
    _validate_wires((*controls, target))
    value = np.asarray(matrix, dtype=np.complex128)
    if (
        value.shape != (2, 2)
        or not np.isfinite(value).all()
        or not np.allclose(value.conj().T @ value, np.eye(2))
    ):
        raise ValueError("Controlled synthesis requires a finite 2x2 unitary")
    if not controls:
        return (Instruction("any", (target,), matrix=value.tolist()),)
    if len(controls) == 1:
        return tuple(_single(value, controls[0], target))
    if np.allclose(np.linalg.det(value), 1):
        a, b, c, _ = _abc(value)
        cnots = _multi_x(controls[:-1], target, (controls[-1],))
        return (
            *_single(c, controls[-1], target),
            *cnots,
            *_single(b, controls[-1], target),
            *cnots,
            *_single(a, controls[-1], target),
        )
    return tuple(_recursive(value, 1, controls, target, ()))
