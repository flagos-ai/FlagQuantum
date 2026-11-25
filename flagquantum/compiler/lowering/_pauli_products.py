"""Bounded compile-time products for at most three logical Pauli qubits.

These transformations compose static operators; they do not execute user states.
Callers own width, term, repetition and sequence-size admission.
"""

from __future__ import annotations

import math

Matrix = list[list[complex]]


def product(left: Matrix, right: Matrix) -> Matrix:
    columns = tuple(zip(*right, strict=True))
    return [
        [sum(x * y for x, y in zip(row, col, strict=True)) for col in columns]
        for row in left
    ]


def identity(size: int) -> Matrix:
    return [[complex(row == col) for col in range(size)] for row in range(size)]


def matrix_power(matrix: Matrix, repetitions: int) -> Matrix:
    result = identity(len(matrix))
    while repetitions:
        if repetitions % 2:
            result = product(matrix, result)
        repetitions //= 2
        if repetitions:
            matrix = product(matrix, matrix)
    return result


def roundoff_unitary(matrix: Matrix) -> Matrix:
    adjoint = [
        list(row)
        for row in zip(*([v.conjugate() for v in row] for row in matrix), strict=True)
    ]
    gram = product(adjoint, matrix)
    residual = max(
        abs(value - (row == col))
        for row, entries in enumerate(gram)
        for col, value in enumerate(entries)
    )
    if not math.isfinite(residual) or residual > 1e-11:
        raise ValueError("compact product exceeds the roundoff envelope")
    # One Newton polar correction is phase equivariant and changes only roundoff.
    correction = [
        [((3 if row == col else 0) - value) / 2 for col, value in enumerate(entries)]
        for row, entries in enumerate(gram)
    ]
    return product(matrix, correction)


def pauli_step(
    positions: dict[int, int],
    terms: list[tuple[tuple[tuple[int, str], ...], float]],
    step: float,
    order: int,
) -> Matrix:
    size = 1 << len(positions)
    matrix = identity(size)
    sequence = terms if order == 1 else [*terms, *reversed(terms)]
    for support, weight in sequence:
        angle = step * weight
        if not math.isfinite(2 * angle):
            raise ValueError("compact Pauli angle must be finite")
        x_mask = sum(
            1 << positions[wire] for wire, axis in support if axis in {"X", "Y"}
        )
        z_mask = sum(
            1 << positions[wire] for wire, axis in support if axis in {"Y", "Z"}
        )
        pauli_phase = 1j ** sum(axis == "Y" for _, axis in support)
        cosine, sine = math.cos(angle), -1j * math.sin(angle) * pauli_phase
        updated = []
        for row in range(size):
            source = row ^ x_mask
            signed_sine = -sine if (source & z_mask).bit_count() % 2 else sine
            updated.append(
                [
                    cosine * previous + signed_sine * paired
                    for previous, paired in zip(
                        matrix[row], matrix[source], strict=True
                    )
                ]
            )
        matrix = updated
    return matrix
