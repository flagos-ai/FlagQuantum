"""Exact compact products for narrow bounded Pauli evolutions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .syntax import Expression, Statement, Token, reject

if TYPE_CHECKING:
    from .adapter import _Adapt


Matrix = list[list[complex]]


def _product(left: Matrix, right: Matrix) -> Matrix:
    columns = tuple(zip(*right, strict=True))
    return [
        [sum(x * y for x, y in zip(row, col, strict=True)) for col in columns]
        for row in left
    ]


def _identity(size: int) -> Matrix:
    return [[complex(row == col) for col in range(size)] for row in range(size)]


def _matrix_power(matrix: Matrix, repetitions: int) -> Matrix:
    result = _identity(len(matrix))
    while repetitions:
        if repetitions % 2:
            result = _product(matrix, result)
        repetitions //= 2
        if repetitions:
            matrix = _product(matrix, matrix)
    return result


def lower_narrow_pauli_sequence(owner: _Adapt, body: tuple[Statement, ...]) -> bool:
    """Compose only a bounded, consecutive, same-view Suzuki schedule."""
    if (
        not 2 <= len(body) <= 256
        or not owner.pauli_bindings
        or any(
            statement.token.text != "suzuki_trotter"
            or len(statement.arguments) != 5
            or statement != Statement(statement.token, statement.arguments)
            or statement.arguments[0].children
            or not 1
            <= len(
                owner.pauli_bindings.get(
                    statement.arguments[0].token.text, statement.arguments[0]
                ).children
            )
            <= 256
            or statement.arguments[-1].children
            for statement in body
        )
    ):
        return False
    target = body[0].arguments[-1]
    if any(
        statement.arguments[-1].token.text != target.token.text for statement in body
    ):
        return False
    if owner.widths.get(target.token.text) not in {1, 2, 3}:
        return False
    view = owner.array_view(target)
    positions = {wire: index for index, (_, wire) in enumerate(view)}
    matrix = _identity(1 << len(view))

    def record(
        token: Token,
        operator: Expression,
        time: float,
        destination: Expression,
        order: int,
        repetitions: int,
    ) -> None:
        nonlocal matrix
        _, terms = owner.pauli_terms(token, operator, destination)
        step = _pauli_step(token, positions, terms, time / repetitions / order, order)
        matrix = _product(_matrix_power(step, repetitions), matrix)

    wires = tuple(reversed(view))
    for start in range(0, len(body), 64):
        matrix = _identity(1 << len(view))
        for statement in body[start : start + 64]:
            owner.suzuki_trotter(statement.token, statement.arguments, emit=record)
        matrix = _roundoff_unitary(matrix, body[start].token)
        entries = tuple(
            tuple((value.real, value.imag) for value in row) for row in matrix
        )
        owner.effect_op(
            "qreg.unitary",
            body[start].token,
            tuple(value for value, _ in wires),
            indices=tuple(index for _, index in wires),
            matrix=entries,
        )
    return True


def _roundoff_unitary(matrix: Matrix, token: Token) -> Matrix:
    adjoint = [
        list(row)
        for row in zip(*([v.conjugate() for v in row] for row in matrix), strict=True)
    ]
    gram = _product(adjoint, matrix)
    residual = max(
        abs(value - (row == col))
        for row, entries in enumerate(gram)
        for col, value in enumerate(entries)
    )
    if not math.isfinite(residual) or residual > 1e-11:
        reject(token, "pauli", "compact product exceeds the roundoff envelope")
    # One Newton polar correction is phase equivariant and changes only roundoff.
    correction = [
        [((3 if row == col else 0) - value) / 2 for col, value in enumerate(entries)]
        for row, entries in enumerate(gram)
    ]
    return _product(matrix, correction)


def _pauli_step(
    token: Token,
    positions: dict[int, int],
    terms: list[tuple[tuple[tuple[int, str], ...], float]],
    step: float,
    order: int,
) -> Matrix:
    size = 1 << len(positions)
    matrix = _identity(size)
    sequence = terms if order == 1 else [*terms, *reversed(terms)]
    for support, weight in sequence:
        angle = step * weight
        if not math.isfinite(2 * angle):
            reject(token, "angle", "compact Pauli angle must be finite")
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
