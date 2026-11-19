"""Validate owned explicit unitary data without retaining numerical SDK objects."""

from math import isfinite


def matrix_error(matrix: object, arity: int) -> str | None:
    """Return the failed unitary invariant, or None for canonical complex rows."""
    if arity < 1:
        return "explicit unitary leaves require at least one qubit"
    if not isinstance(matrix, tuple) or len(matrix) != 2**arity:
        return "matrix must have immutable rows and dimension 2**operand arity"
    size = len(matrix)
    if any(not isinstance(row, tuple) or len(row) != size for row in matrix):
        return "matrix rows must be immutable and square"
    if any(type(value) is not complex for row in matrix for value in row):
        return "matrix entries must be built-in complex values"
    if any(
        not isfinite(value.real) or not isfinite(value.imag)
        for row in matrix
        for value in row
    ):
        return "matrix entries must be finite"
    for left in range(size):
        for right in range(left, size):
            inner = sum(row[left].conjugate() * row[right] for row in matrix)
            if abs(inner - int(left == right)) > 1e-10:
                return "matrix must be unitary within absolute tolerance 1e-10"
    return None
