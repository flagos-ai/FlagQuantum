"""Validate immutable channel operators without numerical SDK dependencies."""

from math import isfinite


def kraus_error(operators: object, arity: int) -> str | None:
    """Return a violated shape, finiteness or trace-preservation invariant."""
    if arity < 1:
        return "channel leaves require at least one qubit"
    if not isinstance(operators, tuple) or not operators:
        return "channel operators must be a nonempty immutable tuple"
    size = 2**arity
    for matrix in operators:
        if not isinstance(matrix, tuple) or len(matrix) != size:
            return "channel matrix dimension must equal 2**operand arity"
        if any(not isinstance(row, tuple) or len(row) != size for row in matrix):
            return "channel matrix rows must be immutable and square"
        if any(type(value) is not complex for row in matrix for value in row):
            return "channel matrix entries must be built-in complex values"
        if any(
            not isfinite(value.real) or not isfinite(value.imag)
            for row in matrix
            for value in row
        ):
            return "channel matrix entries must be finite"
    for left in range(size):
        for right in range(left, size):
            inner = sum(
                row[left].conjugate() * row[right]
                for matrix in operators
                for row in matrix
            )
            if (
                not isfinite(inner.real)
                or not isfinite(inner.imag)
                or abs(inner - int(left == right)) > 1e-10
            ):
                return "channel operators must preserve trace within absolute tolerance 1e-10"
    return None
