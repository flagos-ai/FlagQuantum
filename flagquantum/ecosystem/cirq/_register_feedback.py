"""Translate validated native conditions against single-bit measurement keys."""

from collections.abc import Mapping
from typing import Any


def with_feedback(
    cirq: Any,
    operation: Any,
    clauses: tuple[tuple[tuple[int, int], ...], ...],
    keys: Mapping[int, str],
) -> Any:
    """Attach a source-style key condition or an exact Boolean formula."""
    if len(clauses) == len(clauses[0]) == 1 and clauses[0][0][1] == 1:
        return operation.with_classical_controls(keys[clauses[0][0][0]])

    from sympy import And, Eq, Or, Symbol

    expression = Or(
        *(
            And(*(Eq(Symbol(keys[bit]), value) for bit, value in clause))
            for clause in clauses
        )
    )
    return operation.with_classical_controls(cirq.SympyCondition(expression))
