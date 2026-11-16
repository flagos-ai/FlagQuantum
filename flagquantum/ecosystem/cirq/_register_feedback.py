"""Translate validated native classical dataflow to Cirq conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.classical_dataflow import parse_classical_xor
from ...core.ir import Instruction

if TYPE_CHECKING:
    from sympy.logic.boolalg import Boolean

_Clauses = tuple[tuple[tuple[int, int], ...], ...]


class RegisterFeedback:
    """Track each slot as a measurement key or a Boolean expression over keys."""

    def __init__(self) -> None:
        self._slots: dict[int, str | Boolean] = {}

    def record_measurement(self, slot: int, key: str) -> None:
        self._slots[slot] = key

    def _expression(self, slot: int) -> Boolean:
        from sympy import Eq, Symbol

        value = self._slots[slot]
        if isinstance(value, str):
            return Eq(Symbol(value), 1)
        return value

    def _condition(self, clauses: _Clauses) -> Boolean:
        from sympy import And, Not, Or

        return Or(
            *(
                And(
                    *(
                        self._expression(slot) if bit else Not(self._expression(slot))
                        for slot, bit in clause
                    )
                )
                for clause in clauses
            )
        )

    def assign(self, instruction: Instruction, clauses: _Clauses) -> None:
        from sympy import ITE, Xor

        assignment = parse_classical_xor(instruction)
        result = Xor(
            assignment.constant,
            *(self._expression(slot) for slot in assignment.sources),
        )
        if clauses:
            result = ITE(
                self._condition(clauses), result, self._expression(assignment.target)
            )
        self._slots[assignment.target] = result

    def condition_operation(self, cirq: Any, operation: Any, clauses: _Clauses) -> Any:
        """Attach a source-style key condition or the saved Boolean formula."""
        if len(clauses) == len(clauses[0]) == 1:
            slot, bit = clauses[0][0]
            value = self._slots[slot]
            if bit == 1 and isinstance(value, str):
                return operation.with_classical_controls(value)
        return operation.with_classical_controls(
            cirq.SympyCondition(self._condition(clauses))
        )
