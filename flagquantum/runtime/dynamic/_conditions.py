"""Shared classical-condition helpers for dynamic circuit layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.classical_control import (
    is_classical_control_instruction,
    validate_classical_control,
)
from ...core.ir import Instruction
from ...errors import ValidationError

if TYPE_CHECKING:
    from .circuit import DynamicCircuit


def instruction_conditions(
    instruction: Instruction,
) -> tuple[tuple[int, int], ...]:
    if "condition_clauses" in instruction.metadata:
        raise ValueError(
            "this path evaluates one conjunction of classical conditions; a "
            "disjunction of them is evaluated by run_dynamic, "
            "DynamicCircuit.density_matrix, and the OpenQASM 3 export instead"
        )
    if "conditions" in instruction.metadata:
        return tuple(
            (int(bit), int(value)) for bit, value in instruction.metadata["conditions"]
        )
    return ()


def instruction_condition_clauses(
    instruction: Instruction,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if (
        "conditions" in instruction.metadata
        and "condition_clauses" in instruction.metadata
    ):
        raise ValueError(
            "instruction cannot define both conditions and condition_clauses"
        )
    if "condition_clauses" in instruction.metadata:
        try:
            return tuple(
                tuple((int(bit), int(value)) for bit, value in clause)
                for clause in instruction.metadata["condition_clauses"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "condition_clauses must contain integer bit/value pairs"
            ) from exc
    conditions = instruction_conditions(instruction)
    return (conditions,) if conditions else ()


def validate_condition_clauses(
    clauses: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Check the per-clause shape that every condition spelling shares.

    A clause is one conjunction of bit tests, and two rules belong to the clause
    rather than to the way the clauses are combined: a bit appears once and is
    tested against zero or one, and the pairs are listed in ascending bit order so
    that one conjunction has one spelling. The executor and the exporters all read
    a clause through :func:`instruction_condition_clauses`, so a clause that
    breaks either rule is refused here rather than by whichever reader reached it
    first.
    """

    if any(
        bit < 0 or expected not in {0, 1}
        for clause in clauses
        for bit, expected in clause
    ) or any(len({bit for bit, _ in clause}) != len(clause) for clause in clauses):
        raise ValidationError(
            "classical condition clauses require unique bits and binary values"
        )
    if any(tuple(sorted(clause)) != clause for clause in clauses):
        raise ValidationError("classical condition clauses must be canonically ordered")
    return clauses


def validate_disjunctive_clauses(
    clauses: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Check the rules a disjunction of clauses carries beyond one clause.

    The three rules make one disjunction have one spelling, and each of them also
    reports a condition whose author contradicted themselves: a repeated clause
    and a clause contained in a later one both describe logic that can never be
    reached, so they are refused rather than tidied away, because tidying would
    hide the contradiction and keeping them would charge the executor for a branch
    no trajectory can take.
    """

    validate_condition_clauses(clauses)
    if len(clauses) < 2:
        raise ValidationError(
            "a disjunctive condition requires at least two clauses; one clause is "
            "a conjunction, so state it with conditions or classical_bit instead"
        )
    if (
        tuple(sorted(clauses, key=lambda item: (len(item), item))) != clauses
        or len(set(clauses)) != len(clauses)
        or any(
            set(left).issubset(right)
            for clause_index, left in enumerate(clauses)
            for right in clauses[clause_index + 1 :]
        )
    ):
        raise ValidationError(
            "a disjunctive condition requires clauses that are canonical, "
            "distinct, and minimal"
        )
    return clauses


def instruction_is_dynamic(instruction: Instruction) -> bool:
    """Whether one instruction needs a classical register to be interpreted.

    This is a predicate over the keys a program carries, not a validation pass: a
    caller that asks whether an instruction is dynamic must get an answer even for
    a malformed condition, which the parser refuses a layer later.
    """

    return is_classical_control_instruction(instruction)


def classical_width(circuit: DynamicCircuit) -> int:
    return validate_classical_control(
        circuit._instructions,
        n_qubits=circuit.n_qubits,
    )


__all__ = (
    "classical_width",
    "instruction_condition_clauses",
    "instruction_conditions",
    "instruction_is_dynamic",
    "validate_condition_clauses",
    "validate_disjunctive_clauses",
)
