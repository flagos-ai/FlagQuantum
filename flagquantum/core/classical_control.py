"""Internal validation contract for classical-control instruction metadata."""

from __future__ import annotations

from collections.abc import Iterable

from ..errors import ValidationError
from .ir import Instruction

CLASSICAL_CONTROL_KEYS: tuple[str, ...] = (
    "is_dynamic",
    "classical_bit",
    "conditions",
    "condition_clauses",
)


def is_classical_control_instruction(instruction: Instruction) -> bool:
    """Return whether an instruction carries classical-control metadata."""

    return any(key in instruction.metadata for key in CLASSICAL_CONTROL_KEYS)


def _integer(value: object, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} must be an integer")
    return value


def _condition_pair(value: object, *, owner: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{owner} must contain (classical_bit, value) tuples")
    bit = _integer(value[0], owner=f"{owner} classical bit")
    expected = _integer(value[1], owner=f"{owner} expected value")
    if bit < 0:
        raise ValidationError("classical condition bits must be non-negative")
    if expected not in {0, 1}:
        raise ValidationError("classical condition values must be zero or one")
    return bit, expected


def _condition_clause(value: object, *, owner: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{owner} must be a tuple of classical bit/value pairs")
    if not value:
        raise ValidationError("a classical condition clause must test at least one bit")
    clause = tuple(_condition_pair(pair, owner=owner) for pair in value)
    if tuple(sorted(clause)) != clause:
        raise ValidationError("classical condition clauses must be canonically ordered")
    if len({bit for bit, _ in clause}) != len(clause):
        raise ValidationError("classical condition clauses require unique bits")
    return clause


def _instruction_clauses(
    instruction: Instruction,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    metadata = instruction.metadata
    if "conditions" in metadata and "condition_clauses" in metadata:
        raise ValidationError(
            "instruction cannot define both conditions and condition_clauses"
        )
    if "conditions" in metadata:
        return (_condition_clause(metadata["conditions"], owner="conditions"),)
    if "condition_clauses" not in metadata:
        return ()
    raw_clauses = metadata["condition_clauses"]
    if not isinstance(raw_clauses, tuple):
        raise TypeError("condition_clauses must be a tuple of condition clauses")
    clauses = tuple(
        _condition_clause(clause, owner="condition_clauses") for clause in raw_clauses
    )
    if len(clauses) < 2:
        raise ValidationError(
            "condition_clauses requires at least two clauses; use conditions for "
            "one conjunction"
        )
    if (
        tuple(sorted(clauses, key=lambda item: (len(item), item))) != clauses
        or len(set(clauses)) != len(clauses)
        or any(
            set(left).issubset(right)
            for index, left in enumerate(clauses)
            for right in clauses[index + 1 :]
        )
    ):
        raise ValidationError(
            "condition_clauses must be canonical, distinct, and minimal"
        )
    return clauses


def validate_classical_control(
    instructions: Iterable[Instruction],
    *,
    n_qubits: int,
) -> int:
    """Validate classical-control metadata and return its register width.

    Conditions may read only classical bits produced by an earlier measurement.
    The validator intentionally leaves the public ``CircuitIR`` schema unchanged:
    it validates the four metadata keys already used by dynamic execution.
    """

    wire_count = _integer(n_qubits, owner="n_qubits")
    if wire_count < 0:
        raise ValidationError("n_qubits must be non-negative")

    measured_bits: set[int] = set()
    width = 0
    for instruction in instructions:
        if not isinstance(instruction, Instruction):
            raise TypeError("classical-control validation requires Instruction values")
        if any(wire < 0 or wire >= wire_count for wire in instruction.wires):
            raise ValidationError("dynamic instruction qubit is outside the circuit")

        metadata = instruction.metadata
        if "is_dynamic" in metadata and not isinstance(metadata["is_dynamic"], bool):
            raise TypeError("is_dynamic must be a bool")
        classical_bit: int | None = None
        if "classical_bit" in metadata:
            classical_bit = _integer(metadata["classical_bit"], owner="classical_bit")
            if classical_bit < 0:
                raise ValidationError("classical_bit must be non-negative")
            if instruction.name != "measure":
                raise ValidationError(
                    "classical_bit is valid only on a measure instruction"
                )
            width = max(width, classical_bit + 1)
        if instruction.name == "measure" and classical_bit is None:
            raise ValidationError("a measure instruction requires classical_bit")

        clauses = _instruction_clauses(instruction)
        if clauses and instruction.name in {"measure", "reset"}:
            raise ValidationError(
                f"a conditional {instruction.name} is not constructible"
            )
        read_bits = {bit for clause in clauses for bit, _ in clause}
        unread = sorted(read_bits - measured_bits)
        if unread:
            raise ValidationError(
                f"classical bit {unread[0]} was read before measurement"
            )
        if read_bits:
            width = max(width, max(read_bits) + 1)

        if instruction.name == "measure" and classical_bit is not None:
            measured_bits.add(classical_bit)

    return width


__all__ = (
    "CLASSICAL_CONTROL_KEYS",
    "is_classical_control_instruction",
    "validate_classical_control",
)
