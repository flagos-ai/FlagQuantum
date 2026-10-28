"""Clifford products for FlagQuantum opcodes with no single Stim spelling."""

from __future__ import annotations

from collections.abc import Mapping

GATE_EXPANSIONS: Mapping[str, tuple[tuple[str, tuple[int, ...]], ...]] = {
    "dcx": (("CX", (0, 1)), ("CX", (1, 0))),
    # The omitted ECR correction is an unconditional global phase, which a
    # stabilizer tableau neither stores nor observes.
    "ecr": (("S", (0,)), ("SQRT_X", (1,)), ("CX", (0, 1)), ("X", (0,))),
}


def expand_gate(
    opcode: str, qubits: tuple[int, ...]
) -> tuple[tuple[str, tuple[int, ...]], ...] | None:
    """Map a declared Clifford product onto concrete circuit qubits."""

    expansion = GATE_EXPANSIONS.get(opcode)
    if expansion is None:
        return None
    return tuple(
        (name, tuple(qubits[position] for position in positions))
        for name, positions in expansion
    )
