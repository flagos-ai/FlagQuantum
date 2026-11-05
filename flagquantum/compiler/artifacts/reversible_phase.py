"""Immutable operations for a restricted reversible-phase target."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReversiblePhaseOperation:
    """A lowered opcode with ordered wires and optional classical operands."""

    opcode: str
    wires: tuple[int, ...]
    result_bit: int | None = None
    condition_bit: int | None = None


@dataclass(frozen=True)
class ReversiblePhaseProgram:
    """A validated operation stream using logical wire order."""

    n_wires: int
    n_bits: int
    operations: tuple[ReversiblePhaseOperation, ...]
