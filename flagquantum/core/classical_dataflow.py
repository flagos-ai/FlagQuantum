"""Core-owned internal classical XOR record and dynamic instruction spelling."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from operator import xor

from ..errors import ValidationError
from .ir import Instruction


@dataclass(frozen=True)
class ClassicalXor:
    """Assign XOR of old source bits to one explicitly owned classical slot."""

    sources: tuple[int, ...]
    target: int

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple) or any(
            type(bit) is not int or bit < 0 for bit in (*self.sources, self.target)
        ):
            raise ValidationError(
                "Classical XOR requires nonnegative integer bit indices"
            )

    def instruction(self) -> Instruction:
        """Encode through the existing dynamic IR fields without quantum operands."""
        return Instruction(
            "classical_xor",
            (),
            params={"sources": self.sources},
            metadata={"is_dynamic": True, "classical_bit": self.target},
        )

    def evaluate(self, classical: Sequence[int] | Mapping[int, int]) -> int:
        """Evaluate old values, rejecting undefined bits before any assignment."""
        values = tuple(
            (
                classical.get(bit, -1)
                if isinstance(classical, Mapping)
                else classical[bit] if bit < len(classical) else -1
            )
            for bit in self.sources
        )
        if any(value not in (0, 1) for value in values):
            raise ValidationError("Classical XOR reads an undefined classical bit")
        return reduce(xor, values, 0)


def parse_classical_xor(instruction: Instruction) -> ClassicalXor:
    """Validate the sole authoritative spelling of this internal operation."""
    if (
        instruction.name != "classical_xor"
        or instruction.wires
        or instruction.matrix is not None
        or set(instruction.params) != {"sources"}
        or set(instruction.metadata) != {"is_dynamic", "classical_bit"}
        or instruction.metadata["is_dynamic"] is not True
    ):
        raise ValidationError("Invalid classical XOR instruction")
    return ClassicalXor(
        instruction.params["sources"], instruction.metadata["classical_bit"]
    )
