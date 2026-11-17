"""Bind exact single-qubit synthesis to the native append protocol."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING

from ..compiler.registers.primitives.global_phase import scalar_phase_ir
from ..core.ir import Instruction
from ._clifford_t_repr import to_sequence

if TYPE_CHECKING:
    from ..circuit import Circuit
    from ._su2_ct import SU2CliffordT


# The second element is the SU(2) normalization in units of pi / 8.
_RECIPES: dict[str, tuple[tuple[str, ...], int]] = {
    "I": ((), 0),
    "H": (("h",), 4),
    "S": (("s",), -2),
    "X": (("x",), 4),
    "Y": (("y",), -4),
    "Z": (("z",), 4),
    "Tx": (("h", "t", "h"), -1),
    "Ty": (("sdg", "h", "t", "h", "s"), -1),
    "Tz": (("t",), -1),
    "Tx*": (("h", "tdg", "h"), 1),
    "Tz*": (("tdg",), 1),
}


@dataclass(frozen=True)
class CliffordTOperation:
    """A synthesized SU(2) matrix acting on one existing circuit wire."""

    matrix: SU2CliffordT
    wire: int
    fmt: str = "xyz"

    def instructions(self) -> tuple[Instruction, ...]:
        matrix = self.matrix.rescale() if self.fmt == "xz" else self.matrix
        word = to_sequence(matrix, self.fmt)
        if word is None:
            raise ValueError("No exact word found in the requested Clifford+T format")
        instructions: list[Instruction] = []
        phase = 0
        for label in word:
            gates, factor = _RECIPES[label]
            instructions.extend(Instruction(name, (self.wire,)) for name in gates)
            phase += factor
        angle = ((phase + 8) % 16 - 8) * pi / 8
        return (*instructions, *scalar_phase_ir(self.wire, angle))

    def append_to(self, circuit: Circuit) -> Circuit:
        if not 0 <= self.wire < circuit.n_qubits:
            raise ValueError(
                f"Synthesized rotation wire {self.wire} is outside the circuit"
            )
        for instruction in self.instructions():
            circuit.gate(instruction.name, instruction.wires, params=instruction.params)
        return circuit
