"""Bind and append a register rotation through the existing circuit protocol.

Run: python -m examples.algorithms.register_bit_rotation
"""

from dataclasses import dataclass

import flagquantum as fq
from flagquantum.compiler.artifacts.operand_array import OperandArray
from flagquantum.compiler.registers.arithmetic.carry_parity import _Coefficient
from flagquantum.compiler.registers.bits.bit_rotation_binding import (
    bound_bit_rotation_ir,
)
from flagquantum.core.ir import CircuitIR


@dataclass(frozen=True)
class RegisterBitRotation:
    """Append a rotation of big-endian ports with caller-owned work."""

    target: tuple[int, ...] | OperandArray
    shift: int | tuple[_Coefficient, ...] | OperandArray = 1
    control: _Coefficient = True
    right: bool = False
    clean: tuple[int, ...] = ()
    measurement_bit: int | None = None
    predicate_bit: int | None = None

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        instructions = bound_bit_rotation_ir(
            self.target,
            self.shift,
            control=self.control,
            right=self.right,
            clean=self.clean,
            measurement_bit=self.measurement_bit,
            predicate_bit=self.predicate_bit,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, instructions))


def main() -> None:
    circuit = fq.Circuit(4)
    circuit.x(0)
    circuit.append(RegisterBitRotation((0, 1, 2, 3)))
    assert int(circuit.state().abs().argmax()) == 1


if __name__ == "__main__":
    main()
