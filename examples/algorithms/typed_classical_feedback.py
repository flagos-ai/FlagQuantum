"""Run python -m examples.algorithms.typed_classical_feedback."""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.component_classical_control import (
    classical_control_component_program,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_signed_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ClassicalWordFeedback:
    """Apply X when the stored two-bit signed integer equals minus one."""

    value: int

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        child = bit_gate_function(
            "main",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("x", (0,)),)),
        )
        program = classical_control_component_program(
            ProgramRecord((child,)),
            (((), (-1,)),),
            control_types=(classical_signed_register_type(2),),
        )
        bound = bind_register_program(
            program, ((), (0,)), classical_inputs=(self.value, None)
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, bound.circuit.instructions))


def main() -> None:
    initial = torch.tensor([1 + 2j, 2 - 1j, -3j, 4 + 1j], dtype=torch.complex128)
    initial /= initial.norm()
    for value in range(-2, 2):
        circuit = DynamicCircuit(2, inputs=initial, dtype=torch.complex128)
        circuit.append(ClassicalWordFeedback(value))
        result = run_dynamic(circuit, shots=1, strategy="trajectory")
        expected = initial.reshape(2, 2).flip(0).flatten() if value == -1 else initial
        torch.testing.assert_close(
            result.final_states[0].flatten(), expected, rtol=0, atol=3e-15
        )
        assert Counter(g.name for g in circuit.to_ir().instructions) == {
            "classical_xor": 2,
            "x": 1,
        }
    print("All signed inputs preserve complete amplitudes and spectator entanglement")
    print("Emitted per binding: 2 classical initializers, 1 guarded X, 0 measurements")
    print("Reads add 0 gates and 0 slots; quantum workspace is 0")
    print("Frozen source wrapper has no physical decomposition or QEC estimate")


if __name__ == "__main__":
    main()
