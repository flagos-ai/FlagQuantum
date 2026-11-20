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
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

SIGNED_WORD = classical_signed_register_type(2)


@dataclass(frozen=True)
class ClassicalWordFeedback:
    """Apply X when a stored classical word matches its activation."""

    value: int
    dtype: RegisterType = SIGNED_WORD
    activation: int = -1

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        child = bit_gate_function(
            "main",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("x", (0,)),)),
        )
        program = classical_control_component_program(
            ProgramRecord((child,)),
            (((), (self.activation,)),),
            control_types=(self.dtype,),
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
    field = RegisterType(
        "finite_field",
        0,
        2,
        (("storage", "classical"), ("characteristic", 2), ("degree", 2), ("order", 4)),
    )
    polynomial = RegisterType(
        "finite_field_polynomial",
        0,
        2,
        (
            ("storage", "classical"),
            ("polynomial_degree", 1),
            ("characteristic", 2),
            ("degree", 1),
            ("order", 2),
        ),
    )
    for dtype in (field, polynomial):
        for value in range(4):
            circuit = DynamicCircuit(2, inputs=initial, dtype=torch.complex128)
            circuit.append(ClassicalWordFeedback(value, dtype, 3))
            result = run_dynamic(circuit, shots=1, strategy="trajectory")
            expected = (
                initial.reshape(2, 2).flip(0).flatten() if value == 3 else initial
            )
            torch.testing.assert_close(
                result.final_states[0].flatten(), expected, rtol=0, atol=3e-15
            )
    ternary = RegisterType(
        "finite_field",
        0,
        2,
        (("storage", "classical"), ("characteristic", 3), ("degree", 1), ("order", 3)),
    )
    ternary_polynomial = RegisterType(
        "finite_field_polynomial",
        0,
        4,
        (
            ("storage", "classical"),
            ("polynomial_degree", 1),
            ("characteristic", 3),
            ("degree", 1),
            ("order", 3),
        ),
    )
    for dtype, words, activation in (
        (ternary, (0, 1, 2), 2),
        (ternary_polynomial, (0, 1, 2, 4, 5, 6, 8, 9, 10), 10),
    ):
        for value in words:
            circuit = DynamicCircuit(2, inputs=initial, dtype=torch.complex128)
            circuit.append(ClassicalWordFeedback(value, dtype, activation))
            result = run_dynamic(circuit, shots=1, strategy="trajectory")
            expected = (
                initial.reshape(2, 2).flip(0).flatten()
                if value == activation
                else initial
            )
            torch.testing.assert_close(
                result.final_states[0].flatten(), expected, rtol=0, atol=3e-15
            )
            assert Counter(g.name for g in circuit.to_ir().instructions) == {
                "classical_xor": dtype.classical_bits,
                "x": 1,
            }
    print("Signed, binary and nonbinary field words preserve complex spectator states")
    print("Emitted per binding uses word-width classical initializers and 1 guarded X")
    print("Reads add 0 gates and 0 slots; quantum workspace and measurements are 0")
    print("Frozen source wrapper has no physical decomposition or QEC estimate")


if __name__ == "__main__":
    main()
