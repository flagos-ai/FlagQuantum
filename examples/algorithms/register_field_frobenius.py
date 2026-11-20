"""Execute a stored binary-field Frobenius graph through circuit.append(op).

Run: python -m examples.algorithms.register_field_frobenius
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.finite_field.frobenius_component import (
    field_frobenius_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.registers import RegisterType


@dataclass(frozen=True)
class FieldFrobenius:
    """Bind one typed word, preserving its returned physical output aliases."""

    dtype: RegisterType
    target: tuple[int, ...]
    modulus: int
    count: int = 1
    adjoint: bool = False

    def bind(self) -> BoundRegisterProgram:
        program = field_frobenius_program(
            self.dtype, self.count, modulus=self.modulus, adjoint=self.adjoint
        )
        return bind_register_program(program, (self.target,))

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        bound = self.bind()
        return circuit.compose(CircuitIR(circuit.n_qubits, bound.circuit.instructions))


def main() -> None:
    dtype = RegisterType(
        "finite_field",
        3,
        parameters=(
            ("characteristic", 2),
            ("degree", 3),
            ("order", 8),
            ("polynomial", 11),
            ("polynomial_field_characteristic", 2),
            ("polynomial_field_degree", 1),
            ("polynomial_field_primitive_element", 1),
            ("polynomial_field_modulus", 3),
        ),
    )
    operation = FieldFrobenius(dtype, (0, 1, 2), 11)
    bound = operation.bind()
    state = torch.tensor([1, 2j, -3, 4j, 5, -6j, 7, 8j], dtype=torch.complex128)
    state /= torch.linalg.vector_norm(state)
    circuit = fq.Circuit(3, inputs=state, dtype=torch.complex128).append(operation)
    inverse = FieldFrobenius(dtype, bound.outputs[0], 11, adjoint=True)
    restored = circuit.append(inverse).state().flatten()
    torch.testing.assert_close(restored, state, atol=1e-12, rtol=0)
    assert inverse.bind().outputs == ((0, 1, 2),)
    print("Full complex amplitudes restored, zero work and measurement slots.")
    print("Forward physical outputs:", bound.outputs[0])
    print(
        "Emitted primitive counts:",
        dict(Counter(g.name for g in bound.circuit.instructions)),
    )


if __name__ == "__main__":
    main()
