"""Append stored polynomial addition and restore the complete complex input.

Run: python -m examples.algorithms.register_field_polynomial_addition
"""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.finite_field.polynomial_addition_component import (
    field_polynomial_addition_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.registers import RegisterType


@dataclass(frozen=True)
class FieldPolynomialAddition:
    """Use packed physical views with explicitly typed field coefficients."""

    layout: FieldPolynomialLayout
    coefficient: RegisterType
    inputs: tuple[tuple[int, ...], ...]
    constants: tuple[int, ...] | None = None
    adjoint: bool = False

    def bind(self) -> BoundRegisterProgram:
        program = field_polynomial_addition_program(
            self.layout,
            self.coefficient,
            constants=self.constants,
            adjoint=self.adjoint,
        )
        return bind_register_program(program, self.inputs)

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        bound = self.bind()
        return circuit.compose(CircuitIR(circuit.n_qubits, bound.circuit.instructions))


def main() -> None:
    coefficient = RegisterType(
        "finite_field",
        2,
        parameters=(("characteristic", 2), ("degree", 2), ("order", 4)),
    )
    layout = FieldPolynomialLayout(1, 2)
    source, target = tuple(range(4)), tuple(range(4, 8))
    state = torch.arange(1, 257, dtype=torch.float64).to(torch.complex128)
    state += 1j * torch.flip(state.real, (0,))
    state /= torch.linalg.vector_norm(state)
    operation = FieldPolynomialAddition(layout, coefficient, (source, target))
    circuit = fq.Circuit(8, inputs=state, dtype=torch.complex128).append(operation)
    restored = circuit.append(operation).state().flatten()
    torch.testing.assert_close(restored, state, atol=1e-12, rtol=0)
    print(
        "Polynomial XOR restored every complex amplitude through stored coefficient calls."
    )
    print("Forward CX:", len(operation.bind().circuit.instructions))
    print("Extra work, measurements and classical slots: 0")


if __name__ == "__main__":
    main()
