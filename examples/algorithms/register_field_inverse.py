"""Execute the stored field inverse and retain the source's logical junk.

Run: python -m examples.algorithms.register_field_inverse
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.finite_field.inverse_component import (
    field_inverse_program,
)
from flagquantum.compiler.registers.finite_field.inverse_decomposition import (
    field_inverse_decomposition_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.registers import RegisterType


@dataclass(frozen=True)
class FieldInverse:
    """Bind a mathematical or literal recipe, retaining result/junk aliases.

    The explicit literal profile preserves original defects and rejects unproved
    clean releases; the default computes the correct mathematical inverse.
    """

    dtype: RegisterType
    source: tuple[int, ...]
    workspace: tuple[int, ...]
    modulus: int
    literal: bool = False

    def bind(self) -> BoundRegisterProgram:
        if type(self.literal) is not bool:
            raise ValueError("Inverse profile requires a boolean literal flag")
        program = (
            field_inverse_decomposition_program
            if self.literal
            else field_inverse_program
        )(self.dtype, modulus=self.modulus)
        return bind_register_program(
            program,
            (self.source,),
            clean=self.workspace,
            allow_workspace_outputs=True,
        )

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
    operation = FieldInverse(dtype, (0, 1, 2), tuple(range(3, 9)), 11)
    bound = operation.bind()
    state = torch.zeros(512, dtype=torch.complex128)
    state[3 << 6] = 1
    result = (
        fq.Circuit(9, inputs=state, dtype=torch.complex128)
        .append(operation)
        .state()
        .flatten()
    )
    index = int(result.abs().argmax())
    inverse = sum(
        ((index >> (8 - w)) & 1) << (2 - i) for i, w in enumerate(bound.outputs[1])
    )
    assert inverse == 6
    print("GF(8) inverse(3):", inverse)
    print("Retained logical junk words:", bound.outputs[2:])
    print(
        "Emitted primitive counts:",
        dict(Counter(g.name for g in bound.circuit.instructions)),
    )
    print("Retained junk and clean scratch have distinct ownership.")


if __name__ == "__main__":
    main()
