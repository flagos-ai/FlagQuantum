"""Execute a stored field product through circuit.append(op).

Run: python -m examples.algorithms.register_field_multiplication
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.multiplication_component import (
    field_multiplication_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.registers import RegisterType


@dataclass(frozen=True)
class FieldProduct:
    """Bind MSB field words and retain the returned physical product placement."""

    dtype: RegisterType
    x: tuple[int, ...]
    y: tuple[int, ...]
    target: tuple[int, ...]
    workspace: tuple[int, ...]
    modulus: int
    algorithm: Literal["quadratic", "karatsuba"] = "karatsuba"
    plus_equal_prod: bool = False
    adjoint: bool = False

    def bind(self) -> BoundRegisterProgram:
        if self.algorithm == "karatsuba":
            if self.adjoint or self.plus_equal_prod:
                raise ValueError("Karatsuba example uses the public forward product")
            program = karatsuba_multiplication_program(self.dtype, modulus=self.modulus)
        elif self.algorithm == "quadratic":
            program = field_multiplication_program(
                self.dtype,
                modulus=self.modulus,
                plus_equal_prod=self.plus_equal_prod,
                adjoint=self.adjoint,
            )
        else:
            raise ValueError("Unknown field multiplication algorithm")
        consume = self.plus_equal_prod or self.adjoint
        inputs = (self.x, self.y, self.target) if consume else (self.x, self.y)
        return bind_register_program(
            program,
            inputs,
            clean=self.workspace if consume else self.target + self.workspace,
            input_ownership=(
                tuple((w, False) for w in self.target) if self.adjoint else ()
            ),
            allow_workspace_outputs=True,
        )

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        bound = self.bind()
        return circuit.compose(CircuitIR(circuit.n_qubits, bound.circuit.instructions))


def main() -> None:
    dtype = RegisterType(
        "finite_field",
        2,
        parameters=(
            ("characteristic", 2),
            ("degree", 2),
            ("order", 4),
            ("polynomial", 7),
            ("polynomial_field_characteristic", 2),
            ("polynomial_field_degree", 1),
            ("polynomial_field_primitive_element", 1),
            ("polynomial_field_modulus", 3),
        ),
    )
    operation = FieldProduct(dtype, (0, 1), (2, 3), (4, 5), (), 7)
    bound = operation.bind()
    state = torch.zeros(64, dtype=torch.complex128)
    state[20] = 1  # x=y=1; field product is one.
    result = (
        fq.Circuit(6, inputs=state, dtype=torch.complex128)
        .append(operation)
        .state()
        .flatten()
    )
    index = int(torch.argmax(result.abs()))
    output = sum(
        ((index >> (5 - w)) & 1) << (1 - i) for i, w in enumerate(bound.outputs[2])
    )
    assert output == 1
    print("Field product:", output)
    print("Physical result aliases:", bound.outputs[2])
    print(
        "Emitted primitive counts:",
        dict(Counter(g.name for g in bound.circuit.instructions)),
    )
    print(
        "Frozen analytical model is recorded separately in FIELD_MULTIPLICATION_FILE_AUDIT.md."
    )


if __name__ == "__main__":
    main()
