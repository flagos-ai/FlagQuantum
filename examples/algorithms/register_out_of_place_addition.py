"""Create and consume a sum with circuit.append(op) and explicit output ownership.

Run: python -m examples.algorithms.register_out_of_place_addition
Expected: every sampled erasure branch restores both words and clears the sum.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.arithmetic.out_of_place_add_resource_model import (
    source_out_of_place_add_resource_formula,
)
from flagquantum.compiler.registers.arithmetic.out_of_place_addition_component import (
    out_of_place_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class SumOutput:
    """Own a three-bit sum, or consume that correctly computed sum."""

    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        program = out_of_place_addition_program(2, adjoint=self.adjoint)
        inputs = ((0, 1), (2, 3), (4, 5, 6)) if self.adjoint else ((0, 1), (2, 3))
        bound = bind_register_program(
            program,
            inputs,
            clean=() if self.adjoint else (6, 5, 4),
            allow_workspace_outputs=not self.adjoint,
            input_ownership=(
                ((4, False), (5, False), (6, False)) if self.adjoint else ()
            ),
        )
        return circuit.compose(bound.circuit)


def main() -> None:
    """Check coherent sum amplitudes and sampled measured cleanup."""
    raw = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::8] = amplitudes
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    circuit.append(SumOutput())
    indexes = torch.arange(16)
    sums = (indexes >> 2) + (indexes & 3)
    expected = torch.zeros_like(initial)
    expected[(indexes << 3) | sums] = amplitudes
    result = run_dynamic(circuit, shots=8, seed=23, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), rtol=0, atol=2e-12
    )
    print(
        "Measured forward primitives",
        dict(Counter(g.name for g in circuit.to_ir().instructions)),
    )
    print("Independent forward model", source_out_of_place_add_resource_formula(2))
    circuit.append(SumOutput(adjoint=True))
    restored = run_dynamic(circuit, shots=16, seed=31, strategy="trajectory")
    torch.testing.assert_close(
        restored.final_states.reshape(16, -1),
        initial.expand(16, -1),
        rtol=0,
        atol=2e-12,
    )
    print("Coherent input words restored and all three sum wires zero")
    print(
        "Measured roundtrip primitives",
        dict(Counter(g.name for g in circuit.to_ir().instructions)),
    )
    print(
        "Independent erasure model",
        source_out_of_place_add_resource_formula(2, adjoint=True),
    )


if __name__ == "__main__":
    main()
