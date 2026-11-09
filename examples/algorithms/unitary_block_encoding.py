"""Append a source-style unitary block and compare logical and emitted costs.

Run python -m examples.algorithms.unitary_block_encoding.
"""

from collections.abc import Mapping

import torch

from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler.registers.arithmetic.addition import add_adjoint_ir, add_ir
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def child(
    ports: Mapping[str, tuple[int, ...]],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    return (add_adjoint_ir if adjoint else add_ir)(
        ports["a"], ports["b"], workspace, classical_bits
    )


def main() -> None:
    encoding = UnitaryBlockEncoding(
        (("a", 2), ("b", 2)), child, ancilla_bitsize=1, resource_bitsize=1
    )
    amplitudes = torch.complex(torch.arange(1, 65).double(), torch.arange(64).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::2] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(64):
        left, right, idle = word >> 4, (word >> 2) & 3, word & 3
        output = (left << 4) | (((left + right) % 4) << 2) | idle
        expected[output << 1] = amplitudes[word]
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    circuit.append(
        encoding.make_on(
            system=(0, 1, 2, 3),
            ancilla=(4,),
            resource=(5,),
            workspace=(6,),
            classical_bits=(0,),
        )
    )
    forward_ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=16, seed=43, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=3e-13,
        rtol=3e-13,
    )
    circuit.append(
        encoding.make_on(
            system=(0, 1, 2, 3),
            ancilla=(4,),
            resource=(5,),
            workspace=(6,),
            classical_bits=(0,),
            adjoint=True,
        )
    )
    restored = run_dynamic(circuit, shots=16, seed=47, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=3e-13,
        rtol=3e-13,
    )
    report = estimate_ftqc_resources(
        encoding.resource_model(source_add_cost(2, 2)), emitted_ir=forward_ir
    )
    assert report.emitted is not None
    print("Model T equivalents", report.logical.gates.total_t_count())
    print("Emitted T/T-dagger instructions", report.emitted.t_count)
    print("Model logical peak", report.logical.peak_logical_qubits)
    print("Reserved execution wires", 7)
    print("Auxiliary bystanders and child work verified through complex round trip")


if __name__ == "__main__":
    main()
