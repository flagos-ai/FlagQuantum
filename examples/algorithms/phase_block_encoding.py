"""Append a phased block, preserving the full complex scalar and idle registers.

Run python -m examples.algorithms.phase_block_encoding.
"""

from cmath import exp
from collections import Counter
from math import pi

import torch

from examples.algorithms.unitary_block_encoding import child
from flagquantum.algorithms.primitives.phase_block_encoding import PhaseBlockEncoding
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
    source_add_cost,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    block = UnitaryBlockEncoding(
        (("a", 2), ("b", 2)), child, ancilla_bitsize=1, resource_bitsize=1
    )
    phase = PhaseBlockEncoding(block, 0.25, 1e-9)
    amplitudes = torch.complex(torch.arange(1, 65).double(), torch.arange(64).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::2] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(64):
        left, right, idle = word >> 4, (word >> 2) & 3, word & 3
        output = (left << 4) | (((left + right) % 4) << 2) | idle
        expected[output << 1] = amplitudes[word] * exp(0.25j * pi)
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    forward = block.make_on(
        system=(0, 1, 2, 3),
        ancilla=(4,),
        resource=(5,),
        workspace=(6,),
        classical_bits=(0,),
    )
    circuit.append(phase.make_on(forward, anchor=5))
    forward_ir = circuit.to_ir()
    result = run_dynamic(circuit, shots=16, seed=53, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=3e-13,
        rtol=3e-13,
    )
    inverse = block.make_on(
        system=(0, 1, 2, 3),
        ancilla=(4,),
        resource=(5,),
        workspace=(6,),
        classical_bits=(0,),
        adjoint=True,
    )
    circuit.append(phase.make_on(inverse, anchor=5, adjoint=True))
    restored = run_dynamic(circuit, shots=16, seed=59, strategy="batched")
    torch.testing.assert_close(
        restored.final_states,
        initial.expand_as(restored.final_states),
        atol=3e-13,
        rtol=3e-13,
    )
    report = estimate_ftqc_resources(
        phase.resource_model(block.resource_model(source_add_cost(2, 2))),
        emitted_ir=forward_ir,
    )
    assert report.emitted is not None
    counts = Counter(gate.name for gate in forward_ir.instructions)
    print("Source-model T equivalents", report.logical.gates.total_t_count())
    print("Emitted T/T-dagger instructions", report.emitted.t_count)
    print("Emitted unsynthesized phase instructions", counts["phase"])
    print("Scalar representation overhead: 2 X and 2 phase instructions")
    print("Model logical peak / reserved wires", report.logical.peak_logical_qubits, 7)
    print("Full complex phase, coherent auxiliary ports and clean work verified")


if __name__ == "__main__":
    main()
