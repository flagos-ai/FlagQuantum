"""Run coherent controlled X/Z lookup through circuit.append(op).

Run python -m examples.algorithms.recursive_lookup [--pauli x|z].
"""

import argparse
from collections import Counter
from typing import Literal

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.lookup.recursive_lookup import (
    recursive_lookup_operations,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def verify_lookup(pauli: Literal["x", "z"]) -> None:
    """Check a full two-bit identity table with entangled data and control."""
    table = ((False, False), (False, True), (True, False), (True, True))
    operations = recursive_lookup_operations(
        table, (0, 1), (3, 4), (6, 7), pauli=pauli, control=2
    )
    program = ReversiblePhaseProgram(8, 1, operations)
    generator = torch.Generator().manual_seed(248)
    psi = torch.complex(
        torch.randn(64, generator=generator, dtype=torch.float64),
        torch.randn(64, generator=generator, dtype=torch.float64),
    )
    psi /= psi.norm()
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::4] = psi
    expected = torch.zeros_like(initial)
    for index, amplitude in enumerate(psi):
        address = index >> 4
        controlled = bool(index & 8)
        target = index ^ ((address << 1) if controlled and pauli == "x" else 0)
        phase = (
            -1
            if controlled
            and pauli == "z"
            and ((address & ((index >> 1) & 3)).bit_count() % 2)
            else 1
        )
        expected[target << 2] = phase * amplitude
    op = FieldOperation(8, bind_reversible_phase_execution(program))
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(op)
    result = run_dynamic(circuit, shots=32, seed=74, strategy="batched")
    overlaps = result.final_states @ expected.conj()
    assert bool(torch.allclose(overlaps.abs(), torch.ones_like(overlaps.real)))
    assert bool(
        torch.all(result.final_states.reshape(32, -1, 4)[:, :, 1:].abs() < 1e-12)
    )
    print(
        "Source primitive counts",
        dict(Counter(operation.opcode for operation in operations)),
    )
    counts = count_emitted_instructions(circuit.to_ir())
    print("Expanded IR unconditional", counts.unconditional)
    print("Expanded IR conditional", counts.conditional)
    print("Full-table coherent lookup verified; work cleared")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pauli", choices=("x", "z"), default="x")
    args = parser.parse_args()
    verify_lookup("x" if args.pauli == "x" else "z")


if __name__ == "__main__":
    main()
