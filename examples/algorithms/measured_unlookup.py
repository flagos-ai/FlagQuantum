"""Compute table data and erase it coherently through circuit.append(op).

Run python -m examples.algorithms.measured_unlookup.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.measured_unlookup import (
    measured_unlookup_operations,
)
from flagquantum.compiler.registers.recursive_lookup import recursive_lookup_operations
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    table = ((False,), (True,), (True,), (False,))
    forward = recursive_lookup_operations(
        table, (0, 1), (2,), (4, 5), measurement_bit=0
    )
    inverse = measured_unlookup_operations(
        table,
        (0, 1),
        (2,),
        (4, 5),
        (0, 1, 2, 3),
        measurement_bit=4,
        reused_phase_bits=(0,),
    )
    program = ReversiblePhaseProgram(6, 5, (*forward, *inverse))
    generator = torch.Generator().manual_seed(852)
    psi = torch.complex(
        torch.randn(8, generator=generator, dtype=torch.float64),
        torch.randn(8, generator=generator, dtype=torch.float64),
    )
    psi /= psi.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    for address in range(4):
        for spectator in range(2):
            initial[(address << 4) | (spectator << 2)] = psi[2 * address + spectator]
    op = FieldOperation(6, bind_reversible_phase_execution(program))
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(op)
    result = run_dynamic(circuit, shots=64, seed=32, strategy="batched")
    overlaps = result.final_states @ initial.conj()
    assert bool(torch.allclose(overlaps.abs(), torch.ones_like(overlaps.real)))
    print(
        "Source erasure primitive counts",
        dict(Counter(operation.opcode for operation in inverse)),
    )
    print(
        "Source complete primitive counts",
        dict(Counter(operation.opcode for operation in program.operations)),
    )
    counts = count_emitted_instructions(circuit.to_ir())
    print("Expanded complete IR unconditional", counts.unconditional)
    print("Expanded complete IR conditional", counts.conditional)
    print("Entangled address and spectator restored; table output and work cleared")


if __name__ == "__main__":
    main()
