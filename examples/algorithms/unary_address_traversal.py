"""Flip a target for selected addresses through circuit.append(operation).

Run python -m examples.algorithms.unary_address_traversal.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation,
    ReversiblePhaseProgram,
)
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.kickmix_unary_iteration import (
    unary_iteration_operations,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def make_circuit() -> DynamicCircuit:
    """Select addresses zero and three in a uniform two-bit word."""
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[::8] = 0.5
    operations = unary_iteration_operations(
        (0, 1),
        (3, 4),
        (0, 3),
        lambda value, match: (ReversiblePhaseOperation("cx", (match, 2)),),
    )
    program = ReversiblePhaseProgram(5, 1, operations)
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(5, bind_reversible_phase_execution(program)))
    print("Source primitive counts", dict(Counter(op.opcode for op in operations)))
    return circuit


def main() -> None:
    result = run_dynamic(make_circuit(), shots=32, seed=71, strategy="batched")
    expected = torch.zeros(32, dtype=torch.complex128)
    expected[torch.tensor([4, 8, 16, 28])] = 0.5
    # Each trajectory can carry an unobservable global phase from reset.
    overlap = result.final_states @ expected.conj()
    assert bool(torch.allclose(overlap.abs(), torch.ones_like(overlap.real)))
    assert bool(
        torch.all(result.final_states.reshape(32, 8, 4)[:, :, 1:].abs() < 1e-12)
    )
    print("Selected-address coherent map verified; work cleared")


if __name__ == "__main__":
    main()
