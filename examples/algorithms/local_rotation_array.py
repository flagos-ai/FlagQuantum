"""Append a two-target array using phase-exact local synthesis without an SDK."""

from collections import Counter

import numpy as np
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.optimization.local_unitary_synthesis import local_unitary_ir
from flagquantum.compiler.registers.lookup.source_qrom import source_qrom_workspace_size
from flagquantum.compiler.registers.phase.programmable_rotation_array import (
    _rotation_batches,
    programmable_rotation_array_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def make_example() -> FieldOperation:
    """Bind two angle tables to a phased SWAP base and local H interleaver."""
    angles = ((1, 2, 3), (3, 0, 1))
    selection, load, target = (0, 1), (2, 3), (4, 5)
    identity = np.eye(4, dtype=np.complex128)
    swap = identity[[0, 2, 1, 3]]
    plus, minus = (identity + swap) / 2, (identity - swap) / 2
    h = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    interleaver = np.exp(-0.17j) * np.kron(h, h)

    def rotate(
        control: int, targets: tuple[int, ...], power: int
    ) -> tuple[Instruction, ...]:
        exponent = 2.0 ** (-1 - power)
        matrix = np.exp(0.23j * exponent) * (
            plus + np.exp(1j * np.pi * exponent) * minus
        )
        return local_unitary_ir(matrix, targets, control=control)

    work_size = max(
        source_qrom_workspace_size(0, (2,), (delta,))
        for delta, _ in _rotation_batches(angles, len(load))
    )
    gates = programmable_rotation_array_ir(
        selection,
        load,
        target,
        angles,
        tuple(range(6, 6 + work_size)),
        tuple(range(work_size)),
        rotation=rotate,
        interleaved=(local_unitary_ir(interleaver, target),),
    )
    return FieldOperation(6 + work_size, gates)


def main() -> None:
    op = make_example()
    circuit = DynamicCircuit(op.n_qubits, dtype=torch.complex128)
    circuit.h(0).h(4).h(5)
    circuit.append(op)
    result = run_dynamic(circuit, shots=16, seed=42, strategy="trajectory")
    # The initialized selector visits only valid addresses 0 and 2.
    indices = torch.arange(1 << op.n_qubits)
    dirty = ((indices >> (op.n_qubits - 4)) & 3) != 0
    dirty |= (indices & ((1 << (op.n_qubits - 6)) - 1)) != 0
    assert torch.all(result.final_states.reshape(16, -1)[:, dirty].abs() < 1e-10)
    print("Potential named primitives", dict(Counter(g.name for g in op.instructions)))
    print("Local matrix leaves", sum(g.matrix is not None for g in op.instructions))
    print("Load and work restored for all trajectories")


if __name__ == "__main__":
    main()
