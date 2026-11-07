"""Run python -m examples.algorithms.classical_partition_values."""

from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import product

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.bit_encoding import (
    bit_array_from_bits,
    bit_array_to_bits,
    bit_from_bits,
)
from flagquantum.simulation.classical_partition import (
    PartitionValuePort,
    partition_classical_value,
    unpartition_classical_values,
)
from flagquantum.simulation.classical_pauli_values import x_classical_value
from flagquantum.simulation.integer_encoding import integer_from_bits, integer_to_bits


@dataclass(frozen=True)
class CellFlip:
    """Bind a shaped numerical port to the existing flat wire partition."""

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        ports = partition_register((0, 1, 2, 3, 4), (("cells", 4), ("flag", 1)))
        return circuit.compose(
            CircuitIR(circuit.n_qubits, (Instruction("x", (ports["cells"][2],)),))
        )


def main() -> None:
    ports = (
        PartitionValuePort(
            "cells", 1, (2, 2), bit_from_bits, bit_array_from_bits, bit_array_to_bits
        ),
        PartitionValuePort(
            "flag", 1, (), bit_from_bits, bit_array_from_bits, bit_array_to_bits
        ),
    )
    encode = partial(integer_to_bits, width=5)
    decode = partial(integer_from_bits, width=5)
    grid = torch.arange(1, 65, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for value, spectator in product(range(32), (0, 1)):
        parts = partition_classical_value(value, encode, ports)
        cells = parts["cells"]
        assert isinstance(cells, np.ndarray) and cells.shape == (2, 2)
        cells[1, 0] = x_classical_value(cells[1, 0])
        updated = unpartition_classical_values(parts, ports, decode)
        expected[(updated << 1) | spectator] = initial[(value << 1) | spectator]
    circuit = fq.Circuit(6, inputs=initial, dtype=torch.complex128)
    op = CellFlip()
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert Counter(g.name for g in circuit.to_ir().instructions) == {"x": 1}
    circuit.append(op)
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    print(
        "2x2 cell port and scalar flag preserve full complex amplitudes and spectator"
    )
    print("Partition/unpartition: 0 emitted gates, 0 T, 0 measurements, 0 work")
    print("Explicit cell action: 1 X; inverse restores input")
    print("Source bookkeeping model: 0 gate cost, 5 logical wires; spectator adds one")


if __name__ == "__main__":
    main()
