#!/usr/bin/env python3
"""Build the frozen 36q high-width amplitude TN planning workload."""

from __future__ import annotations

from pathlib import Path

import torch

import flagquantum as fq
from benchmarks.development.tn_gap_common import make_workload, write_workload
from flagquantum.simulation.tensor_execution import _amplitude_projection


def main() -> None:
    rows, columns, cycles = 4, 9, 4
    circuit = fq.Circuit(rows * columns)
    for cycle in range(cycles):
        for wire in range(rows * columns):
            circuit.ry(wire, theta=0.01 * (cycle + 1))
        for row in range(rows):
            for column in range(columns - 1):
                left = row * columns + column
                circuit.cx(left, left + 1)
        for row in range(rows - 1):
            for column in range(columns):
                top = row * columns + column
                circuit.cx(top, top + columns)
    plan = fq.build_tensor_network(circuit, dtype=torch.complex64)
    nodes, output = _amplitude_projection(plan, "0" * (rows * columns))
    workload = make_workload(
        name="flagquantum_36q_4x9_c4_amplitude",
        inputs=tuple(node.labels for node in nodes),
        shapes=tuple(tuple(node.tensor.shape) for node in nodes),
        output=output,
        dtype="complex64",
        category="high_width_sparse_output",
    )
    write_workload(
        workload,
        Path(
            "benchmarks/workloads/tn_cotengra_gap/"
            "flagquantum_36q_4x9_c4_amplitude.json"
        ),
    )


if __name__ == "__main__":
    main()
