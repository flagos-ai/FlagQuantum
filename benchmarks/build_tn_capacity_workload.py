#!/usr/bin/env python3
"""Build the frozen tensor-network capacity workload.

The release contract needs a workload that a single device cannot train and that
a sliced distributed run can, so this freezes the *circuit* and the measurement
protocol rather than any execution strategy. Slicing is how a run executes the
workload, not part of what the workload is, which is why the frozen file is the
expectation's tensor list and its digest.

The earlier candidate for this premise was ``flagquantum_18q_13l_expectation``.
It was falsified by measurement: the whole training step completes on one A800 in
14.3 s at a 0.70 GiB peak, so it cannot be the workload a single device fails to
train. The premise recorded here was chosen from a measured capacity ladder on
one device instead, and it stays honest only as long as re-running the ladder
puts the frozen shape above the device ceiling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import flagquantum as fq
from benchmarks.runners.tn.tn_gap_common import make_workload, write_workload
from flagquantum.simulation.tensor_network.observables import (
    build_tensor_network_expectation,
)

ROWS = 4
COLUMNS = 7
CYCLES = 2
DEFAULT_OUTPUT = Path("benchmarks/workloads/tn_cotengra_gap")


def grid_circuit(rows: int, columns: int, cycles: int) -> fq.Circuit:
    """Build the alternating-grid circuit the capacity ladder measured.

    Every cycle rotates every wire and then entangles along the rows and along
    the columns, so the circuit has no slack wire and the expectation keeps every
    label reachable. The ladder used this construction on one device, so a frozen
    shape can be compared against that measurement directly.
    """

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
    return circuit


def build(rows: int = ROWS, columns: int = COLUMNS, cycles: int = CYCLES):
    """Return the frozen workload for one grid shape."""

    circuit = grid_circuit(rows, columns, cycles)
    expectation = build_tensor_network_expectation(
        circuit, z=list(range(rows * columns))
    )
    return make_workload(
        name=f"flagquantum_{rows}x{columns}x{cycles}_expectation",
        inputs=tuple(node.labels for node in expectation.nodes),
        shapes=tuple(node.tensor.shape for node in expectation.nodes),
        output=expectation.output_labels,
        dtype=str(expectation.nodes[0].tensor.dtype).removeprefix("torch."),
        category="flagquantum_memory_intensive",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--columns", type=int, default=COLUMNS)
    parser.add_argument("--cycles", type=int, default=CYCLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    torch.manual_seed(0)
    workload = build(arguments.rows, arguments.columns, arguments.cycles)
    path = arguments.output / f"{workload.name}.json"
    write_workload(workload, path)
    summary = workload.summary()
    print(f"workload {path}")
    print(f"  tensors={summary['tensor_count']} labels={summary['label_count']}")
    print(f"  dtype={summary['dtype']} identity={summary['identity']}")


if __name__ == "__main__":
    main()
