"""Build the frozen stage-A tensor-network comparison workload set."""

from __future__ import annotations

import argparse
from pathlib import Path

import flagquantum as fq
from benchmarks.development.tn_gap_common import make_workload, write_workload


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/workloads/tn_cotengra_gap"),
    )
    return parser.parse_args()


def _workloads():
    yield make_workload(
        name="small_ring_6",
        inputs=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
        shapes=((2, 2),) * 6,
        category="small_exact",
    )
    yield make_workload(
        name="medium_grid_4x4",
        inputs=(
            (0, 12),
            (0, 1, 13),
            (1, 2, 14),
            (2, 15),
            (3, 12, 16),
            (3, 4, 13, 17),
            (4, 5, 14, 18),
            (5, 15, 19),
            (6, 16, 20),
            (6, 7, 17, 21),
            (7, 8, 18, 22),
            (8, 19, 23),
            (9, 20),
            (9, 10, 21),
            (10, 11, 22),
            (11, 23),
        ),
        shapes=(
            (2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2),
            (2, 2, 2),
            (2, 2, 2, 2),
            (2, 2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2, 2),
            (2, 2, 2, 2),
            (2, 2, 2),
            (2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2),
        ),
        category="public_general",
    )
    yield make_workload(
        name="gradient_ladder_12",
        inputs=tuple(
            [(3 * index, 3 * index + 1, 3 * index + 2) for index in range(12)]
            + [(3 * index + 1, 3 * (index + 1)) for index in range(11)]
            + [(3 * index + 2,) for index in range(12)]
        ),
        shapes=tuple([(2, 2, 2)] * 12 + [(2, 2)] * 11 + [(2,)] * 12),
        category="gradient_topology",
    )
    circuit = fq.Circuit(18)
    for qubit in range(18):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for layer in range(13):
        for qubit in range(layer % 2, 17, 2):
            circuit.cx(qubit, qubit + 1)
    expectation = fq.build_tensor_network_expectation(
        circuit,
        z=list(range(18)),
    )
    yield make_workload(
        name="flagquantum_18q_13l_expectation",
        inputs=tuple(node.labels for node in expectation.nodes),
        shapes=tuple(node.tensor.shape for node in expectation.nodes),
        output=expectation.output_labels,
        dtype=str(expectation.nodes[0].tensor.dtype).removeprefix("torch."),
        category="flagquantum_memory_intensive",
    )


def main() -> None:
    arguments = _arguments()
    for workload in _workloads():
        write_workload(workload, arguments.output / f"{workload.name}.json")


if __name__ == "__main__":
    main()
