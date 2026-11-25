"""Run a matrix-defined weighted graph; requires the existing PyMatching extra."""

from __future__ import annotations

import torch

from flagquantum.qec.adapters.pymatching_graph import PyMatchingGraphDecoder
from flagquantum.qec.decoders.matching_graph import MatchingGraph


def main() -> None:
    checks = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.uint8)
    faults = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.uint8)
    graph, columns = MatchingGraph.from_check_matrix(
        checks,
        faults_matrix=faults,
        weights=torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64),
        error_probabilities=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        merge_strategy="disallow",
        use_virtual_boundary_node=True,
    )
    print(columns)
    print(tuple(edge.weight for edge in graph.edges))
    solution = PyMatchingGraphDecoder(graph).decode((0, 1))
    print(solution.fault_ids, solution.edge_indices, solution.weight)


if __name__ == "__main__":
    main()
