"""Inspect and draw the real source-factorized native CNOT network."""

from functools import partial
from pathlib import Path

import torch

from examples.algorithms.factorized_controlled_not import (
    controlled_not_chain,
    factorized_cnot_supply,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.tensor_network import (
    draw_tensor_network,
    export_tensor_network_dot,
    tensor_index_occurrences,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_indices import (
    register_tensor_outer_indices,
)


def inspection_example() -> tuple[tuple[TensorNetworkNode, ...], dict[int, str]]:
    """Return one CNOT's actual COPY/XOR leaves and native friendly port names."""
    record = controlled_not_chain(1)
    like = torch.empty((), dtype=torch.complex128)
    order = tuple(i for i, _, _ in iter_component_calls(record, "cnot_chain"))
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        record, order, partial(factorized_cnot_supply, like=like), like, "cnot_chain"
    )
    descriptors = register_tensor_outer_indices(
        Signature(record.functions[0].registers or ()),
        incoming,
        outgoing,
        friendly=True,
    )
    names = {
        label: value for label, value in descriptors.items() if isinstance(value, str)
    }
    return nodes, names


def main(directory: Path | None = None) -> None:
    """Print topology and optionally save DOT/PNG; numerical data stays unchanged."""
    nodes, names = inspection_example()
    occurrences = tensor_index_occurrences(nodes)
    outer = tuple(
        label for label, endpoints in occurrences.items() if len(endpoints) == 1
    )
    inner = tuple(
        label for label, endpoints in occurrences.items() if len(endpoints) > 1
    )
    print(
        f"CNOT: {len(nodes)} tensors, {len(outer)} outer indices, {len(inner)} inner index"
    )
    if directory is not None:
        from matplotlib import pyplot as plt

        directory.mkdir(parents=True, exist_ok=True)
        (directory / "controlled-not.dot").write_text(
            export_tensor_network_dot(nodes, index_names=names)
        )
        figure, _ = draw_tensor_network(
            nodes, index_names=names, node_colors={"COPY": "#b6d9ff", "XOR": "#ffd39e"}
        )
        try:
            figure.savefig(directory / "controlled-not.png", dpi=160)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    main()
