# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Composition follows Qualtran v0.7.0 MultiAnd._decompose_via_tree.

"""Compose real predicate bodies and explicit source-ordered And tensor leaves."""

from collections.abc import Mapping

import torch

from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.control.conjunction_program import (
    conjunction_program,
)
from flagquantum.compiler.registers.control.multi_control import multi_and_stages
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.conjunction import conjunction_tensor_node
from flagquantum.simulation.tensor_network.models import TensorNetworkNode

Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def conjunction_providers(
    values: tuple[int, ...],
    like: torch.Tensor,
) -> ComponentTensorProviders[[Connections, Connections]]:
    """Bind exact known predicate definitions to their activation patterns."""
    record = conjunction_program(values)
    definitions = {f.name: f for f in record.functions}
    patterns = {f"and_{p[0]}{p[1]}": p for *_, p in multi_and_stages(values)}

    def supply(
        function: FunctionRecord, incoming: Connections, outgoing: Connections
    ) -> tuple[TensorNetworkNode, ...]:
        if function != definitions[function.name]:
            raise ValueError("conjunction example requires its declared predicate body")
        return (
            conjunction_tensor_node(
                incoming, outgoing, like, values=patterns[function.name]
            ),
        )

    return ComponentTensorProviders(dict.fromkeys(patterns, supply))


def main() -> None:
    """Contract the original notebook's four-control, three-leaf ladder."""
    values = (1, 1, 1, 1)
    record = conjunction_program(values)
    like = torch.empty((), dtype=torch.complex128)
    providers = conjunction_providers(values, like)
    order = tuple(i for i, _, _ in iter_component_calls(record, "multi_and"))
    result = contract_component_tensors(
        record, order, providers.supply, like, "multi_and"
    )
    assert result.shape == (128, 16)
    print(
        f"MultiAnd: {len(order)} source-ordered And tensors, dense shape {tuple(result.shape)}"
    )


if __name__ == "__main__":
    main()
