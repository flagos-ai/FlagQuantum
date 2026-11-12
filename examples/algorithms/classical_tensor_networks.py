"""Bounded contraction of the native classical cross-check's closed graphs.

The explicit cotengra path is diagnostic example policy, not an automatic
backend fallback. Compiler still selects calls; Simulation wires and contracts.
"""

import torch

from examples.algorithms.classical_tensor_crosscheck import (
    crosscheck_providers,
    prepare_classical_crosscheck,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.contraction import (
    _contract_nodes_with_slicing_plan,
    _cotengra_slicing_plan,
)
from flagquantum.simulation.tensor_network.models import (
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)


def select_crosscheck_graph(record: ProgramRecord, kind: str) -> ProgramRecord:
    """Expand stored calls down to this example's declared logical tensor leaves.

    AND and its measurement-assisted adjoint retain their logical tensors. This
    selection is not physical gate flattening and does not synthesize a circuit.
    """
    providers = crosscheck_providers(
        record, kind, torch.empty((), dtype=torch.complex128)
    )
    selected, _ = expand_components_for_tensor(
        record, providers.has_custom_tensor, "classical_vectors", full_flatten=False
    )
    return selected


def closed_crosscheck_network(
    record: ProgramRecord, kind: str
) -> tuple[tuple[TensorNetworkNode, ...], TensorNetworkSlicingPlan]:
    """Build and explicitly plan a closed network with at most 1,024 peak elements.

    Returned peak_size counts complex tensor elements, not resident process RAM.
    Open ports are refused; there is no dense whole-Add conversion. The existing
    Simulation planner owns budget refusal, sliced execution and differentiation.
    """
    like = torch.empty((), dtype=torch.complex128)
    selected = select_crosscheck_graph(record, kind)
    providers = crosscheck_providers(selected, kind, like)
    order = tuple(i for i, _, _ in iter_component_calls(selected, "classical_vectors"))
    nodes, incoming, outgoing = component_tensor_nodes(
        selected, order, providers.supply, like, "classical_vectors"
    )
    if incoming or outgoing:
        raise ValueError("classical cross-check network requires closed ports")
    plan = _cotengra_slicing_plan(
        nodes, (), target_peak_elements=1024, methods=("greedy",), max_repeats=1
    )
    return nodes, plan


def contract_closed_crosscheck(record: ProgramRecord, kind: str) -> torch.Tensor:
    """Contract a prepared closed example with the explicit bounded plan."""
    nodes, plan = closed_crosscheck_network(record, kind)
    return _contract_nodes_with_slicing_plan(nodes, (), plan)


def main() -> None:
    """Contract the original notebook's Add(5) and Add(32) concrete inputs."""
    for width, a, b in ((5, 2, 3), (32, 5555, 6666)):
        record = prepare_classical_crosscheck("add", {"a": a, "b": b}, width=width)
        nodes, plan = closed_crosscheck_network(record, "add")
        result = _contract_nodes_with_slicing_plan(nodes, (), plan)
        torch.testing.assert_close(result, torch.ones_like(result), atol=2e-12, rtol=0)
        print(
            f"Add({width}): {result.item()}, planned peak {plan.peak_size} elements, {plan.n_slices} slices"
        )


if __name__ == "__main__":
    main()
