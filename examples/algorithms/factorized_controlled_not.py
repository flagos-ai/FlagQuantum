"""Use source COPY/XOR leaves with the existing native graph and contraction."""

from collections.abc import Callable, Mapping

import torch

from examples.algorithms.classical_tensor_crosscheck import crosscheck_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.controlled_not import controlled_not_nodes
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)


def controlled_not_chain(repetitions: int) -> ProgramRecord:
    """Compose repeated real CNOT bodies without changing their physical lowering."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("CNOT repetitions must be a positive built-in integer")
    leaf = crosscheck_program("cnot")
    signature = Signature(leaf.functions[0].registers or ())
    builder = ComponentProgramBuilder(signature, leaf.functions, name="cnot_chain")
    values = dict(builder.inputs)
    for _ in range(repetitions):
        values = builder.call_dict("main", values)
    return builder.finalize(values)


def factorized_cnot_supply(
    function: FunctionRecord,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    fresh_label: Callable[[], int],
    *,
    like: torch.Tensor,
) -> tuple[TensorNetworkNode, ...]:
    """Supply the explicitly selected CNOT leaf; refuse unknown definitions."""
    if function != crosscheck_program("cnot").functions[0]:
        raise ValueError("factorized CNOT example requires its declared primitive body")
    return controlled_not_nodes(incoming, outgoing, fresh_label, like)


def main() -> None:
    """Contract three CNOT calls using six rank-three source tensor leaves."""
    from functools import partial

    record = controlled_not_chain(3)
    like = torch.empty((), dtype=torch.complex128)
    order = tuple(i for i, _, _ in iter_component_calls(record, "cnot_chain"))
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        record, order, partial(factorized_cnot_supply, like=like), like, "cnot_chain"
    )
    signature = Signature(record.functions[0].registers or ())
    actual = contract_register_tensors(nodes, signature, incoming, outgoing)
    assert actual.shape == (4, 4)
    assert len(nodes) == 6
    assert torch.equal(actual @ actual, torch.eye(4, dtype=like.dtype))
    print(
        f"CNOT chain: {len(nodes)} nodes, {sum(node.tensor.numel() for node in nodes)} logical tensor elements"
    )


if __name__ == "__main__":
    main()
