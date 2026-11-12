"""Factored pure suppliers share the existing graph-owned numerical namespace."""

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)

pytestmark = pytest.mark.unit
Connections = Mapping[str, tuple[tuple[int, ...], ...]]


def fixture(width: int, shape: tuple[int, ...]) -> ProgramRecord:
    bit = RegisterType("bit", 1)
    leaf = bit_gate_function(
        "h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    signature = Signature(
        (Register("q", bit, (2,)), Register("idle", RegisterType("uint", width), shape))
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    words = np.asarray(builder.inputs["q"]).copy()
    for index in (0, 1, 0):
        words[index] = builder.call_dict("h", {"q": words[index]})["q"]
    return builder.finalize({"q": words, "idle": builder.inputs["idle"]})


def order(record: ProgramRecord) -> tuple[int, ...]:
    return tuple(i for i, _, _ in iter_component_calls(record))


def h_supplier(
    function: FunctionRecord, incoming: Connections, outgoing: Connections
) -> tuple[TensorNetworkNode, ...]:
    matrix = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / 2**0.5
    return (
        bind_register_tensor(
            matrix, Signature(function.registers or ()), incoming, outgoing
        ),
    )


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
@pytest.mark.parametrize("width", [0, 1, 3])
def test_internal_labels_do_not_alias_future_calls_or_idle_edges(
    width: int, shape: tuple[int, ...]
) -> None:
    record = fixture(width, shape)
    like = torch.empty((), dtype=torch.complex128)
    private: list[int] = []
    exterior: set[int] = set()

    def factored(
        function: FunctionRecord,
        incoming: Connections,
        outgoing: Connections,
        fresh: Callable[[], int],
    ) -> tuple[TensorNetworkNode, ...]:
        node = h_supplier(function, incoming, outgoing)[0]
        middle = fresh()
        private.append(middle)
        exterior.update((*node.labels,))
        return TensorNetworkNode(
            node.tensor, (node.labels[0], middle)
        ), TensorNetworkNode(torch.eye(2, dtype=like.dtype), (middle, node.labels[1]))

    old_nodes, old_in, old_out = component_tensor_nodes(
        record, order(record), h_supplier, like
    )
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        record, order(record), factored, like
    )
    exterior.update(
        label
        for ports in (incoming, outgoing)
        for words in ports.values()
        for word in words
        for label in word
    )
    assert len(private) == len(set(private)) == 3
    assert not set(private) & exterior
    assert all(
        sum(node.labels.count(label) for node in nodes) == 2 for label in private
    )
    signature = Signature(record.functions[0].registers or ())
    torch.testing.assert_close(
        contract_register_tensors(nodes, signature, incoming, outgoing),
        contract_register_tensors(old_nodes, signature, old_in, old_out),
        atol=0,
        rtol=0,
    )
    # The numerical namespace is local to one construction, not process-global.
    assert component_tensor_nodes_with_labels(record, order(record), factored, like)[
        1:
    ] == (incoming, outgoing)


def test_original_callback_arity_and_errors_are_preserved() -> None:
    record = fixture(1, ())
    error = RuntimeError("supplier failure")

    def refuse(
        function: FunctionRecord,
        incoming: Connections,
        outgoing: Connections,
        fresh: Callable[[], int],
    ) -> Iterable[TensorNetworkNode | int]:
        raise error

    with pytest.raises(RuntimeError) as caught:
        component_tensor_nodes_with_labels(
            record, order(record), refuse, torch.empty(())
        )
    assert caught.value is error


def test_pure_discard_stops_before_requesting_more_supplier_items() -> None:
    record = fixture(1, ())

    def discard(
        function: FunctionRecord,
        incoming: Connections,
        outgoing: Connections,
        fresh: Callable[[], int],
    ) -> Iterable[TensorNetworkNode | int]:
        yield fresh()
        raise AssertionError("pure traversal requested data after discard")

    with pytest.raises(ValueError, match="tried to discard information"):
        component_tensor_nodes_with_labels(
            record, order(record), discard, torch.empty(())
        )


def test_bad_order_is_rejected_before_allocating_supplier_edges() -> None:
    record = fixture(1, ())

    def refuse(
        function: FunctionRecord,
        incoming: Connections,
        outgoing: Connections,
        fresh: Callable[[], int],
    ) -> Iterable[TensorNetworkNode | int]:
        raise AssertionError("invalid traversal reached supplier")

    with pytest.raises(ValueError):
        component_tensor_nodes_with_labels(record, (2, 0, 1), refuse, torch.empty(()))
