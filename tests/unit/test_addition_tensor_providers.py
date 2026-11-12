"""Explicit Add recipes replace test-only dispatch without changing consumers."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
import torch

from examples.algorithms.factorized_addition_tensors import addition_tensor_providers
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_identity import (
    register_bit_wiring_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from tests.unit.test_register_tensor import native_signature
from tests.unit.test_register_tensor_binding import source_bindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Connections = Mapping[str, tuple[tuple[int, ...], ...]]


@pytest.mark.parametrize("width", [1, 2, 3, 5])
@pytest.mark.parametrize("join", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_rebinding_binary_nodes_match_original_complete_tensor_and_gradient(
    width: int, join: bool, dtype: torch.dtype
) -> None:
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Join, Split

    source = Join(QAny(width)) if join else Split(QAny(width))
    signature = native_signature(source.signature)
    labels: dict[tuple[str, int], int] = {}
    source_in, incoming = source_bindings(source.signature.lefts(), "in", labels)
    source_out, outgoing = source_bindings(source.signature.rights(), "out", labels)
    nodes = register_bit_wiring_nodes(
        signature, incoming, outgoing, torch.empty((), dtype=dtype)
    )
    original_nodes = source.my_tensors(source_in, source_out)
    assert len(nodes) == len(original_nodes) == width
    for node, original in zip(nodes, original_nodes, strict=True):
        np.testing.assert_array_equal(node.tensor.numpy(), original.data)
        assert node.labels == tuple(labels[index] for index in original.inds)
    assert all(
        node.tensor.numel() == 4 and node.tensor.dtype == dtype for node in nodes
    )
    actual = contract_register_tensors(nodes, signature, incoming, outgoing)
    np.testing.assert_array_equal(actual.numpy(), source.tensor_contract())
    state = torch.arange(1, (1 << width) + 1, dtype=torch.float64).to(dtype)
    if dtype.is_complex:
        state = state + 1j * state.flip(0)
    state.requires_grad_()
    output = actual @ state
    torch.testing.assert_close(output, state, atol=0, rtol=0)
    gradient = torch.autograd.grad(output.abs().square().sum(), state)[0]
    torch.testing.assert_close(gradient, 2 * state)


@pytest.mark.parametrize("direction", [RegisterSide.LEFT, RegisterSide.RIGHT])
def test_wiring_cannot_create_or_discard_live_bits(direction: RegisterSide) -> None:
    signature = Signature((Register("q", RegisterType("bit", 1), side=direction),))
    with pytest.raises(ValueError, match="equal input and output"):
        register_bit_wiring_nodes(
            signature, {"q": ((1,),)}, {"q": ((2,),)}, torch.empty(())
        )


def test_empty_wiring_keeps_no_scalar_and_aliases_are_refused() -> None:
    assert register_bit_wiring_nodes(Signature(()), {}, {}, torch.empty(())) == ()
    with pytest.raises(ValueError, match="distinct"):
        register_bit_wiring_nodes(
            Signature.build(q=1), {"q": ((1,),)}, {"q": ((1,),)}, torch.empty(())
        )


@pytest.mark.parametrize("width", [1, 2, 3])
def test_explicit_provider_replacement_preserves_complex_result_and_program(
    width: int,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    graph = addition_component_program(width)
    original = graph
    like = torch.empty((), dtype=torch.complex128)
    recipes = addition_tensor_providers(graph, {}, like)
    selected, _ = expand_components_for_tensor(
        graph, recipes.has_custom_tensor, full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected))
    baseline = contract_component_tensors(selected, order, recipes.supply, like)
    torch.testing.assert_close(
        baseline,
        torch.as_tensor(Add(QUInt(width)).tensor_contract()),
        atol=2e-12,
        rtol=0,
    )
    phase = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)

    def alternate(
        function: FunctionRecord, incoming: Connections, outgoing: Connections
    ) -> tuple[TensorNetworkNode, ...]:
        nodes = tuple(recipes.supply(function, incoming, outgoing))
        assert all(isinstance(node, TensorNetworkNode) for node in nodes)
        first = nodes[0]
        assert isinstance(first, TensorNetworkNode)
        return (
            TensorNetworkNode(first.tensor * torch.exp(1j * phase), first.labels),
            *(node for node in nodes[1:] if isinstance(node, TensorNetworkNode)),
            TensorNetworkNode(torch.exp(-1j * phase), ()),
        )

    replacement = ComponentTensorProviders(
        {
            f.name: alternate
            for f in selected.functions
            if recipes.has_custom_tensor(f.name)
        }
    )
    actual = contract_component_tensors(selected, order, replacement.supply, like)
    torch.testing.assert_close(actual, baseline, atol=2e-12, rtol=0)
    derivative = torch.autograd.grad(actual.real.sum(), phase)[0]
    torch.testing.assert_close(
        derivative, torch.zeros_like(derivative), atol=2e-12, rtol=0
    )
    assert graph == original


def test_name_collision_unknown_leaf_and_permuted_rebind_do_not_silently_fallback() -> (
    None
):
    graph = addition_component_program(2)
    like = torch.empty((), dtype=torch.complex128)
    with pytest.raises(ValueError, match="override"):
        addition_tensor_providers(graph, {"add_cx": (0, True)}, like)
    recipes = addition_tensor_providers(graph, {}, like)
    with pytest.raises(ValueError, match="no tensor provider"):
        recipes.supply(replace(graph.functions[0], name="unknown"), {}, {})
    split = next(
        f
        for f in graph.functions
        if f.registers
        and len(f.registers) == 2
        and len(f.registers[1].shape) == 1
        and len(f.blocks[0].return_values or ()) > 1
        and recipes.has_custom_tensor(f.name)
        and f.registers[0].name == "reg"
    )
    block = split.blocks[0]
    permuted = replace(
        split,
        blocks=(
            replace(block, return_values=tuple(reversed(block.return_values or ()))),
        ),
    )
    changed = replace(
        graph,
        functions=tuple(
            permuted if f.name == split.name else f for f in graph.functions
        ),
    )
    assert not addition_tensor_providers(changed, {}, like).has_custom_tensor(
        split.name
    )
