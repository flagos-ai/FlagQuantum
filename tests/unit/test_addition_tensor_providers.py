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


def test_automatic_vectors_survive_renaming_and_reject_a_changed_lifetime() -> None:
    from flagquantum.compiler.registers.component_basis_vectors import (
        integer_basis_function_data,
    )
    from flagquantum.compiler.registers.component_classical_vectors import (
        flank_component_with_classical_vectors,
    )

    graph = flank_component_with_classical_vectors(
        addition_component_program(2), {"a": 1, "b": 2}, {"a": 1, "b": 3}
    )
    aliases = {
        f.name: f"vector_alias_{i}"
        for i, f in enumerate(graph.functions)
        if integer_basis_function_data(f) is not None
    }
    assert len(aliases) == 4
    renamed = replace(
        graph,
        functions=tuple(
            replace(
                f,
                name=aliases.get(f.name, f.name),
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(op, callee=aliases.get(op.callee, op.callee))
                                if op.callee is not None
                                else op
                            )
                            for op in block.operations
                        ),
                    )
                    for block in f.blocks
                ),
            )
            for f in graph.functions
        ),
    )
    like = torch.empty((), dtype=torch.complex128)
    automatic = addition_tensor_providers(renamed, None, like)
    selected, _ = expand_components_for_tensor(
        renamed, automatic.has_custom_tensor, "classical_vectors", full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "classical_vectors"))
    result = contract_component_tensors(
        selected, order, automatic.supply, like, "classical_vectors"
    )
    torch.testing.assert_close(
        result, torch.ones((), dtype=like.dtype), atol=2e-12, rtol=0
    )
    # Explicit mappings remain caller-owned; an empty mapping does not discover.
    explicit = addition_tensor_providers(renamed, {}, like)
    assert all(not explicit.has_custom_tensor(name) for name in aliases.values())
    state = next(
        f for f in renamed.functions if integer_basis_function_data(f) == (1, True)
    )
    block = state.blocks[0]
    dirty = replace(
        state,
        blocks=(
            replace(
                block,
                operations=(
                    replace(block.operations[0], workspace_dirty=True),
                    *block.operations[1:],
                ),
            ),
        ),
    )
    changed = replace(
        renamed,
        functions=tuple(
            dirty if f.name == state.name else f for f in renamed.functions
        ),
    )
    rejected = addition_tensor_providers(changed, None, like)
    assert not rejected.has_custom_tensor(state.name)
    with pytest.raises(ValueError, match="no tensor provider"):
        rejected.supply(dirty, {}, {})


def test_provider_refuses_untyped_selected_leaf() -> None:
    graph = addition_component_program(2)
    like = torch.empty((), dtype=torch.complex128)
    recipes = addition_tensor_providers(graph, None, like)
    leaf = next(f for f in graph.functions if f.name == "add_cx")
    with pytest.raises(ValueError, match="require typed functions"):
        recipes.supply(replace(leaf, registers=None), {}, {})
