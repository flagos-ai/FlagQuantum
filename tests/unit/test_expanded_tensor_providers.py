"""Expanded provider consumers use positions, never persistent instance IDs."""

from collections import Counter
from itertools import product

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.simulation.matrices import H_MATRIX
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_superoperator,
    contract_component_tensors,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.component_tensor_providers import (
    ComponentTensorProviders,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_expansion import _hierarchy
from tests.unit.test_component_superoperator_nodes import mixed_graph, supplier
from tests.unit.test_component_tensor_nodes import h_supplier
from tests.unit.test_tensor_root_entry import source_root

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("depth,repeats", [(2, 1), (2, 2), (3, 1), (3, 2)])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("root", [False, True])
def test_expanded_provider_ids_positions_source_tensors_phase_and_append(
    depth: int, repeats: int, full: bool, root: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    record, source = _hierarchy(depth, repeats)
    providers = ComponentTensorProviders({"h": h_supplier})
    entry = "tensor_root" if root else "main"
    expand = expand_component_root_for_tensor if root else expand_components_for_tensor
    selected, instance_ids = expand(
        record, providers.has_custom_tensor, full_flatten=full
    )
    positions = tuple(i for i, _, _ in iter_component_calls(selected, entry))
    original = source_root(source) if root else source
    flattened = flatten_for_tensor_contraction(original, full_flatten=full)
    assert instance_ids == tuple(i.i for i, _, _ in flattened.iter_bloqsoqs())
    assert instance_ids != positions
    like = torch.empty((), dtype=torch.complex128)
    with pytest.raises(ValueError, match="order"):
        contract_component_tensors(
            selected, instance_ids, providers.supply, like, entry
        )
    actual = contract_component_tensors(
        selected, positions, providers.supply, like, entry
    )
    np.testing.assert_allclose(
        actual.numpy(), original.tensor_contract(), rtol=0, atol=2e-14
    )
    nodes, _, _ = component_tensor_nodes(
        selected, positions, providers.supply, like, entry
    )
    reference = cbloq_to_quimb(flattened)
    assert len(nodes) == len(reference.tensors) == repeats**depth
    for node, tensor in zip(nodes, reference.tensors, strict=True):
        np.testing.assert_array_equal(node.tensor.numpy(), tensor.data)
    baseline = bind_register_program(record, ((0,),), clean=(2,))
    bound = bind_register_program(selected, ((0,),), clean=(2,), entry=entry)
    assert bound.circuit.instructions == baseline.circuit.instructions
    gates = (
        Circuit(3)
        .append(FieldOperation(3, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert gates == bound.circuit.instructions
    assert Counter(g.name for g in gates) == {"h": repeats**depth}
    model = get_cost_value(original, QECGatesCost())
    assert (
        model.clifford == repeats**depth
        and model.measurement == model.total_t_count() == 0
    )
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    state = torch.arange(1, 5, dtype=torch.float64) * torch.exp(
        1j * theta * torch.arange(4)
    )
    state = state / state.norm()
    output = actual @ state.reshape(2, 2)
    expected = torch.as_tensor(original.tensor_contract()) @ state.reshape(2, 2)
    torch.testing.assert_close(output, expected, atol=2e-14, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(output.real.sum(), theta, retain_graph=True)[0],
        torch.autograd.grad(expected.real.sum(), theta)[0],
        atol=2e-14,
        rtol=0,
    )
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = state.detach()
    branch = forced_branch(initial, gates, ()).reshape(2, 2, 2)
    assert torch.count_nonzero(branch[..., 1]) == 0
    torch.testing.assert_close(branch[..., 0], expected.detach(), atol=2e-14, rtol=0)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("discard", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_expanded_mixed_provider_replacement_all_branches_and_work(
    full: bool, discard: bool, dtype: torch.dtype
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX, MeasureZ
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction

    record, source = mixed_graph(discard)
    like = torch.empty((), dtype=dtype)
    numerical = supplier(like)
    names = ("first", "second", "drop")
    providers = ComponentTensorProviders(dict.fromkeys(names, numerical))
    selected, ids = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, full_flatten=full
    )
    positions = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    original = source_root(source)
    flattened = flatten_for_tensor_contraction(original, full_flatten=full)
    assert (
        ids == tuple(i.i for i, _, _ in flattened.iter_bloqsoqs()) and ids != positions
    )
    actual = contract_component_superoperator(
        selected, positions, providers.supply, like, "tensor_root"
    )
    tol = 2e-6 if dtype == torch.complex64 else 2e-14
    expected_channel = bloq_to_dense(original, full_flatten=full, superoperator=True)
    np.testing.assert_allclose(actual.numpy(), expected_channel, atol=tol, rtol=0)
    replacement = ComponentTensorProviders(
        dict.fromkeys(names, supplier(like, original=True))
    )
    torch.testing.assert_close(
        contract_component_superoperator(
            selected, positions, replacement.supply, like, "tensor_root"
        ),
        actual,
        atol=tol,
        rtol=0,
    )
    state = torch.arange(1, 9, dtype=torch.float64) * torch.exp(
        0.31j * torch.arange(8, dtype=torch.float64)
    )
    state = state / state.norm()
    z = torch.as_tensor(MeasureZ().my_tensors({"q": "q"}, {"c": "c"})[0].data)
    x = torch.as_tensor(MeasureX().my_tensors({"q": "q"}, {"c": "c"})[0].data)
    instrument = torch.einsum(
        "iae,jfb,ijs->abefs",
        z.to(torch.complex128),
        x.to(torch.complex128),
        state.reshape(2, 2, 2),
    )
    for idle in (0, 1):
        baseline = bind_register_program(
            record,
            ((0,), (1,), (2,), ()),
            clean=(3,),
            classical_inputs=(None, None, None, idle),
        )
        bound = bind_register_program(
            selected,
            ((0,), (1,), (2,), ()),
            entry="tensor_root",
            clean=(3,),
            classical_inputs=(None, None, None, idle),
        )
        gates = (
            Circuit(4)
            .append(FieldOperation(4, bound.circuit.instructions))
            .to_ir()
            .instructions
        )
        assert gates == baseline.circuit.instructions
        assert Counter(g.name for g in gates) == {
            "h": 1,
            "measure": 2,
            "classical_xor": 1,
        }
        initial = torch.zeros(16, dtype=torch.complex128)
        initial[::2] = state
        for a, b in product((0, 1), repeat=2):
            branch = forced_branch(initial, gates, (a, b)).reshape(2, 2, 2, 2)
            assert torch.count_nonzero(branch[..., 1]) == 0
            restored = torch.einsum("ij,ejk->eik", H_MATRIX, branch[..., 0])
            torch.testing.assert_close(restored, instrument[a, b], atol=3e-16, rtol=0)
    model = get_cost_value(original, QECGatesCost())
    assert model.measurement == 2 and model.clifford == model.total_t_count() == 0
