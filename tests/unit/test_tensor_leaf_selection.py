"""Selected custom action leaves and expanded gate leaves share one consumer."""

from collections import Counter
from collections.abc import Mapping

import numpy as np
import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor_node,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)
from tests.unit.test_classical_action_tensor import decode, encode
from tests.unit.test_component_tensor_expansion import fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_original_selected_graph_consumes_action_or_gate_supplier_without_rewiring(
    depth: int,
    full: bool,
    dtype: torch.dtype,
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction

    record, source = fixture(depth, 2)
    classified: list[str] = []

    def custom(name: str) -> bool:
        classified.append(name)
        return name in {"custom", "h"}

    selected, labels = expand_components_for_tensor(record, custom, full_flatten=full)
    original = flatten_for_tensor_contraction(source, full_flatten=full)
    instances = list(original.iter_bloqsoqs())
    assert labels == tuple(instance.i for instance, _, _ in instances)
    expected_names = ["h" if full else "custom"] * (2**depth * (2 if full else 1))
    assert [
        "h" if type(instance.bloq).__name__ == "Hadamard" else "custom"
        for instance, _, _ in instances
    ] == expected_names
    assert bool(classified) is not full
    seen: list[str] = []
    like = torch.empty((), dtype=dtype)

    def supply(
        function: FunctionRecord,
        inputs: Mapping[str, tuple[tuple[int, ...], ...]],
        outputs: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        seen.append(function.name)
        sig = Signature(function.registers or ())
        if function.name == "custom":
            return (
                classical_action_tensor_node(
                    sig,
                    inputs,
                    outputs,
                    lambda **values: (values["q"],),
                    decode,
                    encode,
                    like,
                    operation_name="identity_action",
                ),
            )
        assert function.name == "h"
        matrix = torch.tensor([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2)
        return (bind_register_tensor(matrix, sig, inputs, outputs),)

    order = tuple(i for i, _, _ in iter_component_calls(selected))
    nodes, left, right = component_tensor_nodes(selected, order, supply, like)
    assert seen == expected_names
    sig = Signature(selected.functions[0].registers or ())
    matrix = contract_register_tensors(nodes, sig, left, right)
    np.testing.assert_allclose(
        matrix.numpy(), bloq_to_dense(original, full_flatten=False), atol=2e-14, rtol=0
    )
    np.testing.assert_allclose(
        matrix.numpy(), bloq_to_dense(source, full_flatten=full), atol=2e-14, rtol=0
    )

    # Replace only the leaf implementation; selected graph, consumer and bindings stay identical.
    def replacement(
        function: FunctionRecord,
        inputs: Mapping[str, tuple[tuple[int, ...], ...]],
        outputs: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        data = (
            np.eye(2)
            if function.name == "custom"
            else np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        )
        return (
            bind_register_tensor(
                torch.as_tensor(data, dtype=dtype),
                Signature(function.registers or ()),
                inputs,
                outputs,
            ),
        )

    other, other_left, other_right = component_tensor_nodes(
        selected, order, replacement, like
    )
    assert (other_left, other_right) == (left, right)
    torch.testing.assert_close(
        contract_register_tensors(other, sig, other_left, other_right), matrix
    )
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    state = torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128)
    state = state / state.norm() * torch.exp(1j * theta)
    target = matrix.to(torch.complex128) @ state
    torch.testing.assert_close(target, state, atol=2e-14, rtol=0)
    (gradient,) = torch.autograd.grad(target.real.sum(), (theta,))
    torch.testing.assert_close(gradient, -target.imag.sum())

    bound = bind_register_program(selected, ((0,),), clean=(2,))
    assert bound == bind_register_program(record, ((0,),), clean=(2,))
    counts = Counter(op.name for op in bound.circuit.instructions)
    assert counts == {"h": 2 ** (depth + 1)}
    assert get_cost_value(source, QECGatesCost()).clifford == counts["h"]
    initial = torch.tensor([1, 0, 1j, 0, -2j, 0, 3, 0], dtype=torch.complex128)
    initial /= initial.norm()
    result = (
        fq.Circuit(3, inputs=initial, dtype=initial.dtype)
        .append(FieldOperation(3, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(result, initial, atol=2e-14, rtol=0)
    assert torch.count_nonzero(result[1::2]) == 0


@pytest.mark.parametrize("full", [False, True])
def test_selected_missing_supplier_propagates_original_error_without_changing_record(
    full: bool,
) -> None:
    record, _ = fixture(2, 2)
    selected, _ = expand_components_for_tensor(
        record, lambda name: name in {"custom", "h"}, full_flatten=full
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected))
    error = LookupError("required selected supplier unavailable")
    calls: list[str] = []

    def unavailable(
        function: FunctionRecord,
        inputs: Mapping[str, tuple[tuple[int, ...], ...]],
        outputs: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        calls.append(function.name)
        raise error

    before = repr(selected)
    with pytest.raises(LookupError) as observed:
        component_tensor_nodes(selected, order, unavailable, torch.empty(()))
    assert observed.value is error
    assert calls == ["h" if full else "custom"]
    assert repr(selected) == before
