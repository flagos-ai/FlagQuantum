"""Exact locked And axes, full MultiAnd isometries and every erasure branch."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.conjunction_tensors import (
    conjunction_program,
    conjunction_providers,
    main,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.control.multi_control import multi_and_ir
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.tensor_network import tensor_index_occurrences
from flagquantum.simulation.classical_and import and_classical_values
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.conjunction import conjunction_tensor_node
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PATTERNS = [p for n in (2, 3, 4) for p in product((0, 1), repeat=n)]


@pytest.mark.parametrize("values", list(product((0, 1), repeat=2)))
@pytest.mark.parametrize("uncompute", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_original_leaf_data_axis_order_name_and_adjoint(
    values: tuple[int, int], uncompute: bool, dtype: torch.dtype
) -> None:
    from qualtran.bloqs.mcmt import And

    incoming: dict[str, tuple[tuple[int, ...], ...]] = {"ctrl": ((8,), (2,))}
    outgoing: dict[str, tuple[tuple[int, ...], ...]] = {"ctrl": ((6,), (1,))}
    (incoming if uncompute else outgoing)["target"] = ((9,),)
    node = conjunction_tensor_node(
        incoming,
        outgoing,
        torch.empty((), dtype=dtype),
        values=values,
        uncompute=uncompute,
    )
    inc: dict[str, Any] = {"ctrl": np.array([8, 2])}
    out: dict[str, Any] = {"ctrl": np.array([6, 1])}
    (inc if uncompute else out)["target"] = 9
    source = And(*values, uncompute=uncompute).my_tensors(inc, out)[0]
    assert node.labels == tuple(i[0] for i in source.inds) == (8, 2, 6, 1, 9)
    assert set(source.tags) == {node.name}
    np.testing.assert_array_equal(node.tensor.numpy(), source.data)
    assert node.tensor.dtype == dtype and node.tensor.device.type == "cpu"


@pytest.mark.parametrize("values", PATTERNS)
def test_complete_ladder_tensor_topology_physical_gates_phase_and_cleanup(
    values: tuple[int, ...],
) -> None:
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.simulation.tensor import cbloq_to_quimb

    n = len(values)
    source = And(*values) if n == 2 else MultiAnd(values)
    composite = source.as_composite_bloq() if n == 2 else source.decompose_bloq()
    original = cbloq_to_quimb(composite)
    record = conjunction_program(values)
    like = torch.empty((), dtype=torch.complex128)
    providers = conjunction_providers(values, like)
    order = tuple(i for i, _, _ in iter_component_calls(record, "multi_and"))
    nodes, inc, out = component_tensor_nodes(
        record, order, providers.supply, like, "multi_and"
    )
    assert len(nodes) == len(original.tensors) == n - 1
    for node, expected in zip(nodes, original.tensors, strict=True):
        assert node.name in expected.tags
        np.testing.assert_array_equal(node.tensor.numpy(), expected.data)
    source_edges: dict[object, list[tuple[int, int]]] = {}
    for i, tensor in enumerate(original.tensors):
        for axis, label in enumerate(tensor.inds):
            source_edges.setdefault(label, []).append((i, axis))
    assert tuple(tensor_index_occurrences(nodes).values()) == tuple(
        tuple(v) for v in source_edges.values()
    )
    matrix = contract_register_tensors(
        nodes, Signature(record.functions[0].registers or ()), inc, out
    )

    # Replace every exact-axis provider with the existing classical table owner;
    # the graph consumer and compiler definitions are unchanged.
    def alternate(
        function: FunctionRecord,
        incoming: Mapping[str, tuple[tuple[int, ...], ...]],
        outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.registers is not None
        signature = Signature(function.registers)
        pattern = {
            "and_00": (0, 0),
            "and_01": (0, 1),
            "and_10": (1, 0),
            "and_11": (1, 1),
        }[function.name]
        data = classical_action_tensor(
            signature,
            lambda **v: tuple(
                and_classical_values(v["ctrl"], (pattern[0], pattern[1])).values()
            ),
            operation_name="replacement_and",
        )
        return (
            bind_register_tensor(
                torch.as_tensor(data, dtype=like.dtype), signature, incoming, outgoing
            ),
        )

    other_nodes, other_in, other_out = component_tensor_nodes(
        record, order, alternate, like, "multi_and"
    )
    other = contract_register_tensors(
        other_nodes, Signature(record.functions[0].registers or ()), other_in, other_out
    )
    torch.testing.assert_close(other, matrix, rtol=0, atol=0)
    np.testing.assert_array_equal(matrix.numpy(), source.tensor_contract())
    assert matrix.shape == (2 ** (2 * n - 1), 2**n)
    torch.testing.assert_close(
        matrix.mH @ matrix, torch.eye(2**n, dtype=like.dtype), rtol=0, atol=0
    )

    controls = tuple(range(n))
    work = tuple(range(n, 2 * n - 1))
    bound = bind_register_program(
        record,
        tuple((i,) for i in controls),
        clean=work,
        entry="multi_and",
        allow_workspace_outputs=True,
    )
    gates = bound.circuit.instructions
    assert gates == multi_and_ir(controls, values, work)
    outputs = {"ctrl": controls, "target": work[-1:]}
    if n > 2:
        outputs["junk"] = work[:-1]
    expected_gates = reference_arithmetic_gates(
        composite, {"ctrl": controls}, work, {}, expected_outputs=outputs
    )
    assert_source(gates, expected_gates)
    physical = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert physical["t"] + physical["tdg"] == model.total_t_count() == 4 * (n - 1)
    assert physical["measure"] == model.measurement == 0
    assert get_cost_value(source, QubitCount()) == 2 * n - 1
    # A spectator outside the bound pool must remain entangled and untouched.
    phase = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    indices = torch.arange(2 ** (n + 1), dtype=torch.float64)
    amplitudes = (indices + 1) * torch.exp(1j * phase * indices)
    amplitudes = amplitudes / amplitudes.norm()
    initial = torch.zeros(2 ** (2 * n), dtype=torch.complex128)
    initial.reshape(2**n, 2 ** (n - 1), 2)[:, 0, :] = amplitudes.reshape(2**n, 2)
    expected_state = (matrix @ amplitudes.reshape(2**n, 2)).reshape(-1)
    actual = (
        Circuit(2 * n, inputs=initial, dtype=like.dtype)
        .append(FieldOperation(2 * n, gates))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected_state, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual.real.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected_state.real.sum(), phase, retain_graph=True)[0],
        rtol=0,
        atol=2e-13,
    )
    erase = multi_and_ir(
        controls, values, work, classical_bits=tuple(range(n - 1)), uncompute=True
    )
    for history in product((0, 1), repeat=n - 1):
        branch = forced_branch(actual, erase, history)
        torch.testing.assert_close(
            branch, initial / np.sqrt(2 ** (n - 1)), rtol=0, atol=2e-14
        )
        assert torch.count_nonzero(branch.reshape(2**n, 2 ** (n - 1), 2)[:, 1:, :]) == 0


def test_supplier_replacement_and_unknown_definition_refusal() -> None:
    record = conjunction_program((1, 1, 1, 1))
    providers = conjunction_providers(
        (1, 1, 1, 1), torch.empty((), dtype=torch.complex128)
    )
    assert providers.has_custom_tensor("and_11")
    assert not providers.has_custom_tensor("multi_and")
    leaf = next(f for f in record.functions if f.name == "and_11")
    with pytest.raises(ValueError, match="declared predicate"):
        tuple(providers.supply(replace(leaf, name="and_11", blocks=()), {}, {}))
    with pytest.raises(ValueError, match="no tensor provider"):
        tuple(providers.supply(record.functions[0], {}, {}))


@pytest.mark.parametrize(
    "values,uncompute",
    [((1, True), False), ((1,), False), ((0, 2), False), ((1, 1), 1)],
)
def test_refuse_invalid_activation_and_direction(
    values: tuple[int, int], uncompute: Any
) -> None:
    with pytest.raises(ValueError):
        conjunction_tensor_node(
            {}, {}, torch.empty(()), values=values, uncompute=uncompute
        )


def test_reject_malformed_ports() -> None:
    with pytest.raises(ValueError, match="counts"):
        conjunction_tensor_node(
            {"ctrl": ((0,),)},
            {"ctrl": ((1,), (2,)), "target": ((3,),)},
            torch.empty(()),
        )


def test_example(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert (
        "3 source-ordered And tensors, dense shape (128, 16)" in capsys.readouterr().out
    )
