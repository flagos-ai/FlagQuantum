"""Native action suppliers replace selected quantum leaves without new graphs."""

from collections.abc import Mapping
from itertools import product

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_addition import register_add_classical_values
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
from tests.unit.test_classical_action_tensor import (
    decode,
    encode,
    signature_from_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def add_action(**values: object) -> tuple[object, ...]:
    left, right = values["a"], values["b"]
    assert isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer))
    return register_add_classical_values(left, right, 3)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_selected_add_graph_native_callbacks_leaf_axes_and_replacement(
    depth: int,
    dtype: torch.dtype,
) -> None:
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.simulation.tensor._tensor_from_classical import (
        my_tensors_from_classical_action,
    )

    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_components_for_tensor,
    )
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_register_tensor_binding import source_bindings

    record, source = wrapped_add(depth)
    selected, _ = expand_components_for_tensor(
        record, lambda name: name == "add", full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected))
    sig = Signature(selected.functions[0].registers or ())
    like = torch.empty((), dtype=dtype)
    leaves: list[TensorNetworkNode] = []

    def supply(
        function: FunctionRecord,
        inc: Mapping[str, tuple[tuple[int, ...], ...]],
        out: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        assert function.name == "add"
        node = classical_action_tensor_node(
            Signature(function.registers or ()),
            inc,
            out,
            add_action,
            decode,
            encode,
            like,
            operation_name=function.name,
        )
        leaves.append(node)
        return (node,)

    nodes, inc, out = component_tensor_nodes(selected, order, supply, like)
    assert len(leaves) == 2 and all(node.tensor.dtype == dtype for node in nodes)
    leaf = Add(QUInt(2), QUInt(3))
    labels: dict[tuple[str, int], int] = {}
    src_inc, native_inc = source_bindings(leaf.signature.lefts(), "in", labels)
    src_out, native_out = source_bindings(leaf.signature.rights(), "out", labels)
    original = my_tensors_from_classical_action(leaf, src_inc, src_out)[0]
    direct = classical_action_tensor_node(
        signature_from_source(leaf.signature),
        native_inc,
        native_out,
        add_action,
        decode,
        encode,
        like,
        operation_name="add",
    )
    assert direct.labels == tuple(labels[ind] for ind in original.inds)
    np.testing.assert_array_equal(direct.tensor.numpy(), original.data)
    actual = contract_register_tensors(nodes, sig, inc, out)
    np.testing.assert_array_equal(actual.numpy(), source.tensor_contract())

    # Replacement uses a separately supplied table, through the same consumer.
    def alternative(
        function: FunctionRecord,
        inputs: Mapping[str, tuple[tuple[int, ...], ...]],
        outputs: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        return (
            bind_register_tensor(
                torch.as_tensor(original.data, dtype=dtype),
                Signature(function.registers or ()),
                inputs,
                outputs,
            ),
        )

    replacement, left, right = component_tensor_nodes(
        selected, order, alternative, like
    )
    torch.testing.assert_close(
        contract_register_tensors(replacement, sig, left, right), actual
    )
    assert np.count_nonzero(actual.numpy()) == 64


def test_complex_amplitudes_work_branch_roundtrip_and_append() -> None:
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_components_for_tensor,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_component_tensor_expansion import (
        test_measured_add_consumer_preserves_gates_all_history_pairs_and_work,
    )

    record, _ = wrapped_add(2)
    selected, _ = expand_components_for_tensor(
        record, lambda name: name == "add", full_flatten=False
    )
    like = torch.empty((), dtype=torch.complex128)

    def supply(
        function: FunctionRecord,
        inc: Mapping[str, tuple[tuple[int, ...], ...]],
        out: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[TensorNetworkNode, ...]:
        return (
            classical_action_tensor_node(
                Signature(function.registers or ()),
                inc,
                out,
                add_action,
                decode,
                encode,
                like,
                operation_name=function.name,
            ),
        )

    nodes, inc, out = component_tensor_nodes(
        selected, tuple(i for i, _, _ in iter_component_calls(selected)), supply, like
    )
    matrix = contract_register_tensors(
        nodes, Signature(selected.functions[0].registers or ()), inc, out
    )
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    state = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    state = state / state.norm() * torch.exp(1j * theta)
    expected = matrix @ state
    (gradient,) = torch.autograd.grad(expected.real.sum(), (theta,))
    torch.testing.assert_close(gradient, -expected.imag.sum())
    bound = bind_register_program(record, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::4] = state.detach()
    for history in product((0, 1), repeat=4):
        result = forced_branch(initial, bound.circuit.instructions, history).reshape(
            64, 4
        )
        torch.testing.assert_close(
            result[:, 0], expected.detach() / 4, atol=3e-14, rtol=0
        )
        assert torch.count_nonzero(result[:, 1:]) == 0
    test_measured_add_consumer_preserves_gates_all_history_pairs_and_work(False)


@pytest.mark.parametrize(
    "side", [RegisterSide.LEFT, RegisterSide.RIGHT, RegisterSide.THRU]
)
def test_directional_bit_leaf_matches_original_supplier(side: RegisterSide) -> None:
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Side
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        my_tensors_from_classical_action,
    )

    from tests.unit.test_register_tensor_binding import source_bindings

    sig = Signature([Register("q", quantum_bit_type(), side=side)])
    source_sig = SourceSignature([SourceRegister("q", QBit(), side=Side[side.name])])

    def action(**values: object) -> tuple[object, ...]:
        return () if side == RegisterSide.LEFT else (1,)

    class Source:
        signature = source_sig
        call_classically = staticmethod(action)

    labels: dict[tuple[str, int], int] = {}
    src_in, inc = source_bindings(source_sig.lefts(), "in", labels)
    src_out, out = source_bindings(source_sig.rights(), "out", labels)
    original = my_tensors_from_classical_action(Source(), src_in, src_out)[0]
    node = classical_action_tensor_node(
        sig,
        inc,
        out,
        action,
        decode,
        encode,
        torch.empty((), dtype=torch.complex128),
        operation_name="bit",
    )
    assert node.name == "bit" and node.labels == tuple(
        labels[ind] for ind in original.inds
    )
    np.testing.assert_array_equal(node.tensor.numpy(), original.data)


def test_classical_port_refused_before_any_callback() -> None:
    def forbidden(**values: object) -> tuple[object, ...]:
        raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="quantum ports"):
        classical_action_tensor_node(
            Signature([Register("c", classical_bit_type())]),
            {},
            {},
            forbidden,
            decode,
            encode,
            torch.empty(()),
            operation_name="classical",
        )


def test_bad_bindings_refused_before_action() -> None:
    def forbidden(**values: object) -> tuple[object, ...]:
        raise AssertionError("must not be called")

    with pytest.raises(KeyError):
        classical_action_tensor_node(
            Signature.build(q=1),
            {},
            {},
            forbidden,
            decode,
            encode,
            torch.empty(()),
            operation_name="missing",
        )


def test_explicit_scalar_node_and_callback_error() -> None:
    def scalar(**values: object) -> tuple[object, ...]:
        return ()

    node = classical_action_tensor_node(
        Signature([]),
        {},
        {},
        scalar,
        decode,
        encode,
        torch.empty((), dtype=torch.complex128),
        operation_name="scalar",
    )
    assert node.labels == () and node.tensor.shape == () and node.tensor.item() == 1

    def fail(**values: object) -> tuple[object, ...]:
        raise KeyError("unavailable action")

    with pytest.raises(KeyError, match="unavailable action"):
        classical_action_tensor_node(
            Signature([]),
            {},
            {},
            fail,
            decode,
            encode,
            torch.empty(()),
            operation_name="failure",
        )
