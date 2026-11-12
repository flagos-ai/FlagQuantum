"""Integer vector decomposition, complex overlaps and existing zero consumers."""

from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_basis_vectors import (
    register_basis_vector_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
LIKE = torch.empty((), dtype=torch.complex128)


def vector(width: int, state: bool, shape: tuple[int, ...] = ()) -> Signature:
    return Signature(
        (
            Register(
                "x",
                RegisterType("uint", width),
                shape,
                RegisterSide.RIGHT if state else RegisterSide.LEFT,
            ),
        )
    )


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("state", [False, True])
def test_every_small_integer_matches_original_decomposition(
    width: int, state: bool
) -> None:
    from qualtran.bloqs.basic_gates import IntEffect, IntState
    from qualtran.bloqs.basic_gates.z_basis import _ZVector

    signature = vector(width, state)
    connections = {"x": (tuple(range(width)),)}
    for value in range(1 << width):
        source = (IntState if state else IntEffect)(value, width)
        nodes = register_basis_vector_nodes(
            signature, connections, {"x": value}, LIKE, state=state
        )
        leaves = [
            b.bloq
            for b, _, _ in source.decompose_bloq().iter_bloqsoqs()
            if isinstance(b.bloq, _ZVector)
        ]
        assert len(leaves) == len(nodes) == width
        for leaf, node in zip(leaves, nodes, strict=True):
            tensor = leaf.my_tensors({"q": "in"}, {"q": "out"})[0]
            np.testing.assert_array_equal(node.tensor.numpy(), tensor.data)
        actual = contract_register_tensors(
            nodes, signature, {} if state else connections, connections if state else {}
        )
        np.testing.assert_array_equal(actual.numpy(), source.tensor_contract())


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("width", [0, 1, 2])
def test_array_order_empty_and_storage_dtype(
    shape: tuple[int, ...], state: bool, width: int
) -> None:
    signature = vector(width, state, shape)
    count = int(np.prod(shape))
    data = np.arange(count).reshape(shape) % (1 << width)
    labels = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(count))
    nodes = register_basis_vector_nodes(
        signature,
        {"x": labels},
        {"x": data if shape else int(data)},
        torch.empty((), dtype=torch.float64),
        state=state,
    )
    assert tuple(n.labels[0] for n in nodes) == tuple(range(count * width))
    assert all(
        n.tensor.dtype == torch.float64 and n.tensor.device.type == "cpu" for n in nodes
    )
    if not count or not width:
        assert nodes == ()
    else:
        from qualtran.bloqs.basic_gates import IntEffect, IntState

        expected = np.array([1.0])
        for value in data.flat:
            expected = np.kron(
                expected,
                (IntState if state else IntEffect)(int(value), width).tensor_contract(),
            )
        actual = contract_register_tensors(
            nodes,
            signature,
            {} if state else {"x": labels},
            {"x": labels} if state else {},
        )
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("initial", [0, 1])
@pytest.mark.parametrize("final", [0, 1])
def test_original_flanked_complex_overlap_and_gradient(
    initial: int, final: int
) -> None:
    from qualtran import BloqBuilder, QBit
    from qualtran.bloqs.basic_gates import Hadamard, SGate
    from qualtran.simulation.xcheck_classical_quimb import flank_with_classical_vectors

    bb = BloqBuilder()
    q = bb.add_register("q", QBit())
    q = bb.add(Hadamard(), q=q)
    q = bb.add(SGate(), q=q)
    source = bb.finalize(q=q)
    expected = flank_with_classical_vectors(
        source, {"q": initial}, {"q": final}
    ).tensor_contract()
    ket = register_basis_vector_nodes(
        vector(1, True), {"x": ((0,),)}, {"x": initial}, LIKE
    )
    bra = register_basis_vector_nodes(
        vector(1, False), {"x": ((1,),)}, {"x": final}, LIKE, state=False
    )
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    matrix = torch.tensor([[1, 1], [1j, -1j]], dtype=torch.complex128) / np.sqrt(2)
    operator = TensorNetworkNode(matrix * torch.exp(1j * theta), (1, 0))
    result = contract_register_tensors((*ket, operator, *bra), Signature(()), {}, {})
    torch.testing.assert_close(
        result, torch.as_tensor(expected) * torch.exp(1j * theta)
    )
    (gradient,) = torch.autograd.grad(result.real, (theta,))
    torch.testing.assert_close(gradient, (1j * result).real)


@pytest.mark.parametrize(
    "width,value", [(1, -1), (1, 2), (1, 0.5), (2, -1), (2, 4), (2, 0.5)]
)
@pytest.mark.parametrize("state", [False, True])
def test_source_validation_failures(width: int, value: Any, state: bool) -> None:
    from qualtran.bloqs.basic_gates import IntEffect, IntState

    try:
        (IntState if state else IntEffect)(value, width).decompose_bloq()
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            register_basis_vector_nodes(
                vector(width, state),
                {"x": (tuple(range(width)),)},
                {"x": value},
                LIKE,
                state=state,
            )
    else:
        pytest.fail("invalid source case unexpectedly succeeded")


def test_native_binding_refusals_and_missing_value_order() -> None:
    sig = vector(2, True)
    with pytest.raises(ValueError, match="distinct"):
        register_basis_vector_nodes(sig, {"x": ((0, 0),)}, {}, LIKE)
    with pytest.raises(ValueError, match="counts"):
        register_basis_vector_nodes(sig, {"x": ((0,),)}, {}, LIKE)
    with pytest.raises(KeyError):
        register_basis_vector_nodes(sig, {}, {}, LIKE)
    with pytest.raises(KeyError):
        register_basis_vector_nodes(sig, {"x": ((0, 1),)}, {}, LIKE)
    invalid_direction: Any = 1
    with pytest.raises(ValueError, match="boolean"):
        register_basis_vector_nodes(sig, {}, {}, LIKE, state=invalid_direction)
    for register in [
        Register("x", RegisterType("uint", 2), side=RegisterSide.THRU),
        Register("x", RegisterType("uint", 0, 2), side=RegisterSide.RIGHT),
    ]:
        with pytest.raises(ValueError, match="one-sided"):
            register_basis_vector_nodes(Signature((register,)), {}, {}, LIKE)
    with pytest.raises(IndexError):
        register_basis_vector_nodes(
            vector(1, True, (2,)), {"x": ((0,), (1,))}, {"x": np.array([0])}, LIKE
        )


def test_zero_supplier_uses_shared_vector_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation.tensor_network.register_zero_state as zero

    original = register_basis_vector_nodes
    calls: list[Any] = []

    def replacement(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(zero, "register_basis_vector_nodes", replacement)
    from tests.unit.test_register_zero_state import (
        test_replacement_supplier_keeps_initialized_consumer_and_gradient,
    )

    test_replacement_supplier_keeps_initialized_consumer_and_gradient()
    assert calls


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_tensor_vectors_consume_existing_append_and_separate_resource_model(
    width: int,
) -> None:
    from qualtran.bloqs.basic_gates import IntState, OneState, ZeroState
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler.registers.layout.int_vector import (
        int_state_ir,
        int_vector_cost_key,
    )

    for value in range(1 << width):
        gates = int_state_ir(tuple(range(width)), value)
        source = IntState(value, width)
        from flagquantum.core.ir import Instruction

        source_leaves = [
            b.bloq
            for b, _, _ in source.decompose_bloq().iter_bloqsoqs()
            if isinstance(b.bloq, (OneState, ZeroState))
        ]
        assert gates == tuple(
            Instruction("x", (i,))
            for i, leaf in enumerate(source_leaves)
            if isinstance(leaf, OneState)
        )
        assert len(gates) == value.bit_count()
        assert all(g.name == "x" for g in gates)
        assert query_qec_gate_cost(int_vector_cost_key(width)).clifford == 1
        assert get_cost_value(source, QECGatesCost()).clifford == 1
        nodes = register_basis_vector_nodes(
            vector(width, True), {"x": (tuple(range(width)),)}, {"x": value}, LIKE
        )
        expected = contract_register_tensors(
            nodes, vector(width, True), {}, {"x": (tuple(range(width)),)}
        )
        initial = torch.zeros(1 << (width + 1), dtype=torch.complex128)
        initial[0] = torch.tensor(1j, dtype=torch.complex128)
        circuit = fq.Circuit(width + 1, inputs=initial, dtype=torch.complex128)
        circuit.append(FieldOperation(width + 1, gates))
        actual = circuit.state().flatten().reshape(-1, 2)
        torch.testing.assert_close(actual[:, 0], 1j * expected, atol=2e-12, rtol=0)
        assert torch.count_nonzero(actual[:, 1]) == 0
        circuit.append(FieldOperation(width + 1, gates))
        torch.testing.assert_close(
            circuit.state().flatten(), initial, atol=2e-12, rtol=0
        )
