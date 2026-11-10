"""Forward/conjugate tensor copies preserve source axes, density maps and gradients."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.conjugate_pairs import (
    conjugate_pair_labels,
    tensor_conjugate_pairs,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
    register_superoperator_label_groups,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def groups(sig: Signature, inc: Any, out: Any) -> tuple[tuple[int, ...], ...]:
    return register_superoperator_label_groups(sig, inc, out)


def dense(nodes: Any, ordered: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    tensor, _ = _contract_nodes_greedy(
        nodes, tuple(label for group in ordered for label in group)
    )
    return tensor.reshape(tuple(1 << len(group) for group in ordered))


@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (2, 3, 4)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_original_forward_backward_data_axes_storage_and_metadata(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    import quimb.tensor as qtn
    from qualtran.simulation.tensor._quimb import (
        make_backward_tensor,
        make_forward_tensor,
    )

    size = int(np.prod(shape))
    real = torch.arange(1, size + 1, dtype=torch.float64).reshape(shape)
    data = (
        torch.complex(real, real.flip(tuple(range(len(shape)))) + 0.5)
        if dtype.is_complex
        else real
    ).to(dtype)
    labels = tuple(3 + 7 * i for i in range(len(shape)))
    metadata = {"source": "retained"}
    node = TensorNetworkNode(data, labels, "phase", metadata)
    source = qtn.Tensor(
        data.numpy(), inds=tuple(("edge", label) for label in labels), tags={"phase"}
    )
    original = (make_forward_tensor(source), make_backward_tensor(source))
    forward, backward = tensor_conjugate_pairs((node,))
    for native, reference, direction in zip(
        (forward, backward), original, (False, True), strict=True
    ):
        np.testing.assert_array_equal(native.tensor.numpy(force=True), reference.data)
        assert native.tensor.shape == data.shape
        assert native.labels == tuple(
            2 * ind[1] + int(not ind[2]) for ind in reference.inds
        )
        assert native.labels == conjugate_pair_labels(labels, backward=direction)
        assert native.name == node.name and native.metadata is metadata
        assert native.tensor.dtype == dtype and native.tensor.device == data.device
    assert forward.tensor.data_ptr() == data.data_ptr()
    assert forward is not node and backward is not node
    assert node.labels == labels and node.tensor is data
    assert "dag" in original[1].tags
    if shape == (2, 2) and dtype.is_complex:
        assert not torch.equal(backward.tensor, data.mH)


@pytest.mark.parametrize("bad", [(-1,), (True,), (1.5,), ("a",)])
def test_label_validation(bad: Any) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        conjugate_pair_labels(bad)


def test_repeated_sparse_labels_empty_and_invalid_rank() -> None:
    labels = (0, 100000, 0)
    assert conjugate_pair_labels(labels) == (0, 200000, 0)
    assert conjugate_pair_labels(labels, backward=True) == (1, 200001, 1)
    assert tensor_conjugate_pairs(()) == ()
    invalid_direction: Any = 1
    with pytest.raises(ValueError, match="boolean"):
        conjugate_pair_labels((0,), backward=invalid_direction)
    with pytest.raises(ValueError, match="rank"):
        tensor_conjugate_pairs((TensorNetworkNode(torch.eye(2), (0,)),))
    with pytest.raises(ValueError, match="nonnegative"):
        tensor_conjugate_pairs((TensorNetworkNode(torch.ones(2), (-1,)),))


@pytest.mark.parametrize("width", [0, 1, 2])
@pytest.mark.parametrize("shape", [(), (2,), (0,)])
def test_complete_original_superquimb_channel_and_joint_density(
    width: int, shape: tuple[int, ...]
) -> None:
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.simulation.tensor_network.component_tensor_nodes import (
        component_tensor_nodes,
    )
    from tests.unit.test_component_zero_initialization import phase_fixture
    from tests.unit.test_register_zero_state import supplier

    record, source = phase_fixture(width, shape)
    order = tuple(i for i, _, _ in iter_component_calls(record))
    nodes, inc, out = component_tensor_nodes(
        record, order, supplier, torch.empty((), dtype=torch.complex128)
    )
    sig = Signature(record.functions[0].registers or ())
    doubled = tensor_conjugate_pairs(nodes)
    actual = contract_register_superoperator(doubled, sig, inc, out)
    original = cbloq_to_superquimb(source)
    assert len(doubled) == len(original.tensors) == 2 * len(nodes)
    np.testing.assert_allclose(
        actual.numpy(),
        quimb_to_dense(original, source.signature, superoperator=True),
        atol=3e-14,
        rtol=0,
    )
    size = actual.shape[0]
    state = torch.complex(
        torch.arange(1, size + 1, dtype=torch.float64),
        torch.arange(size, 0, -1, dtype=torch.float64),
    )
    state /= state.norm()
    rho = state[:, None] * state.conj()[None, :]
    # Include a second independent ket to test a mixed input density matrix.
    other = torch.roll(state, 1) * torch.exp(
        1j * torch.arange(size, dtype=torch.float64)
    )
    rho = 0.6 * rho + 0.4 * other[:, None] * other.conj()[None, :]
    mapped = torch.einsum("abij,ij->ab", actual, rho)
    n = size.bit_length() - 1
    widths = [int(r.bitsize) for r in sig.lefts() for _ in r.all_idxs()]
    offset = 0
    bindings = []
    for bits in widths:
        bindings.append(tuple(range(offset, offset + bits)))
        offset += bits
    bound = bind_register_program(record, tuple(bindings), clean=(n,))
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": 3, "s": 2}
    result = torch.zeros_like(rho)
    for weight, ket in ((0.6, state), (0.4, other)):
        with_work = torch.zeros(size * 2, dtype=torch.complex128)
        with_work[::2] = ket
        evolved = (
            fq.Circuit(n + 1, inputs=with_work, dtype=torch.complex128)
            .append(FieldOperation(n + 1, bound.circuit.instructions))
            .state()
            .flatten()
        )
        assert torch.count_nonzero(evolved[1::2]) == 0
        result += weight * evolved[::2, None] * evolved[None, ::2].conj()
    torch.testing.assert_close(mapped, result, atol=3e-14, rtol=0)
    torch.testing.assert_close(
        torch.trace(mapped), torch.tensor(1, dtype=torch.complex128), atol=3e-14, rtol=0
    )


@pytest.mark.parametrize("width", [1, 2])
def test_initialized_density_original_source_and_supplier_replacement(
    width: int,
) -> None:
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction
    from qualtran.simulation.tensor._quimb import (
        cbloq_to_superquimb,
        initialize_from_zero,
    )

    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_components_for_tensor,
    )
    from flagquantum.compiler.registers.component_zero_initialization import (
        initialize_component_from_zero,
    )
    from flagquantum.simulation.tensor_network.component_tensor_nodes import (
        component_tensor_nodes,
    )
    from tests.unit.test_component_zero_initialization import phase_fixture
    from tests.unit.test_register_zero_state import supplier

    record, source = phase_fixture(width, ())
    initialized = initialize_component_from_zero(record)
    selected, _ = expand_components_for_tensor(
        initialized,
        lambda name: name.startswith("__fq_zero_") or name in {"h", "s"},
        entry="from_zero",
        full_flatten=False,
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "from_zero"))
    sig = Signature(selected.functions[0].registers or ())

    def consume(supply: Any) -> torch.Tensor:
        nodes, inc, out = component_tensor_nodes(
            selected,
            order,
            supply,
            torch.empty((), dtype=torch.complex128),
            entry="from_zero",
        )
        return contract_register_superoperator(
            tensor_conjugate_pairs(nodes), sig, inc, out
        )

    actual = consume(supplier)
    original = flatten_for_tensor_contraction(
        initialize_from_zero(source), full_flatten=True
    )
    expected = quimb_to_dense(
        cbloq_to_superquimb(original), original.signature, superoperator=True
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-14, rtol=0)

    def alternative(function: Any, inc: Any, out: Any) -> tuple[TensorNetworkNode, ...]:
        return tuple(
            replace(node, tensor=node.tensor * 1j)
            for node in supplier(function, inc, out)
        )

    # Global phase per leaf cancels between forward and conjugate networks.
    torch.testing.assert_close(consume(alternative), actual, atol=3e-14, rtol=0)


def test_complex_autograd_density_and_no_transposition() -> None:
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    state = torch.stack((torch.cos(theta).to(torch.complex128), 1j * torch.sin(theta)))
    nodes = tensor_conjugate_pairs((TensorNetworkNode(state, (13,)),))
    rho = dense(nodes, ((26,), (27,)))
    expected = state[:, None] * state.conj()[None, :]
    torch.testing.assert_close(rho, expected)
    (gradient,) = torch.autograd.grad(rho[0, 1].imag, (theta,))
    torch.testing.assert_close(gradient, -torch.cos(2 * theta))


def test_measured_branch_density_is_unnormalized_until_branch_sum() -> None:
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_component_tensor_expansion import (
        test_measured_add_consumer_preserves_gates_all_history_pairs_and_work,
    )

    record, _ = wrapped_add(2)
    bound = bind_register_program(record, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::4] = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    initial /= initial.norm()
    target = torch.zeros_like(initial)
    for prefix in range(64):
        a, b, spectator = prefix >> 4, (prefix >> 1) & 7, prefix & 1
        target[((a << 4) | (((b + 2 * a) & 7) << 1) | spectator) << 2] = initial[
            prefix << 2
        ]
    labels = tuple(range(8))
    result = torch.zeros((64, 64), dtype=torch.complex128)
    # Full 256x256 density retains work axes before the explicit clean slice.
    for history in product((0, 1), repeat=4):
        branch = forced_branch(initial, bound.circuit.instructions, history)
        nodes = tensor_conjugate_pairs(
            (TensorNetworkNode(branch.reshape((2,) * 8), labels),)
        )
        from flagquantum.core.ir.program.registers import (
            Register,
            RegisterSide,
            RegisterType,
        )

        signature = Signature(
            [Register("state", RegisterType("qubit", 8), side=RegisterSide.RIGHT)]
        )
        density = contract_register_superoperator(
            nodes, signature, {}, {"state": (labels,)}
        )
        torch.testing.assert_close(
            density, (target[:, None] * target.conj()[None, :]) / 16, atol=3e-14, rtol=0
        )
        assert (
            torch.count_nonzero(density[1::4]) == 0
            and torch.count_nonzero(density[:, 1::4]) == 0
        )
        torch.testing.assert_close(
            torch.trace(density),
            torch.tensor(1 / 16, dtype=torch.complex128),
            atol=3e-14,
            rtol=0,
        )
        result += density[::4, ::4]
    torch.testing.assert_close(
        result, target[::4, None] * target[None, ::4].conj(), atol=3e-14, rtol=0
    )
    # Full source primitive/feedback, all inverse histories and append paths.
    test_measured_add_consumer_preserves_gates_all_history_pairs_and_work(False)
