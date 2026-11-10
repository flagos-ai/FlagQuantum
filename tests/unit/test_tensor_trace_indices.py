"""Discard index reindexing matches source channels and existing partial traces."""

from collections import Counter
from itertools import combinations, product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.simulation.tensor_network.conjugate_pairs import (
    conjugate_pair_labels,
    tensor_conjugate_pairs,
)
from flagquantum.simulation.tensor_network.measurement_copy import (
    z_measurement_copy_node,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy
from flagquantum.simulation.tensor_network.trace_indices import trace_conjugate_indices

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def contract(nodes: Any, keep: tuple[int, ...]) -> torch.Tensor:
    ordered = conjugate_pair_labels(keep) + conjugate_pair_labels(keep, backward=True)
    result, _ = _contract_nodes_greedy(nodes, ordered)
    return result.reshape(1 << len(keep), 1 << len(keep))


SUBSETS = [subset for count in range(4) for subset in combinations(range(3), count)]


@pytest.mark.parametrize("discard", SUBSETS)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_all_three_bit_subsets_original_reindex_and_mixed_density(
    discard: tuple[int, ...], dtype: torch.dtype
) -> None:
    import quimb.tensor as qtn
    from qualtran.simulation.tensor._quimb import (
        make_backward_tensor,
        make_forward_tensor,
    )

    labels = (3, 17, 41)
    trace = tuple(labels[i] for i in discard)
    keep = tuple(label for i, label in enumerate(labels) if i not in discard)
    state = torch.complex(
        torch.arange(1, 9, dtype=torch.float64),
        torch.arange(8, 0, -1, dtype=torch.float64),
    ).to(dtype)
    state /= state.norm()
    other = torch.roll(state, 2) * torch.exp(
        1j * torch.arange(8, dtype=torch.float64)
    ).to(dtype)
    actual = torch.zeros((1 << len(keep),) * 2, dtype=dtype)
    expected = torch.zeros_like(actual)
    for weight, ket in ((0.3, state), (0.7, other)):
        metadata = {"purpose": "retained"}
        pure = TensorNetworkNode(ket.reshape(2, 2, 2), labels, "joint", metadata)
        nodes = tensor_conjugate_pairs((pure,))
        traced = trace_conjugate_indices(nodes, trace)
        assert all(
            a.tensor is b.tensor and a.metadata is metadata
            for a, b in zip(nodes, traced, strict=True)
        )
        assert nodes[1].labels == conjugate_pair_labels(labels, backward=True)
        reference = qtn.Tensor(
            ket.numpy().reshape(2, 2, 2),
            inds=tuple(("edge", label) for label in labels),
        )
        original = qtn.TensorNetwork(
            [make_forward_tensor(reference), make_backward_tensor(reference)]
        )
        for label in trace:
            original.reindex(
                {
                    ("edge", label, True): ("edge", label),
                    ("edge", label, False): ("edge", label),
                },
                inplace=True,
            )
        output = tuple(
            ("edge", label, flag) for flag in (True, False) for label in keep
        )
        source = original.contract(output_inds=output)
        source_data = source.data if hasattr(source, "inds") else source
        np.testing.assert_allclose(
            contract(traced, keep).numpy(),
            np.asarray(source_data).reshape(actual.shape),
            atol=2e-6 if dtype == torch.complex64 else 2e-14,
            rtol=0,
        )
        actual += weight * contract(traced, keep)
        axes = tuple(i for i in range(3) if i not in discard) + discard
        by_environment = ket.reshape(2, 2, 2).permute(axes).reshape(1 << len(keep), -1)
        expected += weight * (by_environment @ by_environment.mH)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.trace(actual), torch.tensor(1, dtype=dtype))


@pytest.mark.parametrize("discard", [0, 1, 2])
def test_existing_circuit_partial_trace_bell_ghz_and_coherences(discard: int) -> None:
    import flagquantum as fq
    from flagquantum.simulation.tensor_network.entrypoints import (
        build_tensor_network,
        tensor_network_discard_qubit,
    )

    circuit = fq.Circuit(3, dtype=torch.complex128).h(0).s(0).cx(0, 1).cx(1, 2)
    state = circuit.state().flatten()
    nodes = tensor_conjugate_pairs(
        (TensorNetworkNode(state.reshape(2, 2, 2), (0, 1, 2)),)
    )
    actual = contract(
        trace_conjugate_indices(nodes, (discard,)),
        tuple(i for i in range(3) if i != discard),
    )
    plan = build_tensor_network(circuit, dtype=torch.complex128)
    expected = tensor_network_discard_qubit(plan, discard)[0]
    torch.testing.assert_close(actual, expected)
    # The GHZ environment is entangled: reset/projection would give a pure state.
    assert torch.linalg.matrix_rank(actual) == 2
    torch.testing.assert_close(
        torch.trace(actual @ actual), torch.tensor(0.5, dtype=torch.complex128)
    )


@pytest.mark.parametrize("discard_result", [False, True])
def test_original_measure_z_channel_then_classical_discard(
    discard_result: bool,
) -> None:
    from qualtran import BloqBuilder, CBit, QBit, Register, Side, Signature
    from qualtran.bloqs.basic_gates import Discard, MeasureZ
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction
    from qualtran.simulation.tensor._quimb import (
        DiscardInd,
        cbloq_to_quimb,
        cbloq_to_superquimb,
    )

    native = z_measurement_copy_node(3, 17, 41, torch.empty((), dtype=torch.complex128))
    copy = native.tensor
    # Original local tensor and sentinel are read unchanged.
    source_items = MeasureZ().my_tensors({"q": "input"}, {"c": "output"})
    np.testing.assert_array_equal(copy.numpy(), source_items[0].data)
    assert isinstance(source_items[1], DiscardInd)
    nodes = trace_conjugate_indices(tensor_conjugate_pairs((native,)), (41,))
    bb, inputs = BloqBuilder.from_signature(
        Signature(
            [Register("q", QBit(), side=Side.LEFT)]
            + ([] if discard_result else [Register("c", CBit(), side=Side.RIGHT)])
        )
    )
    q = inputs["q"]
    c = bb.add(MeasureZ(), q=q)
    if discard_result:
        bb.add(Discard(), c=c)
        source = bb.finalize()
        nodes = trace_conjugate_indices(nodes, (17,))
        output = conjugate_pair_labels((3,)) + conjugate_pair_labels(
            (3,), backward=True
        )
    else:
        source = bb.finalize(c=c)
        output = (
            conjugate_pair_labels((17,))
            + conjugate_pair_labels((17,), backward=True)
            + conjugate_pair_labels((3,))
            + conjugate_pair_labels((3,), backward=True)
        )
    flat = flatten_for_tensor_contraction(source)
    with pytest.raises(ValueError, match="discard information"):
        cbloq_to_quimb(flat)
    original = cbloq_to_superquimb(flat)
    expected = quimb_to_dense(original, source.signature, superoperator=True)
    actual, _ = _contract_nodes_greedy(nodes, output)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-14, rtol=0)
    if discard_result:
        torch.testing.assert_close(actual, torch.eye(2, dtype=torch.complex128))
    else:
        assert actual[0, 1].count_nonzero() == 0
        assert actual[1, 0].count_nonzero() == 0
    assert get_source_measurements(source) == 1


def get_source_measurements(source: Any) -> int:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    return int(get_cost_value(source, QECGatesCost()).measurement)


def test_replacement_density_supplier_and_complex_gradient() -> None:
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    # A phase coherence survives tracing the independent environment.
    left = torch.stack((torch.cos(theta).to(torch.complex128), 1j * torch.sin(theta)))
    environment = torch.tensor([1, 2j], dtype=torch.complex128) / np.sqrt(5)
    pure = TensorNetworkNode(torch.outer(left, environment), (7, 31))
    split = (TensorNetworkNode(left, (7,)), TensorNetworkNode(environment, (31,)))

    def consume(nodes: Any) -> torch.Tensor:
        return contract(
            trace_conjugate_indices(tensor_conjugate_pairs(nodes), (31,)), (7,)
        )

    actual = consume((pure,))
    torch.testing.assert_close(actual, consume(split))
    (gradient,) = torch.autograd.grad(actual[0, 1].imag, (theta,))
    torch.testing.assert_close(gradient, -torch.cos(2 * theta))


def test_invalid_trace_requests_dimensions_and_noop() -> None:
    good = tensor_conjugate_pairs((TensorNetworkNode(torch.ones(2), (3,)),))
    assert trace_conjugate_indices(good, ()) == good
    assert trace_conjugate_indices((), ()) == ()
    invalid_requests: Any = ((-1,), (True,), (1.5,))
    for labels in invalid_requests:
        with pytest.raises(ValueError, match="nonnegative"):
            trace_conjugate_indices(good, labels)
    with pytest.raises(ValueError, match="distinct"):
        trace_conjugate_indices(good, (3, 3))
    with pytest.raises(ValueError, match="exposed"):
        trace_conjugate_indices(good, (10,))
    traced = trace_conjugate_indices(good, (3,))
    with pytest.raises(ValueError, match="exposed"):
        trace_conjugate_indices(traced, (3,))
    with pytest.raises(ValueError, match="dimensions"):
        trace_conjugate_indices(
            (
                TensorNetworkNode(torch.ones(2), (6,)),
                TensorNetworkNode(torch.ones(3), (7,)),
            ),
            (3,),
        )
    with pytest.raises(ValueError, match="rank"):
        trace_conjugate_indices((TensorNetworkNode(torch.ones(2), (6, 7)),), (3,))
    with pytest.raises(ValueError, match="nonnegative"):
        trace_conjugate_indices((TensorNetworkNode(torch.ones(2), (-1,)),), ())


def test_all_measured_add_branches_trace_clean_work_and_sum() -> None:
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_component_hierarchy_adjoint import wrapped_add
    from tests.unit.test_component_tensor_expansion import (
        test_measured_add_consumer_preserves_gates_all_history_pairs_and_work,
    )

    record, _ = wrapped_add(2)
    bound = bind_register_program(record, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    assert Counter(g.name for g in bound.circuit.instructions)["measure"] == 4
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::4] = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    initial /= initial.norm()
    target = torch.zeros(64, dtype=torch.complex128)
    for prefix in range(64):
        a, b, spectator = prefix >> 4, (prefix >> 1) & 7, prefix & 1
        target[(a << 4) | (((b + 2 * a) & 7) << 1) | spectator] = initial[prefix << 2]
    total = torch.zeros(64, 64, dtype=torch.complex128)
    for history in product((0, 1), repeat=4):
        branch = forced_branch(initial, bound.circuit.instructions, history)
        assert torch.count_nonzero(branch.reshape(64, 4)[:, 1:]) == 0
        nodes = tensor_conjugate_pairs(
            (TensorNetworkNode(branch.reshape((2,) * 8), tuple(range(8))),)
        )
        density = contract(trace_conjugate_indices(nodes, (6, 7)), tuple(range(6)))
        torch.testing.assert_close(
            density, torch.outer(target, target.conj()) / 16, atol=3e-14, rtol=0
        )
        total += density
    torch.testing.assert_close(
        total, torch.outer(target, target.conj()), atol=3e-14, rtol=0
    )
    torch.testing.assert_close(
        torch.trace(total), torch.tensor(1, dtype=torch.complex128), atol=3e-14, rtol=0
    )
    test_measured_add_consumer_preserves_gates_all_history_pairs_and_work(False)


def test_discard_quantum_environment_original_dephasing_channel() -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import CNOT, DiscardQ, ZeroState
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    bb = BloqBuilder()
    q = bb.add_register("q", 1)
    q, environment = bb.add(CNOT(), ctrl=q, target=bb.add(ZeroState()))
    bb.add(DiscardQ(), q=environment)
    source = bb.finalize(q=q)
    flat = flatten_for_tensor_contraction(source)
    copy = torch.zeros(2, 2, 2, dtype=torch.complex128)
    copy[0, 0, 0] = copy[1, 1, 1] = 1
    # Isometry from input q to output q and its fresh environment.
    nodes = tensor_conjugate_pairs((TensorNetworkNode(copy, (7, 3, 41)),))
    nodes = trace_conjugate_indices(nodes, (41,))
    output = (14, 15, 6, 7)
    actual, _ = _contract_nodes_greedy(nodes, output)
    expected = quimb_to_dense(
        cbloq_to_superquimb(flat), source.signature, superoperator=True
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-14, rtol=0)
    with pytest.raises(NotImplementedError, match="classical simulation"):
        source.call_classically(q=1)
