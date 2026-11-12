"""Reference queue/operand parity and exact native rank-reduction scenarios."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy
from flagquantum.simulation.tensor_network.rank_simplification import (
    simplify_tensor_ranks,
)
from flagquantum.simulation.tensor_network.stages import einsum_pair_by_labels

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

# Rank-preserving chain, protected hyperedge, repeated indices, free sums,
# all-scalar closure, blocked contraction, shared retained hyperedge and isolates.
GRAPHS = (
    ((0, 1), (1, 2), (2, 3)),
    ((0, 1), (1, 2), (1, 3)),
    ((0, 0, 1), (1, 2)),
    ((0, 1), (1, 2), (2,)),
    ((0,), (0,)),
    ((0, 1, 2), (2, 3, 4)),
    ((0, 1), (0, 1), (0, 2)),
    ((0, 1), (2, 3)),
    ((0, 1, 2, 3, 4), (0, 1, 5, 6, 7)),
)
Trace = list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]


def random_nodes(
    graph: Sequence[tuple[int, ...]], seed: int, dtype: torch.dtype = torch.complex128
) -> tuple[TensorNetworkNode, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        TensorNetworkNode(
            torch.randn((2,) * len(labels), dtype=dtype, generator=generator),
            labels,
            name=f"leaf-{i}",
        )
        for i, labels in enumerate(graph)
    )


def dense(nodes: Sequence[TensorNetworkNode], outputs: tuple[int, ...]) -> torch.Tensor:
    return _contract_nodes_greedy(nodes, outputs)[0]


def compare_source(
    nodes: tuple[TensorNetworkNode, ...],
    outputs: tuple[int, ...],
    budget: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TensorNetworkNode, ...]:
    import quimb.tensor as qtn

    import flagquantum.simulation.tensor_network.rank_simplification as native

    source = qtn.TensorNetwork(
        [qtn.Tensor(node.tensor.detach().numpy(), node.labels) for node in nodes]
    )
    original_contract = qtn.Tensor.contract
    original_pair = einsum_pair_by_labels
    source_trace: Trace = []
    native_trace: Trace = []

    # Instrument the call boundary; the reference numerical method runs unchanged.
    def source_contract(self: Any, other: Any, **kwargs: Any) -> Any:
        source_trace.append(
            (tuple(self.inds), tuple(other.inds), tuple(kwargs["output_inds"]))
        )
        return original_contract(self, other, **kwargs)

    def native_pair(
        a: torch.Tensor,
        a_labels: Sequence[int],
        b: torch.Tensor,
        b_labels: Sequence[int],
        out: Sequence[int],
    ) -> torch.Tensor:
        native_trace.append((tuple(a_labels), tuple(b_labels), tuple(out)))
        return original_pair(a, a_labels, b, b_labels, out)

    with monkeypatch.context() as patch:
        patch.setattr(qtn.Tensor, "contract", source_contract)
        patch.setattr(native, "einsum_pair_by_labels", native_pair)
        source.rank_simplify_(output_inds=outputs, max_combinations=budget)
        result = simplify_tensor_ranks(nodes, outputs, max_combinations=budget)
    assert source_trace == native_trace
    assert tuple(t.inds for t in source.tensors) == tuple(n.labels for n in result)
    expected = source.contract(output_inds=outputs)
    if hasattr(expected, "data") and hasattr(expected, "inds"):
        expected = expected.data
    np.testing.assert_allclose(
        dense(result, outputs).numpy(), expected, rtol=1e-11, atol=2e-12
    )
    torch.testing.assert_close(
        dense(result, outputs), dense(nodes, outputs), rtol=1e-11, atol=2e-12
    )
    for left, right, retained in native_trace:
        assert len(retained) <= max(len(set(left)), len(set(right)))
    return result


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("budget", [1, 500])
def test_original_pair_order_hyperedges_and_complete_complex_output(
    graph: tuple[tuple[int, ...], ...], budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections import Counter

    nodes = random_nodes(graph, 17)
    counts = Counter(label for labels in graph for label in labels)
    outputs = tuple(label for label, count in counts.items() if count == 1)
    compare_source(nodes, outputs, budget, monkeypatch)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("budget", [1, 2, 500])
def test_dense_hypergraph_budget_queue_and_occurrence_ties(
    seed: int, budget: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A low budget changes the source's search order only above its combination
    # cutoff. Different ranks and retained hyperedges exercise that branch.
    rng = np.random.default_rng(seed)
    graph = tuple(
        (
            0,
            *tuple(
                int(x)
                for x in rng.choice(
                    np.arange(1, 7), size=int(rng.integers(1, 4)), replace=False
                )
            ),
        )
        for _ in range(7)
    )
    compare_source(random_nodes(graph, seed), (0,), budget, monkeypatch)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_diagonals_free_sums_zero_scalar_gradients_and_immutable_inputs(
    dtype: torch.dtype,
) -> None:
    original = random_nodes(((0, 0, 1), (1, 2), (4,)), 31, dtype)
    originals = tuple(node.tensor.clone() for node in original)
    phase = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    scalar = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    factor = torch.cos(phase) if not dtype.is_complex else torch.exp(1j * phase)
    nodes = (
        replace(original[0], tensor=original[0].tensor * factor),
        *original[1:],
        TensorNetworkNode(scalar, ()),
    )
    result = simplify_tensor_ranks(nodes, (0, 2))
    actual, expected = dense(result, (0, 2)), dense(nodes, (0, 2))
    torch.testing.assert_close(actual, expected)
    actual_loss, expected_loss = actual.real.sum(), expected.real.sum()
    for left, right in zip(
        torch.autograd.grad(actual_loss, (phase, scalar), retain_graph=True),
        torch.autograd.grad(expected_loss, (phase, scalar)),
        strict=True,
    ):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right)
    for node, prior in zip(original, originals, strict=True):
        assert torch.equal(node.tensor, prior)
    assert tuple(node.labels for node in original) == ((0, 0, 1), (1, 2), (4,))


def test_meta_shape_only_outputs_scalar_identity_and_explicit_protection() -> None:
    nodes = random_nodes(((0, 1), (1, 2), (2, 3)), 5)
    outputs = (0, 1, 2, 3)
    cpu = simplify_tensor_ranks(nodes, outputs)
    meta = simplify_tensor_ranks(
        tuple(replace(n, tensor=n.tensor.to("meta")) for n in nodes), outputs
    )
    assert [(n.labels, n.tensor.shape) for n in cpu] == [
        (n.labels, n.tensor.shape) for n in meta
    ]
    assert all(n.tensor.device.type == "meta" for n in meta)
    assert set(outputs).issubset(label for n in cpu for label in n.labels)
    inferred = simplify_tensor_ranks(nodes)
    assert len(inferred) == 1 and inferred[0].labels == (0, 3)
    scalar = torch.tensor(2 + 3j, dtype=torch.complex128)
    closed = simplify_tensor_ranks((TensorNetworkNode(scalar, ()),))
    assert closed[0].tensor is scalar and closed[0].labels == ()
    leaf = nodes[0]
    assert simplify_tensor_ranks((leaf,)) == (leaf,)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_search_budget(budget: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        simplify_tensor_ranks(random_nodes(((0,),), 1), max_combinations=budget)


def test_invalid_empty_outputs_dimensions_and_dtype() -> None:
    with pytest.raises(ValueError, match="at least one"):
        simplify_tensor_ranks(())
    nodes = random_nodes(((0, 1), (1, 2)), 1)
    for outputs in ((0, 0), (99,)):
        with pytest.raises(ValueError, match="distinct existing"):
            simplify_tensor_ranks(nodes, outputs)
    with pytest.raises(ValueError, match="Inconsistent dimension"):
        simplify_tensor_ranks((nodes[0], replace(nodes[1], tensor=torch.ones(3, 2))))
    with pytest.raises(ValueError, match="one dtype and device"):
        simplify_tensor_ranks(
            (nodes[0], replace(nodes[1], tensor=nodes[1].tensor.real))
        )
    with pytest.raises(ValueError, match="one dtype and device"):
        simplify_tensor_ranks(
            (nodes[0], replace(nodes[1], tensor=nodes[1].tensor.to("meta")))
        )


@pytest.mark.parametrize("width", [12, 13])
def test_qualtran_dense_width_trigger_and_wide_register_consumer(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qualtran
    import quimb.tensor as qtn
    from qualtran.simulation.tensor._dense import quimb_to_dense

    from examples.algorithms.tensor_rank_simplification import wide_state_network
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.tensor_network.register_tensor_binding import (
        contract_register_tensors,
    )

    nodes, outputs = wide_state_network(width)
    names = {label: ("q", (), i, "r") for i, label in enumerate(outputs)}
    source = qtn.TensorNetwork(
        [
            qtn.Tensor(
                n.tensor.numpy(), tuple(names.get(label, label) for label in n.labels)
            )
            for n in nodes
        ]
    )
    assert source.contraction_width() == width
    original = source.full_simplify
    triggers: list[dict[str, object]] = []

    def full(**kwargs: Any) -> Any:
        triggers.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(source, "full_simplify", full)
    reference_signature = qualtran.Signature(
        (qualtran.Register("q", qualtran.QUInt(width), side=qualtran.Side.RIGHT),)
    )
    expected = quimb_to_dense(source, reference_signature)
    assert triggers == ([{"inplace": True}] if width == 13 else [])
    reduced = simplify_tensor_ranks(nodes, outputs)
    assert len(reduced) == 1 and reduced[0].tensor.numel() == 2**width
    signature = Signature(
        (Register("q", RegisterType("integer", width), side=RegisterSide.RIGHT),)
    )
    actual = contract_register_tensors(reduced, signature, {}, {"q": (outputs,)})
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-12, atol=2e-12)
    compare_source(nodes, outputs, 500, monkeypatch)


@pytest.mark.parametrize("prepared", [False, True])
def test_real_bell_graph_provider_replacement_preserves_phase_and_gradient(
    prepared: bool,
) -> None:
    from examples.algorithms.bell_tensor_workflow import bell_program, bell_providers
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_component_root_for_tensor,
    )
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.tensor_network.component_tensor_nodes import (
        component_tensor_nodes_with_labels,
    )
    from flagquantum.simulation.tensor_network.register_axis_groups import (
        register_tensor_label_groups,
    )
    from flagquantum.simulation.tensor_network.register_tensor_binding import (
        contract_register_tensors,
    )

    like = torch.empty((), dtype=torch.complex128)
    record = bell_program(prepared=prepared)
    providers = bell_providers(record, like)
    selected, _ = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, "bell"
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    nodes, inc, out = component_tensor_nodes_with_labels(
        selected, order, providers.supply, like, "tensor_root"
    )
    signature = Signature(selected.functions[0].registers or ())
    outputs = tuple(
        label
        for group in register_tensor_label_groups(signature, inc, out)
        for label in group
    )
    phase = torch.tensor(0.29, requires_grad=True, dtype=torch.float64)
    phased = (
        replace(nodes[0], tensor=nodes[0].tensor * torch.exp(1j * phase)),
        *nodes[1:],
    )
    # Only preprocessing is replaced. Binding, contraction, IR and providers stay shared.
    reduced = simplify_tensor_ranks(phased, outputs)
    actual = contract_register_tensors(reduced, signature, inc, out)
    expected = contract_register_tensors(phased, signature, inc, out)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual.imag.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected.imag.sum(), phase)[0],
        rtol=0,
        atol=2e-14,
    )


def test_scalar_products_and_closed_repeated_trace() -> None:
    nodes = (
        TensorNetworkNode(torch.tensor(2j, dtype=torch.complex128), ()),
        TensorNetworkNode(torch.tensor(3j, dtype=torch.complex128), ()),
    )
    assert simplify_tensor_ranks(nodes)[0].tensor.item() == -6
    repeated = TensorNetworkNode(
        torch.tensor([[1j, 99], [88, 2j]], dtype=torch.complex128), (0, 0)
    )
    result = simplify_tensor_ranks((repeated,))
    assert result[0].labels == () and result[0].tensor.item() == 3j
    chain = random_nodes(((0, 1), (1, 2)), 1)
    result = simplify_tensor_ranks((*nodes, *chain))
    torch.testing.assert_close(dense(result, (0, 2)), -6 * dense(chain, (0, 2)))


def test_example_and_bounds(capsys: pytest.CaptureFixture[str]) -> None:
    import runpy
    from pathlib import Path

    from examples.algorithms.tensor_rank_simplification import main, wide_state_network

    main()
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/algorithms/tensor_rank_simplification.py"
        ),
        run_name="__main__",
    )
    assert "3 -> 1 tensors; 8192 complex outputs" in capsys.readouterr().out
    for width in (1, 15, True):
        with pytest.raises(ValueError, match="from 2 to 14"):
            wide_state_network(width)


def test_reference_versions_and_unchanged_dependency_sources() -> None:
    import hashlib
    import importlib.metadata
    import inspect
    from pathlib import Path

    import qualtran.simulation.tensor._dense as original_dense
    import quimb.tensor as qtn

    assert importlib.metadata.version("qualtran") == "0.7.0"
    assert importlib.metadata.version("quimb") == "1.15.0"
    raw = Path(original_dense.__file__).read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "13aa81f50654ccaaa4293e8a63991686723022f5"
    )
    assert (
        hashlib.sha256(
            inspect.getsource(qtn.TensorNetwork.rank_simplify).encode()
        ).hexdigest()
        == "8053227b2b88d818ad4fe41b978ecb693e0270ce5a73ef9c3f88e8daee79d0a2"
    )
    assert (
        hashlib.sha256(
            inspect.getsource(qtn.TensorNetwork.full_simplify).encode()
        ).hexdigest()
        == "d2c3e546b58a2af55a33a2035b0cd9632d1ef4416bbf053d46bfde33e67842df"
    )
