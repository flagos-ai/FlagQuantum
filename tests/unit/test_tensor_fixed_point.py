"""Compare the native default ADCR control loop with unchanged Quimb source."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.simulation.tensor_network.full_simplification import (
    simplify_tensor_network,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def network(seed: int, dtype: torch.dtype) -> tuple[TensorNetworkNode, ...]:
    rng = np.random.default_rng(seed)
    graph = ((0, 1, 2), (1, 3), (2, 4), (3, 4, 5), (5, 6), (6, 7))
    return tuple(
        TensorNetworkNode(
            torch.tensor(
                rng.choice([0.0, 0.0, 1.0, -1.0], size=(2,) * len(labels)), dtype=dtype
            ),
            labels,
        )
        for labels in graph
    )


def compare_loop(
    nodes: Sequence[TensorNetworkNode],
    outputs: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
    *,
    atol: float = 0,
    source_zero_nan: bool = False,
) -> list[tuple[str, int, int]]:
    import quimb.tensor as qtn

    import flagquantum.simulation.tensor_network.full_simplification as native

    source = qtn.TensorNetwork(
        [qtn.Tensor(n.tensor.detach().numpy(), n.labels) for n in nodes]
    )
    expected_trace: list[tuple[str, int, int]] = []
    actual_trace: list[tuple[str, int, int]] = []
    source_caches: list[object] = []
    native_caches: dict[str, list[object]] = {}
    for code, original_name, native_name in (
        ("A", "antidiag_gauge_", "gauge_tensor_antidiagonals"),
        ("D", "diagonal_reduce_", "reduce_tensor_diagonals"),
        ("C", "column_reduce_", "reduce_tensor_columns"),
        ("R", "rank_simplify_", "simplify_tensor_ranks"),
    ):
        original = getattr(qtn.TensorNetwork, original_name)
        operation = getattr(native, native_name)

        def source_step(
            self: Any, _code: str = code, _original: Any = original, **kwargs: Any
        ) -> Any:
            source_caches.append(kwargs["cache"])
            result = _original(self, **kwargs)
            expected_trace.append((_code, self.num_tensors, self.num_indices))
            return result

        def native_step(
            active: Sequence[TensorNetworkNode],
            labels: tuple[int, ...],
            _code: str = code,
            _operation: Any = operation,
            **kwargs: Any,
        ) -> Any:
            native_caches.setdefault(_code, []).append(kwargs["_cache"])
            result = _operation(active, labels, **kwargs)
            actual_trace.append(
                (_code, len(result), len({label for n in result for label in n.labels}))
            )
            return result

        monkeypatch.setattr(qtn.TensorNetwork, original_name, source_step)
        monkeypatch.setattr(native, native_name, native_step)
    import flagquantum.simulation.tensor_network.column_reduction as col
    import flagquantum.simulation.tensor_network.diagonal_reduction as ad
    import flagquantum.simulation.tensor_network.rank_simplification as rank

    source_ops: list[tuple[object, ...]] = []
    native_ops: list[tuple[object, ...]] = []
    original_flip, original_reindex, original_cut = (
        qtn.TensorNetwork.flip_,
        qtn.TensorNetwork.reindex_,
        qtn.TensorNetwork.isel_,
    )
    original_pair = qtn.Tensor.contract
    native_flip, native_merge, native_cut, native_pair = (
        ad._flip_edge,
        ad._merge_edge,
        getattr(col, "_slice_nodes"),
        getattr(rank, "einsum_pair_by_labels"),
    )

    def flip(self: Any, labels: Sequence[int]) -> Any:
        source_ops.append(("A", tuple(labels)))
        return original_flip(self, labels)

    def merge(self: Any, mapping: dict[int, int]) -> Any:
        source_ops.append(("D", tuple(mapping.items())))
        return original_reindex(self, mapping)

    def cut(self: Any, mapping: dict[int, int]) -> Any:
        source_ops.append(("C", tuple(mapping.items())))
        return original_cut(self, mapping)

    def pair(self: Any, other: Any, **kwargs: Any) -> Any:
        source_ops.append(("R", self.inds, other.inds, tuple(kwargs["output_inds"])))
        return original_pair(self, other, **kwargs)

    def nflip(active: Any, label: int) -> Any:
        native_ops.append(("A", (label,)))
        return native_flip(active, label)

    def nmerge(active: Any, removed: int, retained: int) -> Any:
        native_ops.append(("D", ((removed, retained),)))
        return native_merge(active, removed, retained)

    def ncut(active: Any, mapping: Any) -> Any:
        native_ops.append(("C", tuple(mapping.items())))
        return native_cut(active, mapping)

    def npair(a: Any, ai: Any, b: Any, bi: Any, outputs: Any) -> Any:
        native_ops.append(("R", tuple(ai), tuple(bi), tuple(outputs)))
        return native_pair(a, ai, b, bi, outputs)

    monkeypatch.setattr(qtn.TensorNetwork, "flip_", flip)
    monkeypatch.setattr(qtn.TensorNetwork, "reindex_", merge)
    monkeypatch.setattr(qtn.TensorNetwork, "isel_", cut)
    monkeypatch.setattr(qtn.Tensor, "contract", pair)
    monkeypatch.setattr(ad, "_flip_edge", nflip)
    monkeypatch.setattr(ad, "_merge_edge", nmerge)
    monkeypatch.setattr(col, "_slice_nodes", ncut)
    monkeypatch.setattr(rank, "einsum_pair_by_labels", npair)
    source.full_simplify_(output_inds=outputs, atol=atol)
    actual = simplify_tensor_network(nodes, outputs, atol=atol)
    assert actual_trace == expected_trace
    assert native_ops == source_ops
    assert all(cache is source_caches[0] for cache in source_caches)
    assert all(
        all(cache is caches[0] for cache in caches) for caches in native_caches.values()
    )
    result = _contract_nodes_greedy(actual, outputs)[0]
    reference = source.contract(output_inds=outputs)
    if isinstance(reference, qtn.Tensor):
        reference = reference.transpose(*outputs).data
    if source_zero_nan:
        assert np.isnan(reference).all()
        assert bool(torch.isfinite(result).all()) and not bool(result.any())
    else:
        np.testing.assert_allclose(
            result.detach().numpy(), reference, rtol=2e-5, atol=2e-5
        )
    if atol == 0:
        torch.testing.assert_close(result, _contract_nodes_greedy(nodes, outputs)[0])
    return actual_trace


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("seed", range(12))
def test_full_source_round_order_counts_values_and_shared_cache(
    seed: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = network(seed, dtype)
    if dtype.is_complex:
        nodes = (replace(nodes[0], tensor=nodes[0].tensor * (0.3 + 0.7j)), *nodes[1:])
    prior = tuple(n.tensor.clone() for n in nodes)
    trace = compare_loop(
        nodes,
        (0, 7),
        monkeypatch,
        source_zero_nan=not dtype.is_complex and seed in (3, 5, 8),
    )
    assert [code for code, _, _ in trace] == list("ADCR") * (len(trace) // 4)
    for node, old in zip(nodes, prior, strict=True):
        torch.testing.assert_close(node.tensor, old)


def test_three_round_revisit_after_rank_exposes_sparse_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = (
        (4, 3, 1),
        (4, 1, 5, 2),
        (2, 3, 0, 5),
        (5, 3),
        (2, 4, 0, 1),
        (3, 4, 1, 0),
        (2, 1, 4),
    )
    # Fixed sparse ±1 supports require a second productive ADCR round.
    words = (
        "11010010",
        "1-0-00000-0-11--",
        "00--000011-00100",
        "110-",
        "-00-01011000-10-",
        "010-000-0-010001",
        "101101-0",
    )
    nodes = tuple(
        TensorNetworkNode(
            torch.tensor(
                [{"0": 0.0, "1": 1.0, "-": -1.0}[x] for x in word], dtype=torch.float64
            ).reshape((2,) * len(labels)),
            labels,
        )
        for labels, word in zip(graph, words, strict=True)
    )
    trace = compare_loop(nodes, (0,), monkeypatch)
    assert len(trace) == 12


@pytest.mark.parametrize("mixed", [False, True])
def test_same_register_consumer_with_captured_groups_and_phase_gradient(
    mixed: bool,
) -> None:
    from flagquantum.core.ir.program.registers import (
        Register,
        RegisterSide,
        RegisterType,
    )
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.tensor_network.register_superoperator_binding import (
        contract_register_superoperator,
    )
    from flagquantum.simulation.tensor_network.register_tensor_binding import (
        contract_register_tensors,
    )
    from flagquantum.simulation.tensor_network.register_tensor_indices import (
        register_contraction_groups,
    )

    phase = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    nodes = network(4, torch.complex128)
    nodes = (
        replace(nodes[0], tensor=nodes[0].tensor * torch.exp(1j * phase)),
        *nodes[1:],
    )
    outgoing = {"q": ((0, 7),)}
    signature = Signature(
        (Register("q", RegisterType("integer", 2), side=RegisterSide.RIGHT),)
    )
    if mixed:
        nodes = tuple(
            replace(n, labels=tuple(2 * x for x in n.labels)) for n in nodes
        ) + tuple(
            replace(
                n, tensor=n.tensor.conj(), labels=tuple(2 * x + 1 for x in n.labels)
            )
            for n in nodes
        )
    consumer = contract_register_superoperator if mixed else contract_register_tensors
    groups = register_contraction_groups(
        nodes, signature, {}, outgoing, superoperator=mixed
    )
    reduced = simplify_tensor_network(
        nodes, tuple(x for group in groups for x in group), atol=0
    )
    actual = consumer(reduced, signature, {}, outgoing, output_groups=groups)
    expected = consumer(nodes, signature, {}, outgoing)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.real.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected.real.sum(), phase)[0],
    )


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
def test_invalid_tolerance_refused_before_output(value: Any) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        simplify_tensor_network(network(0, torch.float64), atol=value)


def test_scalar_default_outputs_single_round_and_missing_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compare_loop((TensorNetworkNode(torch.tensor(2 + 3j), ()),), (), monkeypatch)
    with pytest.raises(ValueError, match="at least one"):
        simplify_tensor_network(())
    with pytest.raises(ValueError, match="distinct existing"):
        simplify_tensor_network(network(0, torch.float64), (999,))
    leaf = TensorNetworkNode(torch.tensor([1.0, 2.0]), (0,))
    torch.testing.assert_close(simplify_tensor_network((leaf,))[0].tensor, leaf.tensor)


def test_tolerance_error_and_zero_entry_jacobian_remain_explicit() -> None:
    parameter = torch.tensor([1.0, 0.0], requires_grad=True, dtype=torch.float64)
    nodes = (
        TensorNetworkNode(parameter, (0,)),
        TensorNetworkNode(torch.tensor([2.0, 3.0], dtype=torch.float64), (0,)),
    )
    reduced = simplify_tensor_network(nodes, (), atol=0)
    result = _contract_nodes_greedy(reduced, ())[0]
    assert torch.autograd.grad(result, parameter)[0].tolist() == [2.0, 0.0]
    tiny = (
        replace(nodes[0], tensor=torch.tensor([1.0, 0.5e-12], dtype=torch.float64)),
        replace(nodes[1], tensor=torch.tensor([2.0, 1e12], dtype=torch.float64)),
    )
    assert _contract_nodes_greedy(tiny, ())[0].item() == 2.5
    assert (
        _contract_nodes_greedy(simplify_tensor_network(tiny, ()), ())[0].item() == 2.0
    )


def test_reference_full_control_loop_hash_and_executable_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hashlib
    import inspect
    import runpy
    from pathlib import Path

    import quimb.tensor as qtn

    from examples.algorithms.tensor_fixed_point import main

    # Assert the same immutable source control loop used by the R audit.
    assert (
        hashlib.sha256(
            inspect.getsource(qtn.TensorNetwork.full_simplify).encode()
        ).hexdigest()
        == "d2c3e546b58a2af55a33a2035b0cd9632d1ef4416bbf053d46bfde33e67842df"
    )
    main()
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/algorithms/tensor_fixed_point.py"
        ),
        run_name="__main__",
    )
    assert "8192 complex outputs" in capsys.readouterr().out
