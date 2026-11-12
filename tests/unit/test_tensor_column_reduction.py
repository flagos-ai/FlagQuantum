"""Source slice-order parity, explicit tolerance and sparse-gradient boundaries."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.simulation.tensor_network.column_reduction import reduce_tensor_columns
from flagquantum.simulation.tensor_network.contraction import _slice_nodes
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy
from flagquantum.simulation.tensor_network.rank_simplification import (
    simplify_tensor_ranks,
)
from flagquantum.simulation.tensor_network.simplification_axes import (
    squeeze_tensor_axes,
)
from flagquantum.simulation.tensor_network.tensor_columns import find_tensor_column

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


def outputs_for(nodes: Sequence[TensorNetworkNode]) -> tuple[int, ...]:
    counts = Counter(label for node in nodes for label in node.labels)
    return tuple(label for label, count in counts.items() if count == 1)


def dense(nodes: Sequence[TensorNetworkNode], outputs: tuple[int, ...]) -> torch.Tensor:
    return _contract_nodes_greedy(nodes, outputs)[0]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", range(8))
def test_first_column_matches_unchanged_numpy_reference(
    dtype: torch.dtype, case: int
) -> None:
    from quimb.tensor.array_ops import find_columns

    tensors = [
        torch.zeros(2, 3, 4),
        torch.ones(2, 3),
        torch.tensor([[0, 1], [0, 2]]),
        torch.tensor([[0, 0], [1, 2]]),
        torch.tensor([0, 0, 3]),
        torch.ones(()),
        torch.empty(0, 2),
        torch.empty(0),
    ]
    data = tensors[case].to(dtype)
    assert find_tensor_column(data) == find_columns(data.numpy())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("side", [-1, 0, 1])
def test_threshold_is_strictly_greater_than_tolerance(
    dtype: torch.dtype, side: int
) -> None:
    from quimb.tensor.array_ops import find_columns

    real_dtype = (
        torch.float32 if dtype in {torch.float32, torch.complex64} else torch.float64
    )
    value = torch.tensor(1e-12, dtype=real_dtype)
    if side:
        value = torch.nextafter(
            value, torch.tensor(float("inf") if side > 0 else 0.0, dtype=real_dtype)
        )
    data = torch.tensor([1.0, value.item()], dtype=dtype)
    assert find_tensor_column(data, atol=1e-12) == find_columns(
        data.numpy(), atol=1e-12
    )
    assert find_tensor_column(data, atol=0) is None


@pytest.mark.parametrize("dtype", DTYPES)
def test_squeeze_axes_protects_outputs_repeated_labels_scalar_and_meta(
    dtype: torch.dtype,
) -> None:
    import quimb.tensor as qtn

    nodes = (
        TensorNetworkNode(
            torch.arange(6).reshape(1, 2, 1, 3).to(dtype),
            (0, 1, 0, 2),
            "tensor",
            {"owner": "source"},
        ),
        TensorNetworkNode(torch.ones(1, dtype=dtype), (0,)),
        TensorNetworkNode(torch.ones(1, dtype=dtype), (3,)),
        TensorNetworkNode(torch.tensor(2, dtype=dtype), ()),
    )
    for outputs in (outputs_for(nodes), (0, 1, 2, 3), ()):
        source = qtn.TensorNetwork(
            [qtn.Tensor(n.tensor.numpy(), n.labels) for n in nodes]
        )
        source.squeeze_(exclude=outputs)
        actual = squeeze_tensor_axes(nodes, outputs)
        assert tuple(n.labels for n in actual) == tuple(t.inds for t in source.tensors)
        for node, tensor, original in zip(actual, source.tensors, nodes, strict=True):
            np.testing.assert_array_equal(node.tensor.numpy(), tensor.data)
            assert node.metadata is original.metadata and node.name == original.name
        meta = squeeze_tensor_axes(
            tuple(replace(n, tensor=n.tensor.to("meta")) for n in nodes), outputs
        )
        assert [(n.labels, n.tensor.shape) for n in meta] == [
            (n.labels, n.tensor.shape) for n in actual
        ]
    default = squeeze_tensor_axes(nodes)
    assert default[2] is nodes[2] and default[3] is nodes[3]
    assert default[0].tensor.shape == (2, 3)
    assert nodes[0].tensor.shape == (1, 2, 1, 3)


def compare_columns(
    nodes: tuple[TensorNetworkNode, ...],
    outputs: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
    *,
    atol: float = 0,
) -> tuple[TensorNetworkNode, ...]:
    import quimb.tensor as qtn

    import flagquantum.simulation.tensor_network.column_reduction as native

    source = qtn.TensorNetwork([qtn.Tensor(n.tensor.numpy(), n.labels) for n in nodes])
    original_isel = qtn.TensorNetwork.isel_
    source_trace: list[dict[int, int]] = []
    native_trace: list[dict[int, int]] = []

    def source_slice(self: Any, selectors: dict[int, int]) -> Any:
        source_trace.append(dict(selectors))
        return original_isel(self, selectors)

    def native_slice(
        active: Sequence[TensorNetworkNode], selectors: dict[int, int]
    ) -> tuple[TensorNetworkNode, ...]:
        native_trace.append(dict(selectors))
        return _slice_nodes(active, selectors)

    with monkeypatch.context() as patch:
        patch.setattr(qtn.TensorNetwork, "isel_", source_slice)
        patch.setattr(native, "_slice_nodes", native_slice)
        source.column_reduce_(output_inds=outputs, atol=atol)
        actual = reduce_tensor_columns(nodes, outputs, atol=atol)
    assert native_trace == source_trace
    for node, tensor, original in zip(actual, source.tensors, nodes, strict=True):
        assert node.labels == tensor.inds
        np.testing.assert_array_equal(node.tensor.numpy(), tensor.data)
        assert node.metadata is original.metadata and node.name == original.name
    if atol == 0:
        torch.testing.assert_close(dense(actual, outputs), dense(nodes, outputs))
    return actual


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("dtype", DTYPES)
def test_random_sparse_hyperedges_trace_and_each_leaf(
    seed: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(seed)
    labels = ((0, 1, 2), (2, 3), (1, 2, 4), (3, 5), (5, 5, 6))
    tensors = []
    for i, inds in enumerate(labels):
        data = rng.normal(size=(2,) * len(inds))
        data *= rng.random(data.shape) < 0.35
        tensor = torch.tensor(data, dtype=dtype)
        if dtype.is_complex:
            tensor = tensor * (1 + 0.25j)
        tensors.append(TensorNetworkNode(tensor, inds, f"leaf-{i}"))
    nodes = tuple(tensors)
    compare_columns(nodes, outputs_for(nodes), monkeypatch)


def test_protected_first_candidate_stops_and_repeated_slice_is_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = (
        TensorNetworkNode(torch.ones(2, dtype=torch.complex128), (1,)),
        TensorNetworkNode(torch.zeros(2, 2, dtype=torch.complex128), (0, 1)),
    )
    actual = compare_columns(nodes, (0,), monkeypatch)
    assert tuple(n.labels for n in actual) == ((1,), (0, 1))
    repeated = (
        TensorNetworkNode(
            torch.tensor([[0, 0], [0, 3j]], dtype=torch.complex128), (1, 1)
        ),
        TensorNetworkNode(torch.tensor([0, 2j], dtype=torch.complex128), (1,)),
    )
    actual = compare_columns(repeated, (), monkeypatch)
    assert all(n.labels == () for n in actual)
    assert dense(actual, ()).item() == -6


def test_tolerance_can_drop_small_terms_without_global_error_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = (
        TensorNetworkNode(torch.tensor([1.0, 0.5e-12], dtype=torch.complex128), (0,)),
        TensorNetworkNode(torch.tensor([2.0, 1e12], dtype=torch.complex128), (0,)),
    )
    reduced = compare_columns(nodes, (), monkeypatch, atol=1e-12)
    assert dense(reduced, ()).item() == 2
    assert dense(nodes, ()).item() == 2.5
    exact = reduce_tensor_columns(nodes, (), atol=0)
    torch.testing.assert_close(dense(exact, ()), dense(nodes, ()))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fixed_sparse_support_phase_gradient_and_squeeze(dtype: torch.dtype) -> None:
    phase = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    factor = torch.exp(1j * phase) if dtype.is_complex else torch.cos(phase)
    ket = torch.tensor([0, 1], dtype=dtype) * factor
    environment = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
    nodes = (
        TensorNetworkNode(ket.reshape(1, 2), (7, 0)),
        TensorNetworkNode(environment, (0, 1)),
    )
    squeezed = squeeze_tensor_axes(nodes, (1,))
    reduced = simplify_tensor_ranks(reduce_tensor_columns(squeezed, (1,), atol=0), (1,))
    actual, expected = dense(reduced, (1,)), dense(nodes, (1,))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.real.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected.real.sum(), phase)[0],
    )


def test_ambient_zero_entry_gradient_differs_and_matches_original_selection() -> None:
    import quimb.tensor as qtn

    parameter = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    environment = torch.tensor([2.0, 3.0], dtype=torch.float64)
    nodes = (TensorNetworkNode(parameter, (0,)), TensorNetworkNode(environment, (0,)))
    reduced = dense(reduce_tensor_columns(nodes, (), atol=0), ())
    original = dense(nodes, ())
    source = qtn.TensorNetwork([qtn.Tensor(n.tensor, n.labels) for n in nodes])
    source.column_reduce_(output_inds=(), atol=0)
    reference = source.contract(output_inds=())
    actual_gradient = torch.autograd.grad(reduced, parameter, retain_graph=True)[0]
    source_gradient = torch.autograd.grad(reference, parameter, retain_graph=True)[0]
    ambient_gradient = torch.autograd.grad(original, parameter)[0]
    torch.testing.assert_close(actual_gradient, source_gradient)
    assert actual_gradient.tolist() == [2.0, 0.0]
    assert ambient_gradient.tolist() == [2.0, 3.0]


def test_source_nonfinite_and_zero_extent_differences_are_explicit() -> None:
    import quimb.tensor as qtn
    from quimb.tensor.array_ops import find_columns

    data = torch.tensor([float("nan"), 1.0], dtype=torch.float64)
    assert find_columns(data.numpy()) == (0, 1)
    assert find_columns(data) is None
    with pytest.raises(ValueError, match="finite tensor"):
        find_tensor_column(data)
    node = TensorNetworkNode(torch.empty(0, 1), (0, 1))
    source = qtn.TensorNetwork([qtn.Tensor(node.tensor.numpy(), node.labels)])
    with pytest.raises(ValueError):
        source.squeeze_(exclude=())
    native = squeeze_tensor_axes((node,), ())
    assert native[0].labels == (0,) and native[0].tensor.shape == (0,)


@pytest.mark.parametrize("atol", [-1.0, float("nan"), float("inf"), True, "bad"])
def test_invalid_tolerance_is_refused(atol: Any) -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        find_tensor_column(torch.ones(2), atol=atol)


def test_admission_failures_and_scalar_identity() -> None:
    for data, message in (
        (torch.empty(2, device="meta"), "meta"),
        (torch.ones(2, dtype=torch.int64), "floating or complex"),
        (torch.tensor([float("inf")]), "finite tensor"),
    ):
        with pytest.raises(ValueError, match=message):
            find_tensor_column(data)
    scalar = TensorNetworkNode(torch.tensor(2j, dtype=torch.complex128), ())
    assert reduce_tensor_columns((scalar,)) == (scalar,)
    for operation in (squeeze_tensor_axes, reduce_tensor_columns):
        with pytest.raises(ValueError, match="at least one"):
            operation(())
        node = TensorNetworkNode(torch.ones(2), (0,))
        for outputs in ((0, 0), (99,)):
            with pytest.raises(ValueError, match="distinct existing"):
                operation((node,), outputs)


@pytest.mark.parametrize("prepared", [False, True])
def test_real_bell_network_squeeze_columns_rank_replace_same_consumer(
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
    phase = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    phased = (
        replace(nodes[0], tensor=nodes[0].tensor * torch.exp(1j * phase)),
        *nodes[1:],
    )
    reduced = simplify_tensor_ranks(
        reduce_tensor_columns(squeeze_tensor_axes(phased, outputs), outputs, atol=0),
        outputs,
    )
    actual = contract_register_tensors(reduced, signature, inc, out)
    expected = contract_register_tensors(phased, signature, inc, out)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual.imag.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected.imag.sum(), phase)[0],
        rtol=0,
        atol=2e-14,
    )


def test_source_dependency_hashes_unchanged() -> None:
    import hashlib
    import importlib.metadata
    import inspect

    import quimb.tensor as qtn
    from quimb.tensor.array_ops import _numba_find_columns, find_columns

    assert importlib.metadata.version("quimb") == "1.15.0"
    expected = {
        "column_reduce": "cd9f0952b0704c01baf3db84a10aa76b59e693d4a5ec8c228d18bb5ce132018e",
        "squeeze": "f39812d153c60d87fe408a8afad8fa1f265a12ffc3a39b04dd5cb83e573ce33d",
        "find_columns": "6c4ef75c51594c4f6f1a43c52f12b2fbce96c1d1d76776f11d74ac35c6bb59c4",
        "numba_find_columns": "dc5fe0346981af7bf93a10c35546be7adcb0a8ddfaaaf6640ce4d79d2c2143e2",
    }
    original = {
        "column_reduce": qtn.TensorNetwork.column_reduce,
        "squeeze": qtn.Tensor.squeeze,
        "find_columns": find_columns,
        "numba_find_columns": _numba_find_columns,
    }
    for name, function in original.items():
        assert (
            hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            == expected[name]
        )


def test_example_script(capsys: pytest.CaptureFixture[str]) -> None:
    import runpy
    from pathlib import Path

    from examples.algorithms.tensor_column_reduction import main

    main()
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/algorithms/tensor_column_reduction.py"
        ),
        run_name="__main__",
    )
    assert "Squeeze/C/R: 4 -> 1 axis occurrences" in capsys.readouterr().out
