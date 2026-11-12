"""Source A/D operation traces, finite structure and explicit gradient boundaries."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.simulation.tensor_network.diagonal_reduction import (
    gauge_tensor_antidiagonals,
    reduce_tensor_diagonals,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy
from flagquantum.simulation.tensor_network.tensor_structure import find_tensor_diagonal

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


def dense(nodes: Sequence[TensorNetworkNode], outputs: tuple[int, ...]) -> torch.Tensor:
    return _contract_nodes_greedy(nodes, outputs)[0]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("case", range(9))
def test_candidate_order_matches_numpy_reference(
    dtype: torch.dtype, anti: bool, case: int
) -> None:
    from quimb.tensor.array_ops import find_antidiag_axes, find_diag_axes

    rectangular = torch.zeros(3, 2, 3)
    rectangular[torch.arange(3), :, torch.arange(3)] = 1
    values = [
        torch.ones(2, 3),
        torch.diag(torch.tensor([1.0, 2.0])),
        torch.tensor([[0.0, 1.0], [2.0, 0.0]]),
        torch.ones(()),
        torch.zeros(2, 2, 2),
        rectangular,
        torch.empty(0, 0),
        torch.empty(0, 2, 2),
        torch.ones(2, 2),
    ]
    data = values[case].to(dtype)
    original = find_antidiag_axes if anti else find_diag_axes
    assert find_tensor_diagonal(data, antidiagonal=anti) == original(data.numpy())


@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("side", [-1, 0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strict_tolerance_boundary(anti: bool, side: int, dtype: torch.dtype) -> None:
    from quimb.tensor.array_ops import find_antidiag_axes, find_diag_axes

    value = torch.tensor(1e-12, dtype=dtype)
    if side:
        value = torch.nextafter(
            value, torch.tensor(float("inf") if side > 0 else 0.0, dtype=dtype)
        )
    data = torch.tensor([[1.0, 0.0], [value.item(), 2.0]], dtype=dtype)
    if anti:
        data = data.flip((1,))
    original = find_antidiag_axes if anti else find_diag_axes
    assert find_tensor_diagonal(data, antidiagonal=anti) == original(
        data.numpy(), atol=1e-12
    )
    assert find_tensor_diagonal(data, antidiagonal=anti, atol=0) is None


def compare_source(
    nodes: tuple[TensorNetworkNode, ...],
    outputs: tuple[int, ...],
    anti: bool,
    monkeypatch: pytest.MonkeyPatch,
    *,
    atol: float = 0,
) -> tuple[TensorNetworkNode, ...]:
    import quimb.tensor as qtn

    import flagquantum.simulation.tensor_network.diagonal_reduction as native

    source = qtn.TensorNetwork(
        [qtn.Tensor(n.tensor.detach().numpy(), n.labels) for n in nodes]
    )
    source_trace: list[tuple[int, ...]] = []
    native_trace: list[tuple[int, ...]] = []
    original_flip = qtn.TensorNetwork.flip_
    original_reindex = qtn.TensorNetwork.reindex_
    original_native_flip = native._flip_edge
    original_native_merge = native._merge_edge

    def flip(self: Any, labels: list[int]) -> Any:
        source_trace.append(tuple(labels))
        return original_flip(self, labels)

    def reindex(self: Any, mapping: dict[int, int]) -> Any:
        source_trace.extend(tuple(pair) for pair in mapping.items())
        return original_reindex(self, mapping)

    def native_flip(
        active: tuple[TensorNetworkNode, ...], label: int
    ) -> tuple[TensorNetworkNode, ...]:
        native_trace.append((label,))
        return original_native_flip(active, label)

    def native_merge(
        active: tuple[TensorNetworkNode, ...], removed: int, retained: int
    ) -> tuple[TensorNetworkNode, ...]:
        native_trace.append((removed, retained))
        return original_native_merge(active, removed, retained)

    with monkeypatch.context() as patch:
        patch.setattr(qtn.TensorNetwork, "flip_", flip)
        patch.setattr(qtn.TensorNetwork, "reindex_", reindex)
        patch.setattr(native, "_flip_edge", native_flip)
        patch.setattr(native, "_merge_edge", native_merge)
        if anti:
            source.antidiag_gauge_(output_inds=outputs, atol=atol)
            result = gauge_tensor_antidiagonals(nodes, outputs, atol=atol)
        else:
            source.diagonal_reduce_(output_inds=outputs, atol=atol)
            result = reduce_tensor_diagonals(nodes, outputs, atol=atol)
    assert source_trace == native_trace
    for actual, expected, original in zip(result, source.tensors, nodes, strict=True):
        assert actual.labels == expected.inds
        np.testing.assert_array_equal(actual.tensor.detach().numpy(), expected.data)
        assert actual.metadata is original.metadata and actual.name == original.name
    if atol == 0 or anti:
        torch.testing.assert_close(dense(result, outputs), dense(nodes, outputs))
    return result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("outputs", [(), (0,), (1,), (0, 1)])
def test_source_operation_trace_each_leaf_and_output_direction(
    dtype: torch.dtype,
    anti: bool,
    outputs: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = torch.diag(torch.tensor([1.0, 2.0], dtype=dtype))
    if anti:
        data = data.flip((1,))
    nodes = (
        TensorNetworkNode(torch.tensor([2.0, 3.0], dtype=dtype), (0,), "left"),
        TensorNetworkNode(data, (0, 1), "structure", {"test": "metadata"}),
        TensorNetworkNode(torch.tensor([4.0, 5.0], dtype=dtype), (1,), "right"),
    )
    originals = tuple(n.tensor.clone() for n in nodes)
    compare_source(nodes, outputs, anti, monkeypatch)
    for node, before in zip(nodes, originals, strict=True):
        assert torch.equal(node.tensor, before)


@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_three_axis_structure_hyperedges_and_queue(
    anti: bool, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = torch.Generator().manual_seed(seed)
    structured = torch.zeros(2, 3, 2, dtype=torch.complex128)
    structured[0, :, 1 if anti else 0] = torch.randn(
        3, dtype=torch.complex128, generator=generator
    )
    structured[1, :, 0 if anti else 1] = torch.randn(
        3, dtype=torch.complex128, generator=generator
    )
    nodes = (
        TensorNetworkNode(
            torch.randn(2, dtype=torch.complex128, generator=generator), (0,)
        ),
        TensorNetworkNode(structured, (0, 2, 1)),
        TensorNetworkNode(
            torch.randn(2, 3, dtype=torch.complex128, generator=generator), (1, 2)
        ),
    )
    compare_source(nodes, (), anti, monkeypatch)


@pytest.mark.parametrize("anti", [False, True])
def test_protected_first_pair_and_zero_singleton_termination(
    anti: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes = (
        TensorNetworkNode(torch.zeros(2, 2, 2, dtype=torch.complex128), (0, 1, 2)),
        TensorNetworkNode(torch.ones(2, dtype=torch.complex128), (2,)),
    )
    result = compare_source(nodes, (0, 1), anti, monkeypatch)
    assert tuple(n.labels for n in result) == ((0, 1, 2), (2,))
    units = (TensorNetworkNode(torch.ones(1, 1, dtype=torch.complex128), (0, 1)),)
    compare_source(units, (), anti, monkeypatch)


def test_repeated_edge_flip_preserves_trace_while_source_first_axis_changes_it() -> (
    None
):
    import quimb.tensor as qtn

    node = TensorNetworkNode(
        torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float64), (0, 0)
    )
    source = qtn.TensorNetwork([qtn.Tensor(node.tensor.numpy(), node.labels)])
    source.antidiag_gauge_(output_inds=(), atol=0)
    result = gauge_tensor_antidiagonals((node,), (), atol=0)
    assert source.contract(output_inds=()) == 3
    assert dense(result, ()).item() == dense((node,), ()).item() == 0
    assert torch.equal(result[0].tensor, node.tensor.flip((0, 1)))


def test_diagonal_repeated_input_and_created_neighbor_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeated = (
        TensorNetworkNode(
            torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64)), (0, 0)
        ),
    )
    result = compare_source(repeated, (), False, monkeypatch)
    assert result[0].labels == (0,) and dense(result, ()).item() == 3
    nodes = (
        TensorNetworkNode(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64), (0, 1)
        ),
        TensorNetworkNode(
            torch.diag(torch.tensor([5.0, 6.0], dtype=torch.float64)), (0, 1)
        ),
    )
    result = compare_source(nodes, (), False, monkeypatch)
    assert result[0].labels == (0, 0) and result[1].labels == (0,)


@pytest.mark.parametrize("anti", [False, True])
def test_fixed_support_phase_gradient_and_gauge_ambient_gradient(anti: bool) -> None:
    phase = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    data = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.complex128))
    if anti:
        data = data.flip((1,))
    nodes = (
        TensorNetworkNode(data * torch.exp(1j * phase), (0, 1)),
        TensorNetworkNode(
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.complex128), (0, 1)
        ),
    )
    operation = gauge_tensor_antidiagonals if anti else reduce_tensor_diagonals
    actual = dense(operation(nodes, (), atol=0), ())
    expected = dense(nodes, ())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.imag, phase, retain_graph=True)[0],
        torch.autograd.grad(expected.imag, phase)[0],
    )
    if anti:
        parameter = data.real.clone().requires_grad_()
        trainable = (
            replace(nodes[0], tensor=parameter),
            replace(nodes[1], tensor=nodes[1].tensor.real),
        )
        actual = dense(operation(trainable, (), atol=0), ())
        expected = dense(trainable, ())
        torch.testing.assert_close(
            torch.autograd.grad(actual, parameter, retain_graph=True)[0],
            torch.autograd.grad(expected, parameter)[0],
        )


def test_diagonal_tolerance_and_ambient_gradient_match_source_boundary() -> None:
    import quimb.tensor as qtn

    parameter = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64, requires_grad=True
    )
    environment = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    nodes = (
        TensorNetworkNode(parameter, (0, 1)),
        TensorNetworkNode(environment, (0, 1)),
    )
    reduced = dense(reduce_tensor_diagonals(nodes, (), atol=0), ())
    original = dense(nodes, ())
    source = qtn.TensorNetwork([qtn.Tensor(n.tensor, n.labels) for n in nodes])
    source.diagonal_reduce_(output_inds=(), atol=0)
    reference = source.contract(output_inds=())
    actual_gradient = torch.autograd.grad(reduced, parameter, retain_graph=True)[0]
    source_gradient = torch.autograd.grad(reference, parameter, retain_graph=True)[0]
    ambient = torch.autograd.grad(original, parameter)[0]
    torch.testing.assert_close(actual_gradient, source_gradient)
    assert actual_gradient.tolist() == [[3.0, 0.0], [0.0, 6.0]]
    assert ambient.tolist() == [[3.0, 4.0], [5.0, 6.0]]
    tiny = (
        replace(
            nodes[0],
            tensor=torch.tensor([[1.0, 0.5e-12], [0.0, 2.0]], dtype=torch.float64),
        ),
        replace(
            nodes[1],
            tensor=torch.tensor([[3.0, 1e12], [5.0, 6.0]], dtype=torch.float64),
        ),
    )
    assert dense(reduce_tensor_diagonals(tiny, (), atol=1e-12), ()).item() == 15
    assert dense(tiny, ()).item() == 15.5


def test_refusals_and_default_outputs() -> None:
    for operation in (gauge_tensor_antidiagonals, reduce_tensor_diagonals):
        with pytest.raises(ValueError, match="at least one"):
            operation(())
        node = TensorNetworkNode(torch.eye(2), (0, 1))
        assert operation((node,))[0] is node
        with pytest.raises(ValueError, match="distinct existing"):
            operation((node,), (99,))
        with pytest.raises(ValueError, match="meta"):
            operation((replace(node, tensor=node.tensor.to("meta")),))
    with pytest.raises(ValueError, match="built-in boolean"):
        invalid_flag: Any = 1
        find_tensor_diagonal(torch.eye(2), antidiagonal=invalid_flag)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("fixed_point", [False, True])
def test_bell_composition_all_passes_same_consumer_phase_and_gradient(
    prepared: bool,
    fixed_point: bool,
) -> None:
    from examples.algorithms.bell_tensor_workflow import bell_program, bell_providers
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_component_root_for_tensor,
    )
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.simulation.tensor_network.column_reduction import (
        reduce_tensor_columns,
    )
    from flagquantum.simulation.tensor_network.component_tensor_nodes import (
        component_tensor_nodes_with_labels,
    )
    from flagquantum.simulation.tensor_network.rank_simplification import (
        simplify_tensor_ranks,
    )
    from flagquantum.simulation.tensor_network.register_tensor_binding import (
        contract_register_tensors,
    )
    from flagquantum.simulation.tensor_network.register_tensor_indices import (
        register_contraction_groups,
    )
    from flagquantum.simulation.tensor_network.simplification_axes import (
        squeeze_tensor_axes,
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
    groups = register_contraction_groups(nodes, signature, inc, out)
    outputs = tuple(label for group in groups for label in group)
    phase = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    phased = (
        replace(nodes[0], tensor=nodes[0].tensor * torch.exp(1j * phase)),
        *nodes[1:],
    )
    if fixed_point:
        from flagquantum.simulation.tensor_network.full_simplification import (
            simplify_tensor_network,
        )

        reduced = simplify_tensor_network(phased, outputs, atol=0)
    else:
        reduced = squeeze_tensor_axes(phased, outputs)
        reduced = gauge_tensor_antidiagonals(reduced, outputs, atol=0)
        reduced = reduce_tensor_diagonals(reduced, outputs, atol=0)
        reduced = reduce_tensor_columns(reduced, outputs, atol=0)
        reduced = simplify_tensor_ranks(reduced, outputs)
    actual = contract_register_tensors(
        reduced, signature, inc, out, output_groups=groups
    )
    expected = contract_register_tensors(phased, signature, inc, out)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual.imag.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(expected.imag.sum(), phase)[0],
        rtol=0,
        atol=2e-14,
    )


def test_dependency_source_hashes_unchanged() -> None:
    import hashlib
    import importlib.metadata
    import inspect

    import quimb.tensor as qtn
    from quimb.tensor import array_ops

    assert importlib.metadata.version("quimb") == "1.15.0"
    functions = {
        "A": qtn.TensorNetwork.antidiag_gauge,
        "D": qtn.TensorNetwork.diagonal_reduce,
        "FLIP": qtn.Tensor.flip,
        "FA": array_ops.find_antidiag_axes,
        "FD": array_ops.find_diag_axes,
        "NA": array_ops._numba_find_antidiag_axes,
        "ND": array_ops._numba_find_diag_axes,
    }
    expected = {
        "A": "407328b6216b3ec7855139c8d73f4756482e6d80b0cdbaef171035293615cc10",
        "D": "2f8688060fe3b9d8889cd6776b8c70211e5cd0cdcf1cf7cb902faf1ed7157f7d",
        "FLIP": "403fa01a5cfeddf739dbeaa4ffd58aa4b73aea49f3ff3bb45aa6f093c863e429",
        "FA": "af67cbcc85833c1418f6c683560bd90e27dcdd13c7bad3c55140ca552170682f",
        "FD": "300d744353cb525adcf294957b4c913a0088209168e81cc93d7b59457c4b1366",
        "NA": "db17dc20e7d9ffe9f143da414655ed5f0546838a3f981ffbf2182a5f1236ac9d",
        "ND": "2e648627499e3ca84c4e1d9e900a8b49b196019a428de853df2c908482467751",
    }
    for name, function in functions.items():
        assert (
            hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            == expected[name]
        )


def test_example_entry(capsys: pytest.CaptureFixture[str]) -> None:
    import runpy
    from pathlib import Path

    from examples.algorithms.tensor_diagonal_reduction import main

    main()
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/algorithms/tensor_diagonal_reduction.py"
        ),
        run_name="__main__",
    )
    assert "A/D/C/R: 3 -> 1 tensors; 5 -> 1 axis occurrences" in capsys.readouterr().out


@pytest.mark.parametrize("mixed", [False, True])
def test_captured_register_outputs_survive_diagonal_hyperedges(mixed: bool) -> None:
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

    signature = Signature(
        (Register("q", RegisterType("opaque", 1), side=RegisterSide.RIGHT),)
    )
    incoming: dict[str, tuple[tuple[int, ...], ...]] = {}
    outgoing = {"q": ((0,),)}
    # In doubled mode q forward/backward are labels 0 and 1.
    vector = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128, requires_grad=True)
    nodes: tuple[TensorNetworkNode, ...] = (
        TensorNetworkNode(torch.eye(2, dtype=torch.complex128), (0, 2)),
        TensorNetworkNode(vector, (2,)),
    )
    if mixed:
        nodes += (TensorNetworkNode(vector.conj(), (1,)),)
    consumer = contract_register_superoperator if mixed else contract_register_tensors
    groups = register_contraction_groups(
        nodes, signature, incoming, outgoing, superoperator=mixed
    )
    outputs = tuple(label for group in groups for label in group)
    expected = consumer(nodes, signature, incoming, outgoing)
    reduced = reduce_tensor_diagonals(nodes, outputs, atol=0)
    with pytest.raises(KeyError):
        consumer(reduced, signature, incoming, outgoing)
    actual = consumer(reduced, signature, incoming, outgoing, output_groups=groups)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.real.sum(), vector, retain_graph=True)[0],
        torch.autograd.grad(expected.real.sum(), vector)[0],
    )
    with pytest.raises(ValueError, match="captured output groups"):
        consumer(reduced, signature, incoming, outgoing, output_groups=((99,),))
    missing = tuple(node for node in reduced if 0 not in node.labels)
    with pytest.raises(KeyError):
        consumer(
            missing or (TensorNetworkNode(torch.ones(()), ()),),
            signature,
            incoming,
            outgoing,
            output_groups=groups,
        )
