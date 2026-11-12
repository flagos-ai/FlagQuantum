"""Source-default Cotengra width, threshold policy and existing dense consumers."""

import builtins
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.contraction_width import (
    tensor_contraction_width,
)
from flagquantum.simulation.tensor_network.full_simplification import (
    simplify_wide_tensor_network,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "shapes,labels",
    [
        (((),), ((),)),
        (((2,),), ((0,),)),
        (((2, 2),), ((0, 0),)),
        (((2, 3), (3, 5)), ((0, 1), (1, 2))),
        (((2, 3), (3, 5), (3, 7)), ((0, 1), (1, 2), (1, 3))),
        (((2, 2), (2, 2), (2, 2)), ((0, 1), (1, 2), (2, 0))),
        (((3, 3, 2), (2, 5)), ((0, 0, 1), (1, 2))),
        (((2, 2), (3, 3)), ((0, 1), (2, 3))),
    ],
)
@pytest.mark.parametrize("meta", [False, True])
def test_exact_planner_inputs_path_and_width_without_reading_values(
    shapes: tuple[tuple[int, ...], ...],
    labels: tuple[tuple[int, ...], ...],
    meta: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cotengra as ctg
    import quimb.tensor as qtn
    from quimb.tensor.contraction import get_contract_strategy

    assert get_contract_strategy() == "greedy"
    nodes = tuple(
        TensorNetworkNode(torch.empty(shape, device="meta" if meta else "cpu"), inds)
        for shape, inds in zip(shapes, labels, strict=True)
    )
    source = qtn.TensorNetwork([qtn.Tensor(n.tensor, n.labels) for n in nodes])
    calls: list[tuple[object, ...]] = []
    original = ctg.array_contract_tree

    def plan(inputs: Any, **kwargs: Any) -> Any:
        tree = original(inputs, **kwargs)
        calls.append((inputs, kwargs, tuple(tree.get_path()), tree.contraction_width()))
        return tree

    monkeypatch.setattr(ctg, "array_contract_tree", plan)
    expected = source.contraction_width()
    actual = tensor_contraction_width(nodes)
    assert actual == expected
    assert len(calls) == 2 and calls[0] == calls[1]


def test_width_is_not_raw_rank_or_native_memory_proxy() -> None:
    # Two rank-13 leaves contract directly to a scalar. Largest produced
    # intermediate has one element, despite the much larger input storage.
    nodes = tuple(
        TensorNetworkNode(torch.empty((2,) * 13, device="meta"), tuple(range(13)))
        for _ in range(2)
    )
    assert tensor_contraction_width(nodes) == 0
    assert simplify_wide_tensor_network(nodes, (), atol=0) == nodes


@pytest.mark.parametrize("width", [12, 13])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_real_threshold_source_dense_output_and_phase_gradient(
    width: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qualtran
    import quimb.tensor as qtn
    from qualtran.simulation.tensor._dense import quimb_to_dense

    import flagquantum.simulation.tensor_network.full_simplification as native
    from examples.algorithms.tensor_rank_simplification import wide_state_network

    nodes, outputs = wide_state_network(width)
    phase = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    nodes = tuple(replace(n, tensor=n.tensor.to(dtype)) for n in nodes)
    nodes = (
        replace(nodes[0], tensor=nodes[0].tensor * torch.exp(1j * phase)),
        *nodes[1:],
    )
    names = {label: ("q", (), i, "r") for i, label in enumerate(outputs)}
    source = qtn.TensorNetwork(
        [
            qtn.Tensor(
                n.tensor.detach().numpy(), tuple(names.get(x, x) for x in n.labels)
            )
            for n in nodes
        ]
    )
    reference_calls: list[dict[str, object]] = []
    original = source.full_simplify

    def source_full(**kwargs: Any) -> Any:
        reference_calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(source, "full_simplify", source_full)
    native_calls: list[tuple[int, ...]] = []
    actual_full = native.simplify_tensor_network

    def native_full(active: Any, labels: Any, **kwargs: Any) -> Any:
        native_calls.append(tuple(labels))
        return actual_full(active, labels, **kwargs)

    monkeypatch.setattr(native, "simplify_tensor_network", native_full)
    source_signature = qualtran.Signature(
        (qualtran.Register("q", qualtran.QUInt(width), side=qualtran.Side.RIGHT),)
    )
    expected = quimb_to_dense(source, source_signature)
    signature = Signature(
        (Register("q", RegisterType("integer", width), side=RegisterSide.RIGHT),)
    )
    outgoing = {"q": (outputs,)}
    actual = contract_register_tensors(
        nodes, signature, {}, outgoing, simplification_atol=1e-12
    )
    assert len(reference_calls) == len(native_calls) == int(width > 12)
    if width > 12:
        assert reference_calls == [{"inplace": True}] and native_calls == [outputs]
    tolerance = 4e-6 if dtype == torch.complex64 else 3e-14
    assert bool(torch.isfinite(actual).all()) and np.isfinite(expected).all()
    np.testing.assert_allclose(
        actual.detach().numpy(), expected, rtol=tolerance, atol=tolerance
    )
    baseline = contract_register_tensors(nodes, signature, {}, outgoing)
    torch.testing.assert_close(actual, baseline)
    torch.testing.assert_close(
        torch.autograd.grad(actual.imag.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(baseline.imag.sum(), phase)[0],
    )


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("bits", [6, 7])
def test_doubled_and_unitary_groups_before_threshold_and_same_consumer(
    mixed: bool,
    bits: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qualtran
    import quimb.tensor as qtn
    from qualtran.simulation.tensor._dense import quimb_to_dense

    import flagquantum.simulation.tensor_network.full_simplification as native

    phase = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    width = 2 * bits
    values = torch.arange(1, 1 + (1 << width), dtype=torch.float64).to(torch.complex128)
    values = values.reshape((2,) * width) * torch.exp(1j * phase)
    # Mixed state: forward and backward outputs. Unitary: right and left outputs.
    labels = tuple(range(width))
    nodes = (TensorNetworkNode(values, labels),)
    signature = Signature(
        (
            Register(
                "q",
                RegisterType("integer", bits),
                side=RegisterSide.RIGHT if mixed else RegisterSide.THRU,
            ),
        )
    )
    incoming = {} if mixed else {"q": (tuple(range(bits, width)),)}
    outgoing = {"q": (tuple(range(bits)),)}
    names = (
        {i: ("q", (), i // 2, "rb" if i % 2 else "rf") for i in labels}
        if mixed
        else {i: ("q", (), i % bits, "r" if i < bits else "l") for i in labels}
    )
    source = qtn.TensorNetwork(
        [qtn.Tensor(values.detach().numpy(), tuple(names[i] for i in labels))]
    )
    ss = qualtran.Signature(
        (
            qualtran.Register(
                "q",
                qualtran.QUInt(bits),
                side=qualtran.Side.RIGHT if mixed else qualtran.Side.THRU,
            ),
        )
    )
    expected = quimb_to_dense(source, ss, superoperator=mixed)
    calls: list[tuple[int, ...]] = []
    original = native.simplify_tensor_network

    def track(active: Any, outputs: Any, **kwargs: Any) -> Any:
        calls.append(tuple(outputs))
        return original(active, outputs, **kwargs)

    monkeypatch.setattr(native, "simplify_tensor_network", track)
    consumer = contract_register_superoperator if mixed else contract_register_tensors
    actual = consumer(nodes, signature, incoming, outgoing, simplification_atol=0)
    # Cotengra0.8.3 singleton max_size returns before applying log2.
    assert tensor_contraction_width(nodes) == 1 << width
    assert len(calls) == 1
    assert tuple(actual.shape) == (1 << bits, 1 << bits)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=0, atol=1e-12)
    baseline = consumer(nodes, signature, incoming, outgoing)
    torch.testing.assert_close(
        torch.autograd.grad(actual.real.sum(), phase, retain_graph=True)[0],
        torch.autograd.grad(baseline.real.sum(), phase)[0],
    )


@pytest.mark.parametrize("invalid", [-1, True, float("nan"), float("inf"), "0"])
def test_invalid_tolerance_even_below_threshold(invalid: Any) -> None:
    nodes = (TensorNetworkNode(torch.ones(2), (0,)),)
    with pytest.raises(ValueError, match="tolerance"):
        simplify_wide_tensor_network(nodes, (0,), atol=invalid)


@pytest.mark.parametrize("invalid", [True, -1.0, float("nan"), float("inf"), "13"])
def test_invalid_planner_width_is_not_a_trigger(
    invalid: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cotengra as ctg

    monkeypatch.setattr(
        ctg,
        "array_contract_tree",
        lambda *a, **k: SimpleNamespace(contraction_width=lambda: invalid),
    )
    with pytest.raises(ValueError, match="planner"):
        tensor_contraction_width((TensorNetworkNode(torch.ones(2), (0,)),))


def test_no_optional_planner_fallback_and_default_path_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def block(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cotengra":
            raise ImportError("deliberately absent optional package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)
    signature = Signature(
        (Register("q", RegisterType("integer", 1), side=RegisterSide.RIGHT),)
    )
    nodes = (TensorNetworkNode(torch.ones(2), (0,)),)
    with pytest.raises(ImportError, match="optional 'cotengra'"):
        contract_register_tensors(
            nodes, signature, {}, {"q": ((0,),)}, simplification_atol=0
        )
    torch.testing.assert_close(
        contract_register_tensors(nodes, signature, {}, {"q": ((0,),)}), torch.ones(2)
    )


def test_invalid_network_shapes_refused_before_planning() -> None:
    with pytest.raises(ValueError, match="at least one"):
        tensor_contraction_width(())
    with pytest.raises(ValueError, match="positive"):
        tensor_contraction_width((TensorNetworkNode(torch.empty(0), (0,)),))
    with pytest.raises(ValueError, match="Inconsistent dimension"):
        tensor_contraction_width(
            (
                TensorNetworkNode(torch.empty(2), (0,)),
                TensorNetworkNode(torch.empty(3), (0,)),
            )
        )


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_component_entries_forward_explicit_policy_on_real_bell_graph(
    prepared: bool,
    mixed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Iterable, Mapping

    import cotengra as ctg

    from examples.algorithms.bell_tensor_workflow import bell_program, bell_providers
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_component_root_for_tensor,
    )
    from flagquantum.core.ir.program import FunctionRecord
    from flagquantum.simulation.tensor_network.component_dense import (
        contract_component_superoperator,
        contract_component_tensors,
    )

    record = bell_program(prepared=prepared)
    like = torch.empty((), dtype=torch.complex128)
    providers = bell_providers(record, like, factorized_cnot=False)
    selected, _ = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, "bell"
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))

    def no_fresh() -> int:
        raise AssertionError("dense provider must not allocate private labels")

    def supply(
        function: FunctionRecord,
        inc: Mapping[str, tuple[tuple[int, ...], ...]],
        out: Mapping[str, tuple[tuple[int, ...], ...]],
    ) -> Iterable[TensorNetworkNode | int]:
        return providers.supply(function, inc, out, no_fresh)

    calls: list[str] = []
    original = ctg.array_contract_tree

    def track(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["optimize"])
        return original(*args, **kwargs)

    monkeypatch.setattr(ctg, "array_contract_tree", track)
    if mixed:
        actual = contract_component_superoperator(
            selected,
            order,
            providers.supply,
            like,
            "tensor_root",
            simplification_atol=0,
        )
        expected = contract_component_superoperator(
            selected, order, providers.supply, like, "tensor_root"
        )
    else:
        actual = contract_component_tensors(
            selected,
            order,
            supply,
            like,
            "tensor_root",
            simplification_atol=0,
        )
        expected = contract_component_tensors(
            selected, order, supply, like, "tensor_root"
        )
    assert calls == ["greedy"]
    torch.testing.assert_close(actual, expected)


def test_real_wide_bell_output_hyperedge_keeps_precomputed_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections import Counter

    import qualtran
    import quimb.tensor as qtn
    from qualtran.simulation.tensor._dense import quimb_to_dense

    import flagquantum.simulation.tensor_network.register_tensor_binding as consumer
    from examples.algorithms.bell_tensor_workflow import bell_program, bell_providers
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_component_root_for_tensor,
    )
    from flagquantum.simulation.tensor_network.component_tensor_nodes import (
        component_tensor_nodes_with_labels,
    )
    from flagquantum.simulation.tensor_network.register_tensor_indices import (
        register_tensor_outer_indices,
    )

    record = bell_program(prepared=False)
    like = torch.empty((), dtype=torch.complex128)
    providers = bell_providers(record, like)
    selected, _ = expand_component_root_for_tensor(
        record, providers.has_custom_tensor, "bell"
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected, "tensor_root"))
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        selected, order, providers.supply, like, "tensor_root"
    )
    labels = tuple(
        range(
            1 + max(x for n in nodes for x in n.labels),
            10 + max(x for n in nodes for x in n.labels),
        )
    )
    nodes += tuple(
        TensorNetworkNode(torch.tensor([1.0, 2.0], dtype=torch.complex128), (label,))
        for label in labels
    )
    registers = selected.functions[0].registers or ()
    signature = Signature(
        (
            *registers,
            Register("anc", RegisterType("integer", 9), side=RegisterSide.RIGHT),
        )
    )
    outgoing = {**outgoing, "anc": (labels,)}
    names = register_tensor_outer_indices(signature, incoming, outgoing)
    source = qtn.TensorNetwork(
        [
            qtn.Tensor(n.tensor.numpy(), tuple(names.get(x, x) for x in n.labels))
            for n in nodes
        ]
    )
    ss = qualtran.Signature(
        tuple(
            qualtran.Register(
                reg.name,
                qualtran.QUInt(int(reg.bitsize)),
                side=qualtran.Side[reg.side.name],
            )
            for reg in signature
        )
    )
    assert tensor_contraction_width(nodes) == source.contraction_width() == 13
    expected = quimb_to_dense(source, ss)
    original = getattr(consumer, "_contract_nodes_greedy")
    seen: list[bool] = []

    def contract(active: Any, outputs: Any) -> Any:
        counts = Counter(x for n in active for x in n.labels)
        seen.append(any(counts[x] > 1 for x in outputs))
        return original(active, outputs)

    monkeypatch.setattr(consumer, "_contract_nodes_greedy", contract)
    actual = consumer.contract_register_tensors(
        nodes, signature, incoming, outgoing, simplification_atol=0
    )
    assert seen == [True]
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=2e-12)


def test_locked_source_chain_hashes_and_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hashlib
    import importlib.metadata
    import inspect
    import runpy
    from pathlib import Path

    import cotengra as ctg
    import quimb.tensor as qtn
    from quimb.tensor import contraction

    from examples.algorithms.tensor_width_policy import main

    assert importlib.metadata.version("cotengra") == "0.8.3"
    functions = {
        "WIDTH": qtn.TensorNetwork.contraction_width,
        "TREE": qtn.TensorNetwork.contraction_tree,
        "DELEGATE": contraction.array_contract_tree,
        "CTREE": ctg.array_contract_tree,
        "MAX": ctg.ContractionTree.max_size,
        "CWIDTH": ctg.ContractionTree.contraction_width,
    }
    hashes = {
        "WIDTH": "2c8252fa297cae3644361df74a6097c31549a7eca29d9d49f57f01f76dccc0bb",
        "TREE": "0fc56992da6eaa029b2630d4f85715d6226267467462bbabe9b11610c846acca",
        "DELEGATE": "32c5ff93ec3cfa2b14d1757b1ce46c9999fff6e8555cd94395f50dd32647cb6e",
        "CTREE": "a55d170a34cbd751506cead0964588e6100f47b3949fd7a346bcca1e33748aeb",
        "MAX": "fb2cfd0b6ecc8ca083196ac64eb1f478d0c3a84832924fa9d1e31fd580dccd0a",
        "CWIDTH": "530b975ce4e9a23f4d2ca0bdd451764100eb054bf381e3636026d8746e309b1e",
    }
    for key, function in functions.items():
        source = inspect.getsource(function.__code__ if key == "DELEGATE" else function)
        assert hashlib.sha256(source.encode()).hexdigest() == hashes[key]
    main()
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/algorithms/tensor_width_policy.py"
        ),
        run_name="__main__",
    )
    text = capsys.readouterr().out
    assert "Width 12: ADCR not selected" in text and "Width 13: ADCR selected" in text
