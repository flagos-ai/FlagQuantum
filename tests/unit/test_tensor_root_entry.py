"""Explicit typed roots follow original wrapping, selection and numerical consumers."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_component_root_for_tensor,
    expand_components_for_tensor,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_superoperator_nodes import (
    component_superoperator_nodes,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_superoperator_binding import (
    contract_register_superoperator,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    bind_register_tensor,
    contract_register_tensors,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_expansion import _hierarchy
from tests.unit.test_component_superoperator_nodes import mixed_graph, supplier
from tests.unit.test_component_tensor_expansion import source_custom
from tests.unit.test_component_zero_initialization import phase_fixture
from tests.unit.test_program_owned_measurement import measured_function

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Ports = Mapping[str, tuple[tuple[int, ...], ...]]


def source_root(composite: Any) -> Any:
    """Represent a stored decomposition as a normal Bloq, without changing source."""
    from qualtran import Bloq

    factory: Any = type(
        "StoredRoot",
        (Bloq,),
        {
            "signature": property(lambda self: composite.signature),
            "decompose_bloq": lambda self: composite,
        },
    )
    return factory()


def positions(record: ProgramRecord) -> tuple[int, ...]:
    return tuple(i for i, _, _ in iter_component_calls(record, "tensor_root"))


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize(
    "kind,dtype",
    [
        ("custom", torch.float64),
        ("h", torch.float64),
        ("custom", torch.complex128),
        ("h", torch.complex128),
        ("s", torch.complex128),
    ],
)
def test_root_precedence_atomic_retention_original_labels_tensor_and_append(
    kind: str, full: bool, dtype: torch.dtype
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard, SGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.tensor._dense import bloq_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction

    record, _ = _hierarchy(1, 2)
    if kind == "custom":
        record = replace(
            record,
            functions=(
                replace(record.functions[0], name="custom"),
                record.functions[1],
            ),
        )
        original = source_custom()
    else:
        leaf = record.functions[1]
        if kind == "s":
            leaf = replace(
                leaf,
                name="s",
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            replace(op, opcode="s") if op.opcode == "h" else op
                            for op in block.operations
                        ),
                    )
                    for block in leaf.blocks
                ),
            )
        record = ProgramRecord((leaf,))
        original = Hadamard() if kind == "h" else SGate()
    entry = record.functions[0].name
    record = replace(record, canonical_payload={"stale": True})
    before = record.functions
    seen: list[str] = []

    def custom(name: str) -> bool:
        seen.append(name)
        return True

    selected, labels = expand_component_root_for_tensor(
        record, custom, entry, full_flatten=full
    )
    expected = flatten_for_tensor_contraction(original, full_flatten=full)
    assert labels == tuple(i.i for i, _, _ in expected.iter_bloqsoqs())
    expected_names = ["h", "h"] if kind == "custom" and full else [kind]
    assert [
        op.callee for op in selected.functions[0].blocks[0].operations
    ] == expected_names
    assert seen == ([] if full else [entry])
    assert record.functions is before and record.canonical_payload == {"stale": True}
    assert selected.canonical_payload is None and verify_program_record(selected) == ()
    assert all(a is b for a, b in zip(before, selected.functions[1:], strict=True))

    def supply(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> tuple[TensorNetworkNode, ...]:
        if function.name == "custom":
            matrix = torch.eye(2, dtype=dtype)
        elif function.name == "h":
            matrix = torch.tensor([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2)
        else:
            assert function.name == "s" and dtype.is_complex
            matrix = torch.diag(torch.tensor([1, 1j], dtype=dtype))
        return (
            bind_register_tensor(matrix, Signature(function.registers or ()), inc, out),
        )

    nodes, inc, out = component_tensor_nodes(
        selected,
        positions(selected),
        supply,
        torch.empty((), dtype=dtype),
        "tensor_root",
    )
    actual = contract_register_tensors(
        nodes, Signature(before[0].registers or ()), inc, out
    )
    reference = bloq_to_dense(original, full_flatten=full)
    np.testing.assert_allclose(actual.numpy(), reference, rtol=0, atol=2e-14)
    bound = bind_register_program(selected, ((0,),), entry="tensor_root", clean=(2,))
    assert bound == bind_register_program(record, ((0,),), entry=entry, clean=(2,))
    gates = (
        Circuit(3)
        .append(FieldOperation(3, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert gates == bound.circuit.instructions
    counts = Counter(g.name for g in gates)
    assert counts == ({"h": 2} if kind == "custom" else {kind: 1})
    assert get_cost_value(original, QECGatesCost()).clifford == sum(counts.values())
    theta = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    state = torch.arange(1, 5, dtype=torch.float64) * torch.exp(
        1j * theta * torch.arange(4)
    )
    state = state / state.norm()
    with_work = torch.zeros(8, dtype=torch.complex128)
    with_work[::2] = state
    branch = forced_branch(with_work, gates, ()).reshape(2, 2, 2)
    expected_state = torch.as_tensor(reference, dtype=torch.complex128) @ state.reshape(
        2, 2
    )
    torch.testing.assert_close(branch[..., 0], expected_state, atol=2e-14, rtol=0)
    assert torch.count_nonzero(branch[..., 1]) == 0
    target = actual.to(torch.complex128) @ state.reshape(2, 2)
    torch.testing.assert_close(
        torch.autograd.grad(target.real.sum(), theta, retain_graph=True)[0],
        torch.autograd.grad(expected_state.real.sum(), theta)[0],
        atol=2e-14,
        rtol=0,
    )


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
@pytest.mark.parametrize("width", [0, 1])
@pytest.mark.parametrize("full", [False, True])
def test_shaped_root_wiring_and_phaseful_decomposition_uses_existing_consumer(
    shape: tuple[int, ...], width: int, full: bool
) -> None:
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction

    from tests.unit.test_component_expansion import _compare

    record, source = phase_fixture(width, shape)
    # This source root has no custom tensor, so both modes expand it.
    selected, labels = expand_component_root_for_tensor(
        record, lambda name: name != "main", full_flatten=full, name="main_tensor"
    )
    original = flatten_for_tensor_contraction(source_root(source), full_flatten=full)
    comparable = replace(
        selected,
        functions=(
            replace(selected.functions[0], name="main"),
            *selected.functions[1:],
        ),
    )
    # _compare reads the first main entry; remove its retained original definition.
    comparable = replace(
        comparable,
        functions=tuple(
            f for i, f in enumerate(comparable.functions) if i == 0 or f.name != "main"
        ),
    )
    _compare(comparable, labels, original)
    inputs = (
        (0,),
        (1,),
        *(
            tuple(range(2 + i * width, 2 + (i + 1) * width))
            for i in range(int(np.prod(shape)))
        ),
    )
    bound = bind_register_program(
        selected, inputs, entry="main_tensor", clean=(2 + width * int(np.prod(shape)),)
    )
    assert bound == bind_register_program(
        record, inputs, clean=(2 + width * int(np.prod(shape)),)
    )
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": 3, "s": 2}

    def supply(
        function: FunctionRecord, inc: Ports, out: Ports
    ) -> tuple[TensorNetworkNode, ...]:
        matrix = (
            torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
            if function.name == "h"
            else torch.diag(torch.tensor([1, 1j], dtype=torch.complex128))
        )
        return (
            bind_register_tensor(matrix, Signature(function.registers or ()), inc, out),
        )

    order = tuple(i for i, _, _ in iter_component_calls(selected, "main_tensor"))
    nodes, inc, out = component_tensor_nodes(
        selected, order, supply, torch.empty((), dtype=torch.complex128), "main_tensor"
    )
    actual = contract_register_tensors(
        nodes, Signature(record.functions[0].registers or ()), inc, out
    )
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._quimb import cbloq_to_quimb

    expected = quimb_to_dense(cbloq_to_quimb(original), source.signature)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("discard", [False, True])
def test_mixed_root_selection_entire_original_channel_and_supplier_replacement(
    full: bool, dtype: torch.dtype, discard: bool
) -> None:
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    record, source = mixed_graph(discard)
    selected, labels = expand_component_root_for_tensor(
        record, lambda name: name != "main", full_flatten=full
    )
    original = flatten_for_tensor_contraction(source_root(source), full_flatten=full)
    assert labels == tuple(i.i for i, _, _ in original.iter_bloqsoqs())
    like = torch.empty((), dtype=dtype)
    nodes, inc, out = component_superoperator_nodes(
        selected, positions(selected), supplier(like), like, "tensor_root"
    )
    sig = Signature(record.functions[0].registers or ())
    actual = contract_register_superoperator(nodes, sig, inc, out)
    reference = quimb_to_dense(
        cbloq_to_superquimb(original), source.signature, superoperator=True
    )
    np.testing.assert_allclose(
        actual.numpy(),
        reference,
        rtol=0,
        atol=2e-6 if dtype == torch.complex64 else 2e-14,
    )
    other, left, right = component_superoperator_nodes(
        selected,
        positions(selected),
        supplier(like, original=True),
        like,
        "tensor_root",
    )
    torch.testing.assert_close(
        contract_register_superoperator(other, sig, left, right), actual
    )
    assert bind_register_program(
        selected,
        ((0,), (1,), (2,), ()),
        entry="tensor_root",
        classical_inputs=(None, None, None, 1),
        clean=(3,),
    ) == bind_register_program(
        record,
        ((0,), (1,), (2,), ()),
        classical_inputs=(None, None, None, 1),
        clean=(3,),
    )


@pytest.mark.parametrize("full", [False, True])
def test_atomic_measurement_root_full_source_channel_and_all_physical_branches(
    full: bool,
) -> None:
    from qualtran.bloqs.basic_gates import MeasureZ
    from qualtran.simulation.tensor._dense import quimb_to_dense
    from qualtran.simulation.tensor._flattening import flatten_for_tensor_contraction
    from qualtran.simulation.tensor._quimb import cbloq_to_superquimb

    record = ProgramRecord((measured_function("first"),))
    selected, labels = expand_component_root_for_tensor(
        record, lambda name: True, "first", full_flatten=full
    )
    assert labels == (0,)
    like = torch.empty((), dtype=torch.complex128)
    nodes, inc, out = component_superoperator_nodes(
        selected, positions(selected), supplier(like), like, "tensor_root"
    )
    actual = contract_register_superoperator(
        nodes, Signature(record.functions[0].registers or ()), inc, out
    )
    original = flatten_for_tensor_contraction(MeasureZ(), full_flatten=full)
    np.testing.assert_array_equal(
        actual.numpy(),
        quimb_to_dense(
            cbloq_to_superquimb(original), MeasureZ().signature, superoperator=True
        ),
    )
    bound = bind_register_program(selected, ((0,),), entry="tensor_root", clean=(2,))
    assert bound == bind_register_program(record, ((0,),), entry="first", clean=(2,))
    gates = (
        Circuit(3)
        .append(FieldOperation(3, bound.circuit.instructions))
        .to_ir()
        .instructions
    )
    assert Counter(g.name for g in gates) == {"measure": 1}
    assert gates[0].wires == (0,) and gates[0].metadata["classical_bit"] == 0
    state = torch.tensor([1 + 2j, 3 - 1j, -2 + 3j, 1 - 4j], dtype=torch.complex128)
    state /= state.norm()
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[::2] = state
    for bit in (0, 1):
        actual_branch = forced_branch(initial, gates, (bit,)).reshape(2, 2, 2)
        expected = torch.zeros_like(actual_branch)
        expected[bit, :, 0] = state.reshape(2, 2)[bit]
        torch.testing.assert_close(actual_branch, expected, rtol=0, atol=0)
        assert torch.count_nonzero(actual_branch[..., 1]) == 0


def test_invalid_unknown_untyped_collision_classifier_failure_and_depth() -> None:
    record, _ = _hierarchy(1, 2)
    with pytest.raises(ValueError, match="unknown tensor root"):
        expand_component_root_for_tensor(record, lambda name: True, "missing")
    with pytest.raises(ValueError, match="invalid tensor root"):
        expand_component_root_for_tensor(
            replace(record, functions=(*record.functions, record.functions[0])),
            lambda name: True,
        )
    with pytest.raises(ValueError, match="typed logical"):
        expand_component_root_for_tensor(
            ProgramRecord(
                (FunctionRecord("main", (BlockRecord("entry", return_values=()),)),)
            ),
            lambda name: True,
        )
    with pytest.raises(ValueError, match="conflicting"):
        expand_component_root_for_tensor(record, lambda name: True, name="main")
    error = RuntimeError("root classification failed")

    def fail(name: str) -> bool:
        raise error

    with pytest.raises(RuntimeError) as failure:
        expand_component_root_for_tensor(record, fail, full_flatten=False)
    assert failure.value is error
    expanded, _ = expand_component_root_for_tensor(record, fail)
    assert [op.callee for op in expanded.functions[0].blocks[0].operations] == [
        "h",
        "h",
    ]
    for depth in (0, 1):
        with pytest.raises(ValueError, match="Max recursion depth"):
            expand_component_root_for_tensor(record, fail, max_depth=depth)
    assert expand_components_for_tensor(record, fail)[0].functions[0].name == "main"
    empty = ComponentProgramBuilder(Signature([])).finalize({})
    result, labels = expand_component_root_for_tensor(empty, lambda name: False)
    assert result.functions[0].blocks[0].operations == () and labels == ()


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("discard", [False, True])
def test_wrapped_mixed_graph_every_measurement_history_matches_original_instrument(
    full: bool, discard: bool
) -> None:
    from itertools import product

    from qualtran.bloqs.basic_gates import MeasureX, MeasureZ
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.simulation.matrices import H_MATRIX
    from flagquantum.simulation.tensor_network.path_search import _contract_nodes_greedy

    record, source = mixed_graph(discard)
    selected, _ = expand_component_root_for_tensor(
        record, lambda name: name != "main", full_flatten=full
    )
    like = torch.empty((), dtype=torch.complex128)
    raw: list[TensorNetworkNode] = []
    environments: list[int] = []
    _, inc, _ = component_superoperator_nodes(
        selected,
        positions(selected),
        supplier(like, raw=raw, environments=environments),
        like,
        "tensor_root",
    )
    instrument, _ = _contract_nodes_greedy(
        raw,
        (
            inc["q"][0][0],
            inc["q"][1][0],
            *environments,
            raw[0].labels[1],
            raw[1].labels[2],
        ),
    )
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    state = torch.arange(1, 9, dtype=torch.float64) * torch.exp(
        1j * theta * torch.arange(8)
    )
    state = state / state.norm()
    z = torch.as_tensor(MeasureZ().my_tensors({"q": "q"}, {"c": "c"})[0].data)
    x = torch.as_tensor(MeasureX().my_tensors({"q": "q"}, {"c": "c"})[0].data)
    original = torch.einsum("iae,jfb,ijs->abefs", z, x, state.reshape(2, 2, 2))
    actual = torch.einsum("ijefab,ijs->abefs", instrument, state.reshape(2, 2, 2))
    torch.testing.assert_close(actual, original)
    torch.testing.assert_close(
        torch.autograd.grad(actual.real.sum(), theta, retain_graph=True)[0],
        torch.autograd.grad(original.real.sum(), theta)[0],
    )
    for idle in (0, 1):
        bound = bind_register_program(
            selected,
            ((0,), (1,), (2,), ()),
            entry="tensor_root",
            clean=(3,),
            classical_inputs=(None, None, None, idle),
        )
        assert bound == bind_register_program(
            record,
            ((0,), (1,), (2,), ()),
            clean=(3,),
            classical_inputs=(None, None, None, idle),
        )
        gates = (
            Circuit(4)
            .append(FieldOperation(4, bound.circuit.instructions))
            .to_ir()
            .instructions
        )
        assert gates == bound.circuit.instructions
        assert Counter(g.name for g in gates) == {
            "h": 1,
            "measure": 2,
            "classical_xor": 1,
        }
        initial = torch.zeros(16, dtype=torch.complex128)
        initial[::2] = state.detach()
        for a, b in product((0, 1), repeat=2):
            branch = forced_branch(initial, gates, (a, b)).reshape(2, 2, 2, 2)
            assert torch.count_nonzero(branch[..., 1]) == 0
            # Physical X measurement stores a Z-basis projection; undo its H frame.
            restored = torch.einsum("ij,ejk->eik", H_MATRIX, branch[..., 0])
            torch.testing.assert_close(
                restored, original[a, b].detach(), rtol=0, atol=3e-16
            )
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == 2 and model.clifford == 0 and model.total_t_count() == 0
