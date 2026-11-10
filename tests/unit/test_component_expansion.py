"""Selective expansion matches pinned graph traversal, labels and execution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import (
    NoComponentExpansionError,
    expand_component_calls,
    expand_component_calls_once,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source, _source_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _hierarchy(depth: int, repeats: int = 2) -> tuple[ProgramRecord, Any]:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import Hadamard

    q, out = (ValueRef("entry", i, "register") for i in (0, 1))
    wire, rotated = (ValueRef("entry", i, "qubit") for i in (2, 3))
    registers = (Register("q", quantum_bit_type()),)
    leaf = FunctionRecord(
        "h",
        (
            BlockRecord(
                "entry",
                (q,),
                operations=(
                    Operation("split_register", (q,), (wire,)),
                    Operation("h", (wire,), (rotated,)),
                    Operation(
                        "join_register",
                        (rotated,),
                        (out,),
                        register_type=quantum_bit_type(),
                    ),
                ),
                return_values=(out,),
            ),
        ),
        registers=registers,
    )
    definitions: tuple[FunctionRecord, ...] = (leaf,)
    previous = "h"
    source: Any = Hadamard()
    for level in range(depth):
        name = "main" if level == depth - 1 else f"level{level}"
        builder = ComponentProgramBuilder(Signature(registers), definitions, name=name)
        values = builder.inputs
        bb, source_values = BloqBuilder.from_signature(source.signature)
        for _ in range(repeats):
            values = builder.call_dict(previous, values)
            source_values = bb.add_d(source, **source_values)
        module = builder.finalize(values)
        definitions, previous, source = (
            module.functions,
            name,
            bb.finalize(**source_values),
        )
    return ProgramRecord(definitions), source


def _labelled_trace(record: ProgramRecord, labels: tuple[int, ...]) -> list[object]:
    def relabel(value: object) -> object:
        if isinstance(value, tuple):
            return tuple(relabel(part) for part in value)
        if isinstance(value, str) and value.isdecimal():
            return str(labels[int(value)])
        return value

    return [relabel(row) for row in _native_trace(record)]


def _compare(record: ProgramRecord, labels: tuple[int, ...], source: Any) -> None:
    observer = _source(Signature([]))
    assert verify_program_record(record) == ()
    assert _labelled_trace(record, labels) == _source_trace(observer, source)


@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("selection", ["all", "first", "last"])
def test_one_level_only_preserves_retained_labels_and_source_connections(
    depth: int, selection: str
) -> None:
    record, source = _hierarchy(depth)

    def predicate(label: int) -> bool:
        return selection == "all" or label == (0 if selection == "first" else 1)

    seen: list[int] = []

    def selected(label: int, operation: Operation) -> bool:
        seen.append(label)
        return predicate(label)

    expanded, labels = expand_component_calls_once(record, predicate=selected)
    expected = source.flatten_once(lambda instance: predicate(instance.i))
    _compare(expanded, labels, expected)
    assert seen == [0, 1]
    assert len(expanded.functions[0].blocks[0].operations) == (
        4 if selection == "all" else 3
    )
    for position, label in enumerate(labels):
        if label in (0, 1):
            assert (
                expanded.functions[0].blocks[0].operations[position].results
                == record.functions[0].blocks[0].operations[label].results
            )
    assert all(
        a is b
        for a, b in zip(record.functions[1:], expanded.functions[1:], strict=True)
    )


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_recursive_expansion_preserves_source_labels_tensor_and_exact_primitive_stream(
    depth: int,
) -> None:
    record, source = _hierarchy(depth)
    expanded, labels = expand_component_calls(record)
    _compare(expanded, labels, source.flatten())
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == 2**depth
    assert model.measurement == model.total_t_count() == 0
    original = bind_register_program(record, ((0,),), clean=(2,))
    bound = bind_register_program(expanded, ((0,),), clean=(2,))
    assert bound == original
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": 2**depth}
    state = torch.tensor([1, 1j, -2j, 3, 0, 0, 0, 0], dtype=torch.complex128)
    # The last physical bit is clean work, while the middle bit is a spectator.
    state = state.reshape(4, 2)
    state[:, 1] = 0
    state = state.flatten() / state.norm()
    actual = forced_branch(state, bound.circuit.instructions, ())
    np.testing.assert_allclose(
        source.tensor_contract(), source.flatten().tensor_contract(), rtol=0, atol=1e-14
    )
    torch.testing.assert_close(actual, state, rtol=0, atol=1e-14)
    assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("depth,bound", [(1, 0), (1, -1), (2, 1), (3, 2), (4, 3)])
def test_depth_limit_includes_last_no_change_probe(depth: int, bound: int) -> None:
    record, source = _hierarchy(depth)
    with pytest.raises(ValueError, match="Max recursion depth"):
        source.flatten(max_depth=bound)
    with pytest.raises(ValueError, match="Max recursion depth"):
        expand_component_calls(record, max_depth=bound)
    expanded, labels = expand_component_calls(record, max_depth=depth)
    _compare(expanded, labels, source.flatten(max_depth=depth))


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("select_none", [False, True])
def test_no_work_exception_and_recursive_identity_match_source(
    empty: bool, select_none: bool
) -> None:
    from qualtran import BloqBuilder
    from qualtran._infra.composite_bloq import DidNotFlattenAnythingError

    record, source = _hierarchy(1)
    if empty:
        native = ComponentProgramBuilder(Signature([Register("q", quantum_bit_type())]))
        record = native.finalize(native.inputs)
        bb, values = BloqBuilder.from_signature(source.signature)
        source = bb.finalize(**values)
    selected: Callable[[int, Operation], bool] | None = (
        (lambda label, op: False) if select_none else None
    )
    with pytest.raises(NoComponentExpansionError):
        expand_component_calls_once(record, predicate=selected)
    with pytest.raises(DidNotFlattenAnythingError):
        source.flatten_once(lambda instance: not select_none)
    result, labels = expand_component_calls(record, predicate=selected)
    assert result is record
    assert labels == tuple(range(len(record.functions[0].blocks[0].operations)))


@pytest.mark.parametrize("labels", [(1,), (1, 1), (-1, 2), (True, 2), (1.5, 2)])
def test_invalid_occurrence_labels_are_rejected_without_mutation(labels: Any) -> None:
    record, _ = _hierarchy(2)
    for expand in (expand_component_calls_once, expand_component_calls):
        with pytest.raises(ValueError, match="instance_ids"):
            expand(record, instance_ids=labels)
    assert verify_program_record(record) == ()


def test_predicate_failure_propagates_without_partial_result() -> None:
    record, _ = _hierarchy(3)

    def failure(label: int, operation: Operation) -> bool:
        if label == 1:
            raise RuntimeError("predicate failed")
        return True

    for expand in (expand_component_calls_once, expand_component_calls):
        with pytest.raises(RuntimeError, match="predicate failed"):
            expand(record, predicate=failure)
    assert verify_program_record(record) == ()


@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
@pytest.mark.parametrize("width", [0, 2])
def test_empty_identity_decomposition_preserves_array_permutation_and_zero_width(
    shape: tuple[int, ...], width: int
) -> None:
    from qualtran import BloqBuilder

    from flagquantum.core.ir.program.registers import RegisterType
    from tests.unit.test_program_classical_registers import identity

    registers = (Register("word", RegisterType("uint", width), shape),)
    native = ComponentProgramBuilder(
        Signature(registers), [identity("identity", registers)]
    )
    source = _source(Signature(registers))
    bb, values = BloqBuilder.from_signature(source.signature(Signature(registers)))
    source_identity = bb.finalize(**values)
    output = native.call_dict("identity", native.inputs)["word"]
    source_output = source.builder.add_d(source_identity, **source.inputs)["word"]
    output = np.asarray(output).reshape(-1)[::-1].reshape(shape)
    source_output = np.asarray(source_output).reshape(-1)[::-1].reshape(shape)
    record = native.finalize({"word": output if shape else output[()]})
    composite = source.builder.finalize(
        word=source_output if shape else source_output[()]
    )
    expanded, labels = expand_component_calls_once(record)
    assert labels == () and expanded.functions[0].blocks[0].operations == ()
    if shape == (0, 2):
        with pytest.raises(KeyError, match="word"):
            composite.flatten_once()
    else:
        _compare(expanded, labels, composite.flatten_once())
    assert verify_program_record(record) == ()


def test_invalid_unreachable_definition_is_verified_before_predicate() -> None:
    record, _ = _hierarchy(2)
    leaf = record.functions[-1]
    unreachable = replace(
        leaf, name="unreachable", blocks=(replace(leaf.blocks[0], return_values=()),)
    )
    invalid = replace(record, functions=(*record.functions, unreachable))
    seen = []

    def selected(label: int, operation: Operation) -> bool:
        seen.append(label)
        return True

    with pytest.raises(ValueError, match="invalid component graph"):
        expand_component_calls_once(invalid, predicate=selected)
    assert seen == []


def _wrapped_erasure(width: int, depth: int) -> tuple[ProgramRecord, Any]:
    from qualtran import QGF, BloqBuilder
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulMBUC

    from flagquantum.compiler.registers.finite_field.product_erasure_program import (
        product_erasure_program,
    )
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    base = product_erasure_program(FiniteFieldEncoding(2, width).register_type())
    registers = base.functions[0].registers
    assert registers is not None
    definitions = tuple(
        replace(f, name="erasure") if f.name == "main" else f for f in base.functions
    )
    previous = "erasure"
    source: Any = GF2MulMBUC(QGF(2, width)).decompose_bloq()
    for level in range(depth):
        name = "main" if level == depth - 1 else f"wrapper{level}"
        builder = ComponentProgramBuilder(Signature(registers), definitions, name=name)
        record = builder.finalize(builder.call_dict(previous, builder.inputs))
        bb, values = BloqBuilder.from_signature(source.signature)
        source = bb.finalize(**bb.add_d(source, **values))
        definitions, previous = record.functions, name
    return ProgramRecord(definitions), source


@pytest.mark.parametrize("width", [1, 2, 3, 8])
@pytest.mark.parametrize("depth", [1, 2])
def test_directional_erasure_expansion_matches_original_all_classical_connections(
    width: int, depth: int
) -> None:
    record, source = _wrapped_erasure(width, depth)
    current, labels = record, tuple(
        range(len(record.functions[0].blocks[0].operations))
    )
    for _ in range(depth):
        current, labels = expand_component_calls_once(current, instance_ids=labels)
        source = source.flatten_once()
        _compare(current, labels, source)
    expanded, labels = expand_component_calls(record)
    _compare(expanded, labels, source.flatten())
    inputs = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    assert bind_register_program(expanded, inputs) == bind_register_program(
        record, inputs
    )


@pytest.mark.parametrize("width", [1, 2, 3])
def test_expanded_erasure_all_complex_histories_and_work_zero(width: int) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from tests.unit.test_field_erasure_program import state_with_known_product

    record, _ = _wrapped_erasure(width, 2)
    expanded, _ = expand_component_calls(record)
    inputs = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    bound = bind_register_program(expanded, inputs, clean=(3 * width + 1,))
    circuit = Circuit(3 * width + 2).append(
        FieldOperation(3 * width + 2, bound.circuit.instructions)
    )
    assert circuit.to_ir().instructions == bound.circuit.instructions
    from itertools import product

    initial, logical = state_with_known_product(width)
    measured = [g.wires[0] for g in bound.circuit.instructions if g.name == "measure"]
    for history in product((0, 1), repeat=width):
        branch = forced_branch(initial, bound.circuit.instructions, history).reshape(
            (2,) * (3 * width + 2)
        )
        indices: list[int | slice] = [slice(None)] * (3 * width + 2)
        for wire, outcome in zip(measured, history, strict=True):
            indices[wire] = outcome
        indices[-1] = 0
        actual = branch[tuple(indices)].flatten()
        torch.testing.assert_close(
            actual, logical / (2 ** (width / 2)), rtol=0, atol=2e-15
        )
        assert torch.count_nonzero(branch[..., 1]) == 0
        assert torch.count_nonzero(branch) == torch.count_nonzero(actual)
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "h": width,
        "measure": width,
        "classical_xor": width,
        "cz": width**2,
    }


def test_gapped_occurrence_labels_survive_selection_and_repeated_expansion() -> None:
    record, source = _hierarchy(3)
    from qualtran import BloqBuilder

    bb, values = BloqBuilder.from_signature(source.signature)
    children = [instance.bloq for instance, _, _ in source.iter_bloqsoqs()]
    for label, child in zip((10, 30), children, strict=True):
        bb._i = label
        values = bb.add_d(child, **values)
    source = bb.finalize(**values)
    expanded, labels = expand_component_calls_once(
        record, instance_ids=(10, 30), predicate=lambda label, op: label == 30
    )
    source = source.flatten_once(lambda instance: instance.i == 30)
    _compare(expanded, labels, source)
    assert 10 in labels and min(label for label in labels if label != 10) == 31
    expanded, labels = expand_component_calls(expanded, instance_ids=labels)
    _compare(expanded, labels, source.flatten())


@pytest.mark.parametrize("count", [2, 3])
def test_nonempty_array_body_reorders_outputs_and_preserves_atomic_mixed_calls(
    count: int,
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard

    leaf = _hierarchy(1)[0].functions[-1]
    signature = Signature([Register("q", quantum_bit_type(), (count,))])
    builder = ComponentProgramBuilder(signature, (leaf,), name="parallel")
    source = _source(signature)
    outputs = [
        builder.call_dict("h", {"q": q})["q"]
        for q in np.asarray(builder.inputs["q"]).flat
    ]
    source_outputs = [
        source.builder.add(Hadamard(), q=q) for q in np.asarray(source.inputs["q"]).flat
    ]
    child = builder.finalize({"q": np.array(outputs[::-1])})
    source_child = source.builder.finalize(q=np.array(source_outputs[::-1]))
    root_signature = Signature([*signature, Register("spectator", quantum_bit_type())])
    native = ComponentProgramBuilder(root_signature, child.functions)
    root = _source(root_signature)
    word = native.call_dict("parallel", {"q": native.inputs["q"]})["q"]
    source_word = root.builder.add_d(source_child, q=root.inputs["q"])["q"]
    spectator = native.call_dict("h", {"q": native.inputs["spectator"]})["q"]
    source_spectator = root.builder.add(Hadamard(), q=root.inputs["spectator"])
    record = native.finalize({"q": word, "spectator": spectator})
    composite = root.builder.finalize(q=source_word, spectator=source_spectator)
    expanded, labels = expand_component_calls_once(record)
    _compare(expanded, labels, composite.flatten_once())
    inputs = tuple((i,) for i in range(count + 1))
    assert bind_register_program(expanded, inputs) == bind_register_program(
        record, inputs
    )
    assert labels[-1] == 1


def test_custom_entry_block_and_canonical_payload_survive_expansion() -> None:
    record, _ = _hierarchy(2)
    main = record.functions[0]
    renamed = replace(
        main,
        name="selected",
        entry="body",
        blocks=(replace(main.blocks[0], name="body"),),
    )
    record = replace(
        record,
        functions=(renamed, *record.functions[1:]),
        canonical_payload={"witness": "retained"},
    )
    expanded, labels = expand_component_calls(record, entry="selected")
    assert expanded.canonical_payload is record.canonical_payload
    assert expanded.functions[0].entry == "body"
    assert expanded.functions[0].blocks[0].name == "body"
    assert expanded.functions[0].name == "selected"
    assert len(labels) == 4
    assert verify_program_record(expanded) == ()


def test_composite_allocation_wrapper_retains_source_priority_before_and_after_expansion() -> (
    None
):
    from qualtran import QBit

    from flagquantum.core.ir.program.registers import RegisterSide

    result_signature = Signature(
        [Register("reg", quantum_bit_type(), side=RegisterSide.RIGHT)]
    )
    child = ComponentProgramBuilder(result_signature, name="allocate_wrapper")
    result = child.allocate(dtype=quantum_bit_type())
    definitions = child.finalize({"reg": result}).functions
    source_child = _source(result_signature)
    source_allocation = source_child.builder.allocate(dtype=QBit())
    composite_child = source_child.builder.finalize(reg=source_allocation)
    signature = Signature([Register("q", quantum_bit_type())])
    builder = ComponentProgramBuilder(signature, definitions)
    source = _source(signature)
    direct = builder.allocate(dtype=quantum_bit_type())
    wrapped = builder.call_dict("allocate_wrapper", {})["reg"]
    assert isinstance(wrapped, type(direct))
    builder.free(direct)
    builder.free(wrapped)
    record = builder.finalize(builder.inputs)
    source_direct = source.builder.allocate(dtype=QBit())
    source_wrapped = source.builder.add_d(composite_child)["reg"]
    source.builder.free(source_direct)
    source.builder.free(source_wrapped)
    composite = source.builder.finalize(**source.inputs)
    _compare(record, (0, 1, 2, 3), composite)
    expanded, labels = expand_component_calls_once(record)
    _compare(expanded, labels, composite.flatten_once())
    assert verify_program_record(expanded) == ()


def test_composite_free_wrapper_keeps_current_layer_priority() -> None:
    from qualtran import QBit

    from flagquantum.core.ir.program.registers import RegisterSide

    signature = Signature([Register("reg", quantum_bit_type(), side=RegisterSide.LEFT)])
    builder = ComponentProgramBuilder(signature, name="free_wrapper")
    word = builder.inputs["reg"]
    from flagquantum.core.ir.program.register_bindings import RegisterBinding

    assert isinstance(word, RegisterBinding)
    builder.free(word)
    definitions = builder.finalize({}).functions
    source_child = _source(signature)
    source_child.builder.free(source_child.inputs["reg"])
    child = source_child.builder.finalize()
    root_signature = Signature([Register("q", quantum_bit_type())])
    root = ComponentProgramBuilder(root_signature, definitions)
    source = _source(root_signature)
    first = root.allocate(dtype=quantum_bit_type())
    second = root.allocate(dtype=quantum_bit_type())
    root.call_dict("free_wrapper", {"reg": first})
    root.free(second)
    record = root.finalize(root.inputs)
    source_first = source.builder.allocate(dtype=QBit())
    source_second = source.builder.allocate(dtype=QBit())
    source.builder.add_d(child, reg=source_first)
    source.builder.free(source_second)
    composite = source.builder.finalize(**source.inputs)
    _compare(record, (0, 1, 2, 3), composite)
    expanded, labels = expand_component_calls_once(record)
    _compare(expanded, labels, composite.flatten_once())
