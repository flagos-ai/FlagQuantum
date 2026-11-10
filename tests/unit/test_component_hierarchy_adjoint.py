"""Hierarchical adjoints reuse exact leaves and retain source call structure."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_add_calls import executable_add
from tests.unit.test_component_expansion import _hierarchy
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_adjoint(composite: Any) -> Any:
    """Wrap a real stored decomposition in the original default Adjoint protocol."""
    from qualtran import Adjoint, Bloq

    def decompose(component: object) -> Any:
        return composite

    # The optional pinned source is loaded dynamically, so its base is untyped
    # to native mypy. This real source component stores an executable body.
    stored_type = type(
        "StoredComponent",
        (Bloq,),
        {"signature": composite.signature, "decompose_bloq": decompose},
    )
    return Adjoint(stored_type())


def paired_components(
    record: ProgramRecord, source: Any, terminals: set[str]
) -> dict[str, Any]:
    """Observe each real source graph once without duplicating graph semantics."""
    functions = {f.name: f for f in record.functions}
    pairs = {"main": source}
    pending = ["main"]
    observer = _source(Signature([]))
    while pending:
        name = pending.pop()
        if name in terminals:
            continue
        composite = pairs[name]
        assert tuple(
            observer.signature(Signature(functions[name].registers or ()))
        ) == tuple(composite.signature)
        assert _native_trace(record, name) == _source_trace(observer, composite)
        block = functions[name].blocks[0]
        for instance, _, _ in composite.iter_bloqsoqs():
            target = block.operations[instance.i].callee
            assert target is not None
            if target in pairs:
                assert pairs[target] == instance.bloq
            else:
                pairs[target] = instance.bloq
                pending.append(target)
    return pairs


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
@pytest.mark.parametrize("repeats", [1, 2])
def test_stored_hierarchies_match_source_shared_adjoints_and_complex_states(
    depth: int, repeats: int
) -> None:
    record, source = _hierarchy(depth, repeats)
    inverse = adjoint_component_hierarchy(record, {"h": "h"})
    expected = source_adjoint(source).decompose_bloq()
    paired_components(inverse, expected, {"h"})
    assert len(inverse.functions) == len(record.functions) + depth - 1
    assert all(
        a is b
        for a, b in zip(
            record.functions[1:],
            inverse.functions[1 : len(record.functions)],
            strict=True,
        )
    )
    assert verify_program_record(record) == verify_program_record(inverse) == ()
    bound = bind_register_program(inverse, ((0,),))
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": repeats**depth}
    initial = torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex128) / 30**0.5
    matrix = expected.tensor_contract()
    expected_state = torch.as_tensor(
        np.einsum("ij,j->i", matrix, initial.numpy(), optimize=False)
    )
    torch.testing.assert_close(
        forced_branch(initial, bound.circuit.instructions, ()), expected_state
    )
    flattened, _ = expand_component_calls(inverse)
    assert bind_register_program(flattened, ((0,),)).circuit == bound.circuit


def wrapped_add(depth: int) -> tuple[ProgramRecord, Any]:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add

    signature = Signature(
        [
            Register("a", RegisterType("uint", 2)),
            Register("b", RegisterType("uint", 3)),
            Register("spectator", RegisterType("uint", 1)),
        ]
    )
    definitions = (*executable_add(), *executable_add(adjoint=True))
    source: Any = Add(QUInt(2), QUInt(3))
    previous = "add"
    for level in range(depth):
        name = "main" if level == depth - 1 else f"layer{level}"
        native = ComponentProgramBuilder(signature, definitions, name=name)
        upstream = _source(signature)
        outputs = {name: native.inputs[name] for name in ("a", "b")}
        expected_outputs = {name: upstream.inputs[name] for name in ("a", "b")}
        if level:
            outputs["spectator"] = native.inputs["spectator"]
            expected_outputs["spectator"] = upstream.inputs["spectator"]
        for _ in range(2 if level == 0 else 1):
            outputs = native.call_dict(previous, outputs)
            expected_outputs = upstream.builder.add_d(source, **expected_outputs)
        module = native.finalize(
            {
                **outputs,
                "spectator": outputs.get("spectator", native.inputs["spectator"]),
            }
        )
        source = upstream.builder.finalize(
            **{
                **expected_outputs,
                "spectator": expected_outputs.get(
                    "spectator", upstream.inputs["spectator"]
                ),
            }
        )
        definitions, previous = module.functions, name
    return ProgramRecord(definitions), source


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("expand", [False, True])
def test_generated_add_adjoints_keep_exact_feedback_all_branches_and_work_zero(
    depth: int, expand: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, source = wrapped_add(depth)
    inverse = adjoint_component_hierarchy(record, {"add": "add_adjoint"})
    expected = source_adjoint(source).decompose_bloq()
    paired_components(inverse, expected, {"add_adjoint"})
    if expand:
        inverse, _ = expand_component_calls(inverse)
    bound = bind_register_program(inverse, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    measurements = tuple(g for g in bound.circuit.instructions if g.name == "measure")
    gates = tuple(
        g
        for offset in (0, 2)
        for g in reference_arithmetic_gates(
            Add(QUInt(2), QUInt(3)).adjoint(),
            {"a": (0, 1), "b": (2, 3, 4)},
            (6, 7),
            {
                g.wires[0]: g.metadata["classical_bit"]
                for g in measurements[offset : offset + 2]
            },
        )
    )
    assert_source(bound.circuit.instructions, gates)
    model = get_cost_value(source_adjoint(source), QECGatesCost())
    assert model.total_t_count() == 16 and model.measurement == 4
    assert get_cost_value(source_adjoint(source), QubitCount()) == 8
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "cx": 40,
        "h": 12,
        "t": 8,
        "tdg": 8,
        "s": 4,
        "measure": 4,
        "cz": 4,
        "reset": 4,
    }
    initial = torch.zeros(256, dtype=torch.complex128)
    amplitudes = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    amplitudes /= amplitudes.norm()
    initial[::4] = amplitudes
    expected_state = torch.zeros_like(initial)
    for prefix in range(64):
        a, b, spectator = prefix >> 4, (prefix >> 1) & 7, prefix & 1
        target = (a << 4) | (((b - 2 * a) & 7) << 1) | spectator
        expected_state[target << 2] = initial[prefix << 2]
    for history in product((0, 1), repeat=4):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, expected_state / 4, rtol=0, atol=1e-12)
        assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0
    # A second derivation shares the original Add as its specialized leaf.
    forward = adjoint_component_hierarchy(
        adjoint_component_hierarchy(record, {"add": "add_adjoint"}),
        {"add_adjoint": "add"},
    )
    restored = bind_register_program(forward, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    for history in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(expected_state, restored.circuit.instructions, history),
            initial / 4,
            rtol=0,
            atol=1e-12,
        )


def test_specialized_composite_mapping_stops_generation_and_keeps_forward_definition() -> (
    None
):
    record, source = _hierarchy(4)
    inverse = adjoint_component_hierarchy(record, {"h": "h", "level0": "level0"})
    paired_components(inverse, source.adjoint(), {"h", "level0"})
    assert len(inverse.functions) == len(record.functions) + 2
    assert inverse.functions[3] is record.functions[3]


def test_generated_names_skip_existing_definitions_and_unused_mapping_is_ignored() -> (
    None
):
    record, source = _hierarchy(3)
    h = record.functions[-1]
    sentinel = replace(h, name="__fq_adjoint_0")
    record = replace(record, functions=(*record.functions, sentinel))
    inverse = adjoint_component_hierarchy(record, {"h": "h", "unused": "absent"})
    paired_components(inverse, source.adjoint(), {"h"})
    assert next(f for f in inverse.functions if f.name == sentinel.name) is sentinel
    assert {f.name for f in inverse.functions[len(record.functions) :]} == {
        "__fq_adjoint_1",
        "__fq_adjoint_2",
    }


@pytest.mark.parametrize(
    "mapping", [{}, {"h": "missing"}, {"h": "main"}, {"main": "h", "h": "h"}]
)
def test_missing_specialized_or_root_mapping_fails_without_partial_output(
    mapping: dict[str, str],
) -> None:
    record, _ = _hierarchy(3)
    original = record.functions
    with pytest.raises(ValueError):
        adjoint_component_hierarchy(record, mapping)
    assert record.functions is original and verify_program_record(record) == ()


def test_empty_identity_nested_body_has_a_real_empty_adjoint_definition() -> None:
    from tests.unit.test_program_classical_registers import identity

    registers = (Register("q", RegisterType("bit", 1)),)
    builder = ComponentProgramBuilder(
        Signature(registers), (identity("identity", registers),)
    )
    record = builder.finalize(builder.call_dict("identity", builder.inputs))
    inverse = adjoint_component_hierarchy(record, {})
    assert len(inverse.functions) == 3
    assert inverse.functions[-1].blocks[0].operations == ()
    assert bind_register_program(inverse, ((0,),)).circuit.instructions == ()


def test_raw_composite_wrapper_retains_original_source_decomposition_refusal() -> None:
    from qualtran import Adjoint

    _, source = _hierarchy(2)
    with pytest.raises(ValueError, match="ill-defined"):
        Adjoint(source).decompose_bloq()
    assert source_adjoint(source).decompose_bloq() == source.adjoint()


@pytest.mark.parametrize("allocate", [False, True])
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("width", [0, 2])
def test_generated_directional_wrappers_use_actual_clean_or_borrowed_duals(
    allocate: bool, dirty: bool, width: int
) -> None:
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Allocate, Free

    from flagquantum.compiler.registers.layout.component_workspace import (
        allocate_word_function,
        free_word_function,
    )
    from flagquantum.core.ir.program.registers import RegisterSide

    dtype = RegisterType("opaque", width)
    signature = Signature(
        [
            Register(
                "reg", dtype, side=RegisterSide.RIGHT if allocate else RegisterSide.LEFT
            )
        ]
    )
    definitions: tuple[FunctionRecord, ...] = (
        allocate_word_function("allocate", dtype, dirty),
        free_word_function("free", dtype, dirty),
    )
    previous = "allocate" if allocate else "free"
    source: Any = (
        Allocate(QAny(width), dirty=dirty)
        if allocate
        else Free(QAny(width), dirty=dirty)
    )
    for name in ("inner", "main"):
        native = ComponentProgramBuilder(signature, definitions, name=name)
        upstream = _source(signature)
        outputs = native.call_dict(previous, native.inputs)
        expected_outputs = upstream.builder.add_d(source, **upstream.inputs)
        record = native.finalize(outputs)
        source = upstream.builder.finalize(**expected_outputs)
        definitions, previous = record.functions, name
    inverse = adjoint_component_hierarchy(
        record, {"allocate": "free", "free": "allocate"}
    )
    paired_components(inverse, source.adjoint(), {"allocate", "free"})
    assert inverse.functions[0].registers == tuple(signature.adjoint())
    assert len(inverse.functions) == len(record.functions) + 1


def test_custom_hierarchical_entry_preserves_names_and_clears_forward_cache() -> None:
    record, _ = _hierarchy(3)
    function = record.functions[0]
    function = replace(
        function,
        name="custom",
        entry="body",
        blocks=(replace(function.blocks[0], name="body"),),
    )
    custom = replace(
        record,
        functions=(function, *record.functions[1:]),
        canonical_payload={"forward": True},
    )
    inverse = adjoint_component_hierarchy(custom, {"h": "h"}, "custom")
    expected = adjoint_component_hierarchy(record, {"h": "h"})
    assert inverse.functions[0].name == "custom"
    assert inverse.functions[0].entry == inverse.functions[0].blocks[0].name == "body"
    assert inverse.canonical_payload is None
    assert (
        bind_register_program(inverse, ((0,),), entry="custom").circuit
        == bind_register_program(expected, ((0,),)).circuit
    )


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_generated_array_wrapper_retains_rebindings_and_empty_metadata(
    width: int, shape: tuple[int, ...]
) -> None:
    signature = Signature([Register("word", RegisterType("uint", width), shape)])
    native = ComponentProgramBuilder(signature, name="reorder")
    upstream = _source(signature)
    returned = []
    for builder in (native, upstream):
        outputs = np.empty(shape, dtype=object)
        for index in signature.get_left("word").all_idxs():
            word = np.asarray(builder.inputs["word"])[index]
            outputs[index] = builder.join(
                builder.split(word)[::-1], signature.get_left("word").dtype
            )
        returned.append(outputs if shape else outputs[()])
    child = native.finalize({"word": returned[0]})
    source = upstream.builder.finalize(word=returned[1])
    outer = ComponentProgramBuilder(signature, child.functions)
    record = outer.finalize(outer.call_dict("reorder", outer.inputs))
    source_outer = _source(signature)
    outputs = source_outer.builder.add_d(source, **source_outer.inputs)
    expected = source_outer.builder.finalize(**outputs)
    definitions = child.functions[1:]
    mapping = {
        definitions[i].name: definitions[i ^ 1].name for i in range(len(definitions))
    }
    inverse = adjoint_component_hierarchy(record, mapping)
    if shape == (0, 2):
        with pytest.raises(ValueError, match="word"):
            source_adjoint(expected).decompose_bloq()
        assert inverse.functions[0].registers == record.functions[0].registers
    else:
        paired_components(
            inverse, source_adjoint(expected).decompose_bloq(), set(mapping.values())
        )
    assert len(inverse.functions) == len(record.functions) + 1
