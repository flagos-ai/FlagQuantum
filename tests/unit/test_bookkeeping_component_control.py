"""Pinned bookkeeping callbacks preserve controls, wire order and ownership."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_bindings import RegisterBindingError
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from flagquantum.compiler.registers.control.component_bookkeeping import (
    bookkeeping_component_control_system,
)
from flagquantum.compiler.registers.layout.component_auto_partition import (
    _partition_program,
)
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import BlockRecord, FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Kind = Literal["allocate", "free", "split", "join", "partition", "unpartition"]
KINDS: tuple[Kind, ...] = (
    "allocate",
    "free",
    "split",
    "join",
    "partition",
    "unpartition",
)
CASES: tuple[tuple[Kind, int, bool], ...] = tuple(
    (kind, width, dirty)
    for kind in KINDS
    for width in (0, 1, 3)
    for dirty in ((False, True) if kind in ("allocate", "free") else (False,))
)


def fixture(kind: Kind, width: int, dirty: bool) -> tuple[ProgramRecord, Any]:
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Allocate, Free, Join, Partition, Split

    dtype = RegisterType("opaque", width)
    if kind == "allocate":
        return ProgramRecord((allocate_word_function("leaf", dtype, dirty),)), Allocate(
            QAny(width), dirty
        )
    if kind == "free":
        return ProgramRecord((free_word_function("leaf", dtype, dirty),)), Free(
            QAny(width), dirty
        )
    if kind == "split":
        return ProgramRecord((split_word_function("leaf", dtype),)), Split(QAny(width))
    if kind == "join":
        return ProgramRecord((join_word_function("leaf", dtype),)), Join(QAny(width))
    parts = (
        Register("first", RegisterType("opaque", width // 2)),
        Register("second", RegisterType("opaque", width - width // 2)),
    )
    source_registers = tuple(_source(Signature(parts)).signature(Signature(parts)))
    source = Partition(width, source_registers, partition=kind == "partition")
    return _partition_program(parts, "leaf", kind == "unpartition"), source


@pytest.mark.parametrize("kind,width,dirty", CASES)
@pytest.mark.parametrize("mode", ("positive", "negative", "array", "groups"))
def test_actual_bookkeeping_callback_is_unconditional_and_preserves_control_identity(
    kind: Kind, width: int, dirty: bool, mode: str
) -> None:
    from qualtran import CtrlSpec, QBit
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, source = fixture(kind, width, dirty)
    controls: tuple[Register, ...]
    if mode == "array":
        controls = (Register("added", quantum_bit_type(), (2,)),)
        spec = CtrlSpec(cvs=(0, 1))
    elif mode == "groups":
        controls = (
            Register("a", quantum_bit_type()),
            Register("b", quantum_bit_type()),
        )
        spec = CtrlSpec(qdtypes=(QBit(), QBit()), cvs=(np.array(0), np.array(1)))
    else:
        controls = (Register("added", quantum_bit_type()),)
        spec = CtrlSpec(cvs=int(mode == "positive"))
    selected, adder = bookkeeping_component_control_system(record, "leaf")
    original, callback = source.get_ctrl_system(spec)
    assert selected is record and original is source
    registers = record.functions[0].registers
    assert registers is not None
    signature = Signature((*controls, *registers))
    builder = ComponentProgramBuilder(signature, record.functions, name="caller")
    observer = _source(signature)
    values = {r.name: builder.inputs[r.name] for r in Signature(registers).lefts()}
    src_values = {r.name: observer.inputs[r.name] for r in Signature(registers).lefts()}
    before = dict(values)
    added = tuple(builder.inputs[r.name] for r in controls)
    src_added = tuple(observer.inputs[r.name] for r in controls)
    returned, data = adder(builder, added, values)
    src_returned, src_data = callback(observer.builder, src_added, src_values)
    assert returned is added and src_returned is src_added
    assert values.keys() == before.keys() == src_values.keys()
    assert all(values[k] is before[k] for k in values)
    names = tuple(r.name for r in signature.rights())
    module = builder.finalize(dict(zip(names, (*returned, *data), strict=True)))
    composite = observer.builder.finalize(
        **dict(zip(names, (*src_returned, *src_data), strict=True))
    )
    assert normalize_trace(_native_trace(module, "caller")) == normalize_trace(
        _source_trace(observer, composite)
    )
    logical_controls = sum(int(r.total_qubits()) for r in controls)
    cursor = logical_controls
    input_words: list[tuple[int, ...]] = [(i,) for i in range(logical_controls)]
    for r in Signature(registers).lefts():
        for _ in r.all_idxs():
            n = int(r.bitsize)
            input_words.append(tuple(range(cursor, cursor + n)))
            cursor += n
    pool = tuple(range(cursor, cursor + width)) if kind == "allocate" else ()
    ownership = (
        tuple((q, dirty) for word in input_words[logical_controls:] for q in word)
        if kind == "free"
        else ()
    )
    bound = bind_register_program(
        module,
        tuple(input_words),
        clean=() if dirty else pool,
        borrowed=pool if dirty else (),
        input_ownership=ownership,
        allow_workspace_outputs=True,
        entry="caller",
    )
    assert bound.circuit.instructions == ()
    assert bound.outputs[:logical_controls] == tuple(
        (i,) for i in range(logical_controls)
    )
    assert not any(set(e.wires) & set(range(logical_controls)) for e in bound.events)
    if kind in ("allocate", "free"):
        assert [(e.kind, e.wires, e.dirty) for e in bound.events] == [
            (
                kind,
                (
                    pool
                    if kind == "allocate"
                    else tuple(range(logical_controls, logical_controls + width))
                ),
                dirty,
            )
        ]
    else:
        assert bound.events == ()
    model = get_cost_value(composite, QECGatesCost())
    assert model.total_t_count() == model.measurement == model.clifford == 0
    assert get_cost_value(composite, QubitCount()) == logical_controls + width
    if width and kind == "allocate":
        with pytest.raises(ValueError, match="ownership transfer"):
            bind_register_program(
                module,
                tuple(input_words),
                clean=() if dirty else pool,
                borrowed=pool if dirty else (),
                entry="caller",
            )
    if width and kind == "free":
        with pytest.raises(ValueError, match="caller input wires"):
            bind_register_program(module, tuple(input_words), entry="caller")
    # A zero-gate callback must preserve an arbitrary complex state, including
    # borrowed work entangled with the control; lifetime flags are not resets.
    initial = torch.complex(
        torch.arange(1, (1 << bound.circuit.n_wires) + 1, dtype=torch.float64),
        torch.arange(1 << bound.circuit.n_wires, dtype=torch.float64) - 3,
    )
    if kind in ("allocate", "free") and not dirty:
        for index in range(initial.numel()):
            if index % (1 << width):
                initial[index] = 0
    initial /= initial.norm()
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit

    actual = (
        Circuit(bound.circuit.n_wires, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(bound.circuit.n_wires, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, initial, atol=2e-12, rtol=0)


def test_bookkeeping_factory_does_not_hide_computational_descendants() -> None:
    from tests.unit.test_component_control_selection import fixtures

    record, _, _, _ = fixtures("spectator", True)
    with pytest.raises(ValueError, match="finite typed"):
        bookkeeping_component_control_system(record)
    from tests.unit.test_program_owned_measurement import measured_function

    with pytest.raises(ValueError, match="computational"):
        bookkeeping_component_control_system(ProgramRecord((measured_function(),)))
    identity = ComponentProgramBuilder(
        Signature([]), record.functions, name="identity"
    ).finalize({})
    selected, _ = bookkeeping_component_control_system(identity, "identity")
    assert selected is identity


def test_empty_controls_still_invoke_bookkeeping_and_consume_once() -> None:
    from qualtran import CtrlSpec

    record, source = fixture("free", 1, False)
    _, adder = bookkeeping_component_control_system(record, "leaf")
    _, callback = source.get_ctrl_system(CtrlSpec())
    signature = Signature(record.functions[0].registers or ())
    builder = ComponentProgramBuilder(signature, record.functions, name="caller")
    observer = _source(signature)
    assert adder(builder, (), builder.inputs) == ((), ())
    returned, outputs = callback(observer.builder, (), observer.inputs)
    assert returned == () and tuple(outputs) == ()
    with pytest.raises(RegisterBindingError, match="not an available"):
        adder(builder, (), builder.inputs)
    from qualtran import BloqError

    with pytest.raises(BloqError, match="not an available"):
        callback(observer.builder, (), observer.inputs)


def test_invalid_bookkeeping_modules_fail_before_adder_construction() -> None:
    with pytest.raises(ValueError, match="unknown"):
        bookkeeping_component_control_system(fixture("allocate", 1, False)[0])
    untyped = ProgramRecord(
        (FunctionRecord("main", (BlockRecord("entry", return_values=()),)),)
    )
    with pytest.raises(ValueError, match="finite typed"):
        bookkeeping_component_control_system(untyped)
    record, _ = fixture("allocate", 1, False)
    duplicate = replace(record, functions=(*record.functions, record.functions[0]))
    with pytest.raises(ValueError, match="invalid bookkeeping"):
        bookkeeping_component_control_system(duplicate, "leaf")


def test_bookkeeping_requires_logical_explicit_returns() -> None:
    from flagquantum.core.ising import IsingBraidProgram

    with pytest.raises(ValueError, match="logical"):
        bookkeeping_component_control_system(IsingBraidProgram().to_ir())
    implicit = ProgramRecord(
        (FunctionRecord("main", (BlockRecord("entry"),), registers=()),)
    )
    with pytest.raises(ValueError, match="invalid bookkeeping"):
        bookkeeping_component_control_system(implicit)


def test_repeated_allocation_leaf_keeps_distinct_output_ownership() -> None:
    record, _ = fixture("allocate", 1, False)
    bit = quantum_bit_type()
    signature = Signature(
        (
            Register("left", RegisterType("opaque", 1), side=RegisterSide.RIGHT),
            Register("right", RegisterType("opaque", 1), side=RegisterSide.RIGHT),
        )
    )
    builder = ComponentProgramBuilder(signature, record.functions)
    left = builder.call_dict("leaf", {})["reg"]
    right = builder.call_dict("leaf", {})["reg"]
    module = builder.finalize({"left": left, "right": right})
    selected, adder = bookkeeping_component_control_system(module)
    assert selected is module
    outer_signature = Signature((Register("ctrl", bit), *tuple(signature)))
    outer = ComponentProgramBuilder(outer_signature, module.functions, name="caller")
    controls = (outer.inputs["ctrl"],)
    returned, outputs = adder(outer, controls, {})
    assert returned is controls and outputs[0] != outputs[1]
    caller = outer.finalize(
        {"ctrl": returned[0], "left": outputs[0], "right": outputs[1]}
    )
    bound = bind_register_program(
        caller, ((0,),), clean=(2, 1), allow_workspace_outputs=True, entry="caller"
    )
    assert bound.outputs == ((0,), (2,), (1,))
    assert tuple(e.wires for e in bound.events) == ((2,), (1,))
    assert bound.circuit.instructions == ()
