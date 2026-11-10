"""Pinned Cast callbacks, shaped wire views and mixed lifetime ownership."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.control.component_bookkeeping import (
    bookkeeping_component_control_system,
)
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (0, 1, 3))
@pytest.mark.parametrize("shape", ((), (2,), (1, 2), (0, 2)))
@pytest.mark.parametrize("reverse", (False, True))
def test_shaped_cast_actual_callback_and_physical_order(
    width: int, shape: tuple[int, ...], reverse: bool
) -> None:
    from qualtran import CtrlSpec, QAny, QUInt
    from qualtran.bloqs.bookkeeping import Cast
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    a, b = RegisterType("opaque", width), RegisterType("uint", width)
    x, y = QAny(width), QUInt(width)
    if reverse:
        a, b, x, y = b, a, y, x
    leaf = cast_word_function("cast", a, b, shape)
    assert verify_program_record(ProgramRecord((leaf,))) == ()
    assert leaf.registers is not None
    source = Cast(x, y, shape)
    signature = Signature(
        (Register("ctrl", RegisterType("bit", 1), (2,)), *leaf.registers)
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    observer = _source(signature)
    _, add = bookkeeping_component_control_system(ProgramRecord((leaf,)), "cast")
    original, callback = source.get_ctrl_system(CtrlSpec(cvs=(0, 1)))
    assert original is source
    controls = (builder.inputs["ctrl"],)
    source_controls = (observer.inputs["ctrl"],)
    returned, outputs = add(builder, controls, {"reg": builder.inputs["reg"]})
    src_returned, src_outputs = callback(
        observer.builder, source_controls, {"reg": observer.inputs["reg"]}
    )
    assert returned is controls and src_returned is source_controls
    record = builder.finalize({"ctrl": returned[0], "reg": outputs[0]})
    composite = observer.builder.finalize(ctrl=src_returned[0], reg=src_outputs[0])
    assert normalize_trace(_native_trace(record)) == normalize_trace(
        _source_trace(observer, composite)
    )
    count = int(np.prod(shape))
    words = (
        (0,),
        (1,),
        *(tuple(range(2 + i * width, 2 + (i + 1) * width)) for i in range(count)),
    )
    bound = bind_register_program(record, words)
    assert bound.outputs == words and bound.events == ()
    assert bound.circuit.instructions == ()
    # Actual upstream physical callback preserves the original array object.
    qubits = np.arange(count * width).reshape((*shape, width))
    operation, views = source.as_cirq_op(None, reg=qubits)
    assert operation is None and views["reg"] is qubits
    cost = get_cost_value(composite, QECGatesCost())
    assert cost.total_t_count() == cost.measurement == cost.clifford == 0


@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_mixed_directional_composition_transfers_and_consumes_owned_words(
    dirty: bool, reverse: bool
) -> None:
    from qualtran import CtrlSpec, QAny, QBit, QUInt
    from qualtran.bloqs.bookkeeping import Allocate, Cast, Free
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    opaque, uint = RegisterType("opaque", 2), RegisterType("uint", 2)
    a, b = (uint, opaque) if reverse else (opaque, uint)
    x, y = (QUInt(2), QAny(2)) if reverse else (QAny(2), QUInt(2))
    functions = (
        cast_word_function("cast", a, b, (2,)),
        allocate_word_function("allocate", RegisterType("bit", 1), dirty),
        free_word_function("free", RegisterType("bit", 1), dirty),
    )
    signature = Signature(
        (
            Register("ctrl", RegisterType("bit", 1)),
            Register("consume", RegisterType("bit", 1), side=RegisterSide.LEFT),
            Register("data", a, (2,), RegisterSide.LEFT),
            Register("spectator", RegisterType("bit", 1)),
            Register("data", b, (2,), RegisterSide.RIGHT),
            Register("introduced", RegisterType("bit", 1), side=RegisterSide.RIGHT),
        )
    )
    builder = ComponentProgramBuilder(signature, functions)
    observer = _source(signature)
    native_values: dict[str, Any] = dict(builder.inputs)
    source_values: dict[str, Any] = dict(observer.inputs)
    for name, source, input_name, output_name in (
        ("cast", Cast(x, y, (2,)), "data", "data"),
        (
            "allocate",
            Allocate(dtype=QBit(), dirty=dirty),
            None,
            "introduced",
        ),
        (
            "free",
            Free(dtype=QBit(), dirty=dirty),
            "consume",
            None,
        ),
    ):
        _, add = bookkeeping_component_control_system(ProgramRecord(functions), name)
        _, callback = source.get_ctrl_system(CtrlSpec(cvs=0))
        inputs = {} if input_name is None else {"reg": native_values.pop(input_name)}
        src_inputs = (
            {} if input_name is None else {"reg": source_values.pop(input_name)}
        )
        _, outputs = add(builder, (native_values["ctrl"],), inputs)
        _, src_outputs = callback(
            observer.builder, (source_values["ctrl"],), src_inputs
        )
        if output_name is not None:
            native_values[output_name] = outputs[0]
            source_values[output_name] = src_outputs[0]
    record = builder.finalize(native_values)
    composite = observer.builder.finalize(**source_values)
    assert normalize_trace(_native_trace(record)) == normalize_trace(
        _source_trace(observer, composite)
    )
    args = ((0,), (1,), (2, 3), (4, 5), (6,))
    kwargs: dict[str, Any] = dict(
        clean=() if dirty else (7,),
        borrowed=(7,) if dirty else (),
        input_ownership=((1, dirty),),
        allow_workspace_outputs=True,
    )
    bound = bind_register_program(record, args, **kwargs)
    assert bound.outputs == ((0,), (6,), (2, 3), (4, 5), (1,))
    assert [(e.kind, e.wires, e.dirty) for e in bound.events] == [
        ("free", (1,), dirty),
        ("allocate", (1,), dirty),
    ]
    assert bound.circuit.instructions == ()
    with pytest.raises(ValueError, match="ownership"):
        bind_register_program(record, args, **{**kwargs, "input_ownership": ()})
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit

    initial = torch.complex(
        torch.arange(1, 257, dtype=torch.float64),
        torch.arange(256, dtype=torch.float64) - 7,
    )
    if not dirty:
        for i in range(256):
            if i & 65:  # Consumed clean input and newly allocated clean output.
                initial[i] = 0
    initial /= initial.norm()
    actual = (
        Circuit(8, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(8, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, initial, atol=2e-12, rtol=0)
    cost = get_cost_value(composite, QECGatesCost())
    assert cost.total_t_count() == cost.measurement == cost.clifford == 0
    assert get_cost_value(composite, QubitCount()) == 7
    assert [
        type(instance.bloq).__name__
        for instance, _, _ in composite.iter_bloqsoqs()
        if type(instance.bloq).__name__ in {"Allocate", "Free"}
    ] == ["Free", "Allocate"]


def test_original_shaped_adjoint_drops_shape_without_repairing_reference() -> None:
    from qualtran import QAny, QUInt
    from qualtran.bloqs.bookkeeping import Cast

    source = Cast(QAny(2), QUInt(2), (2,))
    assert source.shape == (2,) and source.adjoint().shape == ()
    observer = _source(Signature((Register("reg", RegisterType("opaque", 2), (2,)),)))
    output = observer.builder.add(source, reg=observer.inputs["reg"])
    with pytest.raises(AssertionError):
        observer.builder.add(source.adjoint(), reg=output)
    forward = cast_word_function(
        "forward", RegisterType("opaque", 2), RegisterType("uint", 2), (2,)
    )
    backward = cast_word_function(
        "backward", RegisterType("uint", 2), RegisterType("opaque", 2), (2,)
    )
    assert forward.registers is not None and backward.registers is not None
    assert tuple(Signature(forward.registers).adjoint().lefts()) == tuple(
        Signature(backward.registers).lefts()
    )
    assert bind_register_program(
        ProgramRecord((backward,)), ((3, 1), (2, 0)), entry="backward"
    ).outputs == ((3, 1), (2, 0))


@pytest.mark.parametrize("shape", ((), (2,), (1, 2), (0, 2)))
def test_explicit_shaped_reverse_roundtrip_keeps_original_call_graph(
    shape: tuple[int, ...],
) -> None:
    from qualtran import QAny, QUInt
    from qualtran.bloqs.bookkeeping import Cast

    opaque, uint = RegisterType("opaque", 2), RegisterType("uint", 2)
    signature = Signature((Register("reg", opaque, shape),))
    forward = cast_word_function("forward", opaque, uint, shape)
    backward = cast_word_function("backward", uint, opaque, shape)
    builder = ComponentProgramBuilder(signature, (forward, backward))
    observer = _source(signature)
    data = builder.call_dict("forward", {"reg": builder.inputs["reg"]})
    data = builder.call_dict("backward", data)
    record = builder.finalize(data)
    src_data = observer.builder.add(
        Cast(QAny(2), QUInt(2), shape), reg=observer.inputs["reg"]
    )
    src_data = observer.builder.add(Cast(QUInt(2), QAny(2), shape), reg=src_data)
    composite = observer.builder.finalize(reg=src_data)
    assert normalize_trace(_native_trace(record)) == normalize_trace(
        _source_trace(observer, composite)
    )
    words = tuple((2 * i + 1, 2 * i) for i in range(int(np.prod(shape))))
    if not words:
        with pytest.raises(ValueError, match="at least one physical wire"):
            bind_register_program(record, words)
        return
    bound = bind_register_program(record, words)
    assert bound.outputs == words and bound.circuit.instructions == ()


@pytest.mark.parametrize(
    "source,target,shape",
    (
        (RegisterType("uint", 1), RegisterType("opaque", 2), ()),
        (RegisterType("int", 2), RegisterType("opaque", 2), ()),
        (RegisterType("opaque", 2), RegisterType("opaque", 2), (-1,)),
        (RegisterType("opaque", 2), RegisterType("opaque", 2), (True,)),
    ),
)
def test_cast_rejects_incompatible_or_unadmitted_ports(
    source: RegisterType, target: RegisterType, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        cast_word_function("cast", source, target, shape)
