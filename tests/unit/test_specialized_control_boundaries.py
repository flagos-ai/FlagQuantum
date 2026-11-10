"""Specialized dispatch retains original wrapper and shaped-callback boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.control.component_control_selection import (
    select_component_control_system,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_auto_partition import source_partition
from tests.unit.test_component_control import add_h, add_phase
from tests.unit.test_component_control_selection import fixtures, native_fallback
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("layers", (1, 2))
def test_original_unwrap_result_and_wrapped_callback_are_distinct_entries(
    layers: int,
) -> None:
    from qualtran import CtrlSpec, QAny
    from qualtran.bloqs.mcmt.specialized_ctrl import _get_ctrl_system_1bit_cv

    record, source, native, originals = fixtures("spectator", True)
    module, _ = native[1]
    layout = AutoPartition(
        (("control", (("ctrl", 1),)), ("data", (("q", 1), ("spectator", 1))))
    )
    wrapper = source_partition(originals[1], layout)
    module = auto_partition_component_program(module, layout, name="packed0")
    if layers == 2:
        layout = AutoPartition(
            (("control", (("control", 1),)), ("data", (("data", 2),)))
        )
        wrapper = source_partition(wrapper, layout)
        module = auto_partition_component_program(
            module, layout, entry="packed0", name="packed1"
        )
    entry = f"packed{layers-1}"
    controls = (Register("added", RegisterType("opaque", 1)),)
    patterns = (ControlRegister(1, (1,)),)
    selected, adder = select_component_control_system(
        controls,
        patterns,
        lambda value: (module, "control"),
        native_fallback(record, {"h": add_h, "p": add_phase}),
        entry=entry,
    )
    original, callback = _get_ctrl_system_1bit_cv(
        source,
        CtrlSpec(qdtypes=QAny(1)),
        current_ctrl_bit=None,
        get_ctrl_bloq_and_ctrl_reg_name=lambda value: (wrapper, "control"),
    )
    assert original == originals[1]
    assert selected.functions[0] is native[1][0].functions[0]
    signature = Signature((controls[0], Register("data", RegisterType("opaque", 2))))
    builder = ComponentProgramBuilder(signature, selected.functions, name="caller")
    observer = _source(signature)
    values = {"data": builder.inputs["data"]}
    source_values = {"data": observer.inputs["data"]}
    added, data = adder(builder, (builder.inputs["added"],), values)
    src_added, src_data = callback(
        observer.builder, (observer.inputs["added"],), source_values
    )
    caller = builder.finalize({"added": added[0], "data": data[0]})
    composite = observer.builder.finalize(
        added=tuple(src_added)[0], data=tuple(src_data)[0]
    )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(observer, composite)
    )
    assert tuple(values) == tuple(source_values)
    gates = bind_register_program(
        caller, ((0,), (1, 2)), clean=(3,), entry="caller"
    ).circuit
    assert (
        gates
        == bind_register_program(native[1][0], ((0,), (1,), (2,)), clean=(3,)).circuit
    )


@pytest.mark.parametrize("value", (0, 1))
def test_shaped_single_control_passes_original_collection_without_scalar_coercion(
    value: int,
) -> None:
    from qualtran import CtrlSpec

    record, source, native, _ = fixtures("spectator", True)
    controls = (Register("added", RegisterType("bit", 1), (1,)),)
    module, adder = select_component_control_system(
        controls,
        (ControlRegister(1, (value,)),),
        native.get,
        native_fallback(record, {"h": add_h, "p": add_phase}),
    )
    _, callback = source.get_ctrl_system(CtrlSpec(cvs=[value]))
    signature = Signature((*controls, *(record.functions[0].registers or ())))
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {
        r.name: builder.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    source_values = {
        r.name: observer.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    with pytest.raises(AssertionError):
        adder(builder, (builder.inputs["added"],), values)
    with pytest.raises(AssertionError):
        callback(observer.builder, (observer.inputs["added"],), source_values)
    assert tuple(values) == tuple(source_values)
    name = native[value][1]
    assert (
        np.asarray(values[name]).shape == np.asarray(source_values[name]).shape == (1,)
    )


def test_custom_entry_and_nonleading_control_preserve_selected_port_order() -> None:
    record, _, native, _ = fixtures("spectator", True)
    selected, name = native[1]
    root = selected.functions[0]
    assert root.registers is not None
    ctrl, q, spare = root.registers
    specialized = ComponentProgramBuilder(
        Signature((q, ctrl, spare)), selected.functions, name="special"
    )
    selected = specialized.finalize(specialized.call_dict("main", specialized.inputs))
    controls = (Register("added", RegisterType("bit", 1)),)
    module, adder = select_component_control_system(
        controls,
        (ControlRegister(1, (1,)),),
        lambda value: (selected, name),
        native_fallback(record, {"h": add_h, "p": add_phase}),
        entry="special",
    )
    signature = Signature((*controls, *(record.functions[0].registers or ())))
    builder = ComponentProgramBuilder(signature, module.functions, name="caller")
    values = {
        r.name: builder.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    added, data = adder(builder, (builder.inputs["added"],), values)
    caller = builder.finalize(
        dict(zip((r.name for r in signature), (*added, *data), strict=True))
    )
    assert (
        bind_register_program(
            caller, ((0,), (1,), (2,)), clean=(3,), entry="caller"
        ).circuit
        == bind_register_program(native[1][0], ((0,), (1,), (2,)), clean=(3,)).circuit
    )
