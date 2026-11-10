# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 specialized_ctrl.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Bind packed And controls to an existing positive single-control Program."""

from dataclasses import replace

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import ControlAdder
from ..component_graph import _call_entry, _namespace_helpers
from .component_predicate import control_predicate_program
from .control_predicate import ControlRegister


def _array_and_program(
    values: tuple[int, ...], name: str, erase: bool
) -> ProgramRecord:
    """Adapt the shared predicate's opaque junk to the original bit-array port."""
    ctrl = Register("ctrl", quantum_bit_type(), (len(values),))
    predicate = control_predicate_program(
        (ctrl,), (ControlRegister(1, values),), name="predicate", uncompute=erase
    )
    side = RegisterSide.LEFT if erase else RegisterSide.RIGHT
    junk = (
        (Register("junk", quantum_bit_type(), (len(values) - 2,), side=side),)
        if len(values) > 2
        else ()
    )
    builder = ComponentProgramBuilder(
        Signature((ctrl, *junk, Register("target", quantum_bit_type(), side=side))),
        predicate.functions,
        name=name,
    )
    inputs = dict(builder.inputs)
    if erase and junk:
        inputs["junk"] = builder.join(inputs["junk"])
    outputs = builder.call_dict("predicate", inputs)
    if not erase and junk:
        word = outputs["junk"]
        assert isinstance(word, RegisterBinding)
        outputs["junk"] = builder.split(word)
    return _namespace_helpers(builder.finalize(outputs), name)


def multi_controlled_component_program(
    record: ProgramRecord,
    control_name: str,
    values: tuple[int, ...],
    entry: str = "main",
) -> ProgramRecord:
    """Pack at least two activations around a stored positive-control body.

    The named original control is a scalar THRU one-qubit port, preserved by
    the child. It may appear anywhere in the child signature. Admitted one-qubit
    dtypes use the existing call compatibility rules.
    New control-array order is the activation order. Predicate work starts clean,
    stays live through the child and is measurement-erased before release.
    Child private work retains its existing lifetime. Invalid entries, control
    ports and nonbinary activations raise ValueError. Inputs remain immutable.
    """
    function = _call_entry(record, entry)
    assert function.registers is not None
    original_control = next(
        (r for r in function.registers if r.name == control_name), None
    )
    if (
        original_control is None
        or original_control.shape
        or original_control.side != RegisterSide.THRU
        or executable_register_width(original_control.dtype) != 1
    ):
        raise ValueError("specialized control requires a named scalar THRU qubit")
    data_registers = tuple(r for r in function.registers if r.name != control_name)
    if len(values) < 2 or any(
        type(value) is not int or value not in (0, 1) for value in values
    ):
        raise ValueError("specialized control requires at least two binary activations")
    names = {f.name for f in record.functions}
    index = 0
    prefix = "__fq_specialized_0_"
    while any(name.startswith(prefix) for name in names):
        index += 1
        prefix = f"__fq_specialized_{index}_"
    child = replace(function, name=prefix + "positive")
    definitions = [child, *(f for f in record.functions if f.name != entry)]
    for erase, suffix in ((False, "and"), (True, "erase")):
        definitions.extend(_array_and_program(values, prefix + suffix, erase).functions)
    control = Register(control_name, quantum_bit_type(), (len(values),))
    builder = ComponentProgramBuilder(
        Signature((control, *data_registers)), definitions, name=entry
    )
    conjunction = builder.call_dict(
        prefix + "and", {"ctrl": builder.inputs[control_name]}
    )
    data = {
        r.name: builder.inputs[r.name]
        for r in data_registers
        if r.side & RegisterSide.LEFT
    }
    data[control_name] = conjunction.pop("target")
    outputs = builder.call_dict(child.name, data)
    conjunction["target"] = outputs.pop(control_name)
    outputs[control_name] = builder.call_dict(prefix + "erase", conjunction)["ctrl"]
    return replace(
        record, canonical_payload=None, functions=builder.finalize(outputs).functions
    )


def specialized_component_control_system(
    record: ProgramRecord,
    control_name: str,
    current_value: int,
    added_value: int,
    entry: str = "main",
) -> tuple[ProgramRecord, ControlAdder]:
    """Return the source's two-control helper and scalar-control packing adder.

    The supplied body is the positive specialization. current_value describes
    the original operation's activation. added_value is the added activation.
    The builder owns the returned definitions. The adder pops the original
    control, inserts a packed two-bit array in that same dictionary, calls once
    and returns the new control followed by old-control/data successors.
    """
    module = multi_controlled_component_program(
        record, control_name, (added_value, current_value), entry
    )

    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        old = values.pop(control_name)
        values[control_name] = np.array([control, old], dtype=object)
        packed, *data = builder.call_tuple(entry, values)
        assert isinstance(packed, np.ndarray)
        return packed[0], (packed[1], *data)

    return module, add
