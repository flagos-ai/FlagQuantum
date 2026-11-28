# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Controlled, bitwise XorK/BitwiseNot,
# Toffoli.get_ctrl_system and ControlledViaAnd.get_ctrl_system.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/bitwise.py

"""Share literal bitwise control propagation across arithmetic consumers."""

from dataclasses import replace
from functools import partial
from typing import Literal

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import ControlAdder, control_component_hierarchy
from ..component_definition_match import matches_component_definition
from ..component_gate_function import bit_gate_function
from ..control.component_bookkeeping import bookkeeping_scalar_control_adder
from ..control.multi_control_component import merged_pauli_control_program
from .bitwise_component import bitwise_word_program


def _preserve_control(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
    *,
    callee: str,
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    return control, tuple(builder.call_dict(callee, values).values())


def _controlled_x(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_dict(
        "bitwise_controlled_x", {"ctrl": control, "target": values["q"]}
    )
    return outputs["ctrl"], (outputs["target"],)


def _controlled_cx(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    controls, target = builder.call_tuple(
        "bitwise_toffoli",
        {"ctrl": np.array([control, values["ctrl"]]), "target": values["target"]},
    )
    assert isinstance(controls, np.ndarray)
    return controls[0], (controls[1], target)


def _toffoli_function() -> FunctionRecord:
    bit = quantum_bit_type()
    return bit_gate_function(
        "bitwise_toffoli",
        (Register("ctrl", bit, (2,)), Register("target", bit)),
        CircuitIR(3, (Instruction("ccx", (0, 1, 2)),)),
    )


def _controlled_toffoli(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    """Use Toffoli.get_ctrl_system's merged scalar/array X predicate literally."""
    outputs = builder.call_program(
        merged_pauli_control_program((1, 1), "x"),
        {"ctrl1": control, "ctrl2": values["ctrl"], "q": values["target"]},
    )
    return outputs["ctrl1"], (outputs["ctrl2"], outputs["q"])


def nested_controlled_xor_constant_program(
    dtype: RegisterType, pattern: int, *, control_count: int = 2
) -> ProgramRecord:
    """Retain distinct positive scalar controls on the source XOR loader.

    Source nested Controlled retains the inner Split/XOR/Join calls and turns
    each loaded CNOT into an atomic Toffoli. This is distinct from a grouped
    predicate that first computes a shared conjunction into private work.
    Further controls retain the exact merged-predicate callback per loaded bit.
    """
    if type(control_count) is not int or control_count < 2:
        raise ValueError("Nested XOR loading requires at least two positive controls")
    record = (
        controlled_bitwise_word_program(dtype, "xor_constant", pattern=pattern)
        if control_count == 2
        else nested_controlled_xor_constant_program(
            dtype, pattern, control_count=control_count - 1
        )
    )
    adders: dict[str, ControlAdder] = (
        {"bitwise_controlled_x": _controlled_cx}
        if control_count == 2
        else {"bitwise_toffoli": _controlled_toffoli}
    )
    if control_count > 3:
        previous = merged_pauli_control_program(
            (1, 1), "x", scalar_controls=control_count - 3
        )
        merged = merged_pauli_control_program(
            (1, 1), "x", scalar_controls=control_count - 2
        )
        previous_ports = previous.functions[0].registers
        assert previous_ports is not None
        previous_controls = tuple(r.name for r in previous_ports[:-1])

        def add_merged(
            builder: ComponentProgramBuilder,
            control: BindingCollection,
            values: dict[str, BindingCollection],
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            inputs = {"ctrl1": control, "q": values["q"]}
            inputs.update(
                {
                    f"ctrl{i + 2}": values[name]
                    for i, name in enumerate(previous_controls)
                }
            )
            outputs = builder.call_program(merged, inputs)
            return outputs["ctrl1"], (
                *(outputs[f"ctrl{i + 2}"] for i in range(len(previous_controls))),
                outputs["q"],
            )

        adders = {
            function.name: add_merged
            for function in record.functions
            if matches_component_definition(record, function.name, previous)
        }
    callees = {
        operation.callee for operation in record.functions[0].blocks[0].operations
    }
    for function in record.functions:
        if (
            function.name in callees
            and function.registers is not None
            and any(
                register.side != RegisterSide.THRU for register in function.registers
            )
        ):
            adders[function.name] = bookkeeping_scalar_control_adder(function)
    return control_component_hierarchy(
        (
            replace(record, functions=(*record.functions, _toffoli_function()))
            if control_count == 2
            else record
        ),
        adders,
    )


def controlled_bitwise_word_program(
    dtype: RegisterType,
    operation: Literal["bitwise_not", "xor_constant", "xor"],
    *,
    pattern: int | None = None,
) -> ProgramRecord:
    """Control complement, encoded XOR or two-word XOR through actual child calls."""
    record = bitwise_word_program(dtype, operation, pattern=pattern)
    bit = quantum_bit_type()
    cx = bit_gate_function(
        "bitwise_controlled_x",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    if operation == "xor":
        cx = _toffoli_function()
    adders: dict[str, ControlAdder] = {}
    for function in record.functions:
        assert function.registers is not None
        if function.name.endswith("bitwise_x"):
            adders[function.name] = _controlled_x
        elif function.name.endswith("bitwise_cx"):
            adders[function.name] = _controlled_cx
        elif any(register.side != RegisterSide.THRU for register in function.registers):
            adders[function.name] = partial(_preserve_control, callee=function.name)
    return control_component_hierarchy(
        replace(record, functions=(*record.functions, cx)), adders
    )
