# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 mcmt control and conjunction decompositions.

"""Qualify conjunction and controlled-X descendants through shared native owners."""

from collections.abc import Mapping
from dataclasses import replace

from examples.algorithms.basic_gate_child_decomposition import child_decomposition
from examples.algorithms.control_predicate_catalog import PredicateExample
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.multi_control_component import (
    multi_control_x_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.basis_symbols import pauli_basis_wire_symbol
from flagquantum.drawer.control_symbols import (
    control_predicate_wire_symbol,
    control_register_wire_symbol,
    controlled_wire_title,
)
from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    RegisterWireSymbolSupplier,
    SymbolClass,
)


def _patterns(
    count: int, parameters: Mapping[str, object]
) -> tuple[tuple[int, ...], ...]:
    patterns = [(0,) * count, (1,) * count]
    value = parameters.get("val")
    if type(value) is int and 0 <= value < 1 << count:
        patterns.append(tuple((value >> i) & 1 for i in reversed(range(count))))
    control = parameters.get("cv")
    if type(control) is int and control in (0, 1):
        patterns.append((control, *((1,) * (count - 1))))
    return tuple(dict.fromkeys(patterns))


def _controlled_x_symbols(values: tuple[int, ...]) -> RegisterWireSymbolSupplier:
    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if register is None:
            return controlled_wire_title(
                *pauli_basis_wire_symbol(step, None, index, gate="x")
            )
        if register.name == "ctrl":
            assert index is not None
            return control_register_wire_symbol(register, index, values)
        return pauli_basis_wire_symbol(step, register, index, gate="x")

    return supply


def _via_and_symbols(
    record: ProgramRecord, values: tuple[int, ...]
) -> RegisterWireSymbolSupplier:
    functions = {function.name: function for function in record.functions}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        registers = functions[step.name].registers or ()
        if registers and registers[0].name == "ctrl_0":
            return control_predicate_wire_symbol(
                step,
                register,
                index,
                values=(values,),
                uncompute=registers[-1].side == RegisterSide.LEFT,
            )
        return controlled_not_wire_symbol(step, register, index)

    return supply


def control_child_decomposition(
    record: ProgramRecord, entry: str, parameters: Mapping[str, object]
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Use concrete port domains and declared polarities to qualify full bodies.

    Uniform polarities and owning constant/control metadata select constructor
    candidates. Every reachable phase, measurement, feedback and operand must
    match before source symbols are reused. No polarity is inferred by a name.
    """
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    if not registers:
        return None
    ports = tuple(register.name for register in registers)
    first = registers[0]
    if (
        first.dtype.encoding != "bit"
        or len(first.shape_symbolic) != 1
        or type(first.shape_symbolic[0]) is not int
        or first.shape_symbolic[0] < 2
    ):
        return None
    count = first.shape_symbolic[0]
    for values in _patterns(count, parameters):
        selected = None
        if ports == ("ctrl", "junk", "target"):
            selected = child_decomposition(record, entry, ConjunctionCall(values))
        elif ports in {("ctrl_0", "junk", "target"), ("ctrl_0", "target")}:
            selected = child_decomposition(
                record,
                entry,
                PredicateExample((first,), (ControlRegister(1, values),)),
            )
        elif ports in {("controls", "target"), ("ctrl", "q")}:
            template = multi_control_x_program(values)
            label = "MultiControlX"
            symbols = _controlled_x_symbols(values)
            if ports == ("ctrl", "q"):
                callee = template.functions[0].blocks[0].operations[0].callee
                assert callee is not None
                child = next(f for f in template.functions if f.name == callee)
                template = replace(
                    template,
                    functions=(child, *(f for f in template.functions if f != child)),
                )
                label, symbols = "ControlledViaAnd", _via_and_symbols(template, values)
            if matches_component_definition(record, entry, template):
                selected = label, template, symbols
        if selected is not None:
            return selected
    return None
