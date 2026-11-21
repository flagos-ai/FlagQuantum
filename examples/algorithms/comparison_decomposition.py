# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/comparison.py.

"""Present comparison source calls through the existing native component owners."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal, cast

from examples.algorithms.subtraction_decomposition import subtraction_symbols
from flagquantum.compiler.registers.comparison._component_gates import (
    comparison_bit_functions,
)
from flagquantum.compiler.registers.comparison.constant_component import (
    less_than_constant_program,
)
from flagquantum.compiler.registers.comparison.controlled_component import (
    controlled_greater_program,
)
from flagquantum.compiler.registers.comparison.equality_component import (
    equality_constant_program,
    equality_program,
)
from flagquantum.compiler.registers.comparison.half_component import (
    half_comparison_program,
)
from flagquantum.compiler.registers.comparison.ordering_component import (
    greater_than_program,
    less_equal_program,
)
from flagquantum.compiler.registers.comparison.tree_component import (
    mixer_program,
    single_qubit_comparison_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.addition_symbols import addition_wire_symbol
from flagquantum.drawer.basis_symbols import pauli_basis_wire_symbol
from flagquantum.drawer.bitwise_symbols import bitwise_wire_symbol
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.conjunction_symbols import (
    and_wire_symbol,
    multi_and_wire_symbol,
)
from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    PortSide,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)


def symbolic_greater_than_program(ports: Signature) -> ProgramRecord:
    """Retain GreaterThan's source graph with a declared symbolic comparison.

    Word widths remain unresolved. The existing one-bit X definition is reused,
    and physical admission remains with the register binder.
    """
    comparison = FunctionRecord(
        "LessThanEqual",
        (),
        registers=tuple(
            Signature.build_from_dtypes(
                x=ports[0].dtype, y=ports[1].dtype, target=ports[2].dtype
            )
        ),
    )
    x_gate, _ = comparison_bit_functions()
    builder = ComponentProgramBuilder(ports, (comparison, x_gate))
    result = builder.call_dict(
        comparison.name,
        {
            "x": builder.inputs["a"],
            "y": builder.inputs["b"],
            "target": builder.inputs["target"],
        },
    )
    target = builder.call_dict(x_gate.name, {"q": result["target"]})["q"]
    return builder.finalize({"a": result["x"], "b": result["y"], "target": target})


def comparison_catalog_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Select native owners and retain all three catalog source refusals.

    Concrete comparison owners retain the source SDK layer order and lazy
    typed views. EqualsAConstant keeps its existing nominal output view.
    """
    if operation == "LessThanConstant" and type(ports[0].bitsize) is int:
        threshold = parameters["less_than_val"]
        assert type(threshold) is int
        return less_than_constant_program(ports[0].bitsize, threshold)
    if operation == "LessThanEqual" and type(ports[0].bitsize) is int:
        other = ports[1].bitsize
        assert type(other) is int
        return less_equal_program(ports[0].bitsize, other)
    if operation == "BiQubitsMixer":
        return mixer_program()
    if operation == "SingleQubitCompare":
        return single_qubit_comparison_program()
    if operation == "GreaterThan" and type(ports[0].bitsize) is int:
        other = ports[1].bitsize
        assert type(other) is int
        return greater_than_program(ports[0].bitsize, other)
    if operation == "Equals":
        return equality_program(ports[0].dtype)
    if operation == "EqualsAConstant":
        width, value = ports[0].bitsize, parameters["val"]
        assert type(width) is int and type(value) is int
        return equality_constant_program(width, value)
    if operation == "CLinearDepthGreaterThan":
        control = parameters["cv"]
        assert type(control) is int
        return controlled_greater_program(ports[1].dtype, control_value=control)
    relations: dict[str, Literal["gt", "ge", "lt", "le"]] = {
        "LinearDepthHalfGreaterThan": "gt",
        "LinearDepthHalfGreaterThanEqual": "ge",
        "LinearDepthHalfLessThan": "lt",
        "LinearDepthHalfLessThanEqual": "le",
    }
    if operation in relations:
        return half_comparison_program(ports[0].dtype, relations[operation])
    return None


def comparison_symbols(
    record: ProgramRecord, operation: str, parameters: Mapping[str, object]
) -> RegisterWireSymbolSupplier:
    """Compose comparison, MCX, arithmetic, workspace and partition glyphs."""
    functions = {f.name: f for f in record.functions}
    fallback = subtraction_symbols(record)

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        registers = functions[step.name].registers or ()
        ports = tuple(r.name for r in registers)
        if ports == ("ctrl", "target"):
            if registers[0].shape:
                return and_wire_symbol(
                    step,
                    register,
                    index,
                    values={
                        "LessThanEqual": (1, 0),
                        "SingleQubitCompare": (0, 1),
                    }.get(operation, (1, 1)),
                )
            return controlled_not_wire_symbol(step, register, index)
        if ports == ("ctrl", "junk", "target"):
            if register is None and registers[-1].side.name == "LEFT":
                return "Text", {"text": "†", "fontsize": 10}
            return multi_and_wire_symbol(
                step, register, index, values=(0,) * registers[0].shape[0]
            )
        if ports == ("q",):
            return pauli_basis_wire_symbol(step, register, index, gate="x")
        if ports == ("controls", "target"):
            count = registers[0].shape[0]
            if register is None:
                return "TextBox", {"text": "CCX" if count == 2 else f"C^{count}X"}
            if register.name == "target":
                return "ModPlus", {}
            assert index is not None
            if operation == "Equals":
                polarity = 0
            elif operation == "EqualsAConstant":
                value = parameters["val"]
                assert type(value) is int
                polarity = (value >> (count - 1 - index[0])) & 1
            else:
                control = parameters["cv"]
                assert type(control) is int
                polarity = 1 if index[0] else control
            return "Circle", {"filled": bool(polarity)}
        if ports in {("x", "y", "ancilla"), ("a", "b", "less_than", "greater_than")}:
            if register is None:
                return "Text", {
                    "text": (
                        "BiQubitsMixer" if ports[0] == "x" else "SingleQubitCompare"
                    ),
                    "fontsize": 10,
                }
            if register.name == ports[0]:
                return "TextBox", {
                    "text": "BiQubitsMixer" if ports[0] == "x" else "SingleQubitCompare"
                }
            return partition_wire_symbol(step, register, index)
        if ports == ("x", "y", "target"):
            if register is None:
                return "Text", {"text": "", "fontsize": 10}
            return "TextBox", {
                "text": "z∧(x<=y)" if register.name == "target" else register.name
            }
        if ports == ("x", "y"):
            if registers[0].side != registers[1].side:
                title = (
                    "SignExtend"
                    if registers[0].bitsize < registers[1].bitsize
                    else "SignTruncate"
                )
                if register is None:
                    return "Text", {
                        "text": title if len(title) <= 10 else title[:8] + "..",
                        "fontsize": 10,
                    }
                return partition_wire_symbol(step, register, index)
            return bitwise_wire_symbol(step, register, index, operation="xor")
        if ports == ("a", "b", "c"):
            return addition_wire_symbol(
                replace(step, gate="out_of_place_add", name="OutOfPlaceAdder"),
                None if register is None else register.name,
                None if register is None else cast(PortSide, register.side.name),
            )
        return fallback(step, register, index)

    return supply


def comparison_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Draw the complete immediate comparison graph in the native Drawer."""
    record = comparison_catalog_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=comparison_symbols(record, operation, parameters),
    )
