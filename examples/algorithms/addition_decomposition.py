# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/addition.py.

"""Draw complete addition source families through their existing native owners."""

from collections.abc import Mapping

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.arithmetic.constant_addition_component import (
    constant_addition_program,
)
from flagquantum.compiler.registers.arithmetic.out_of_place_addition_component import (
    out_of_place_addition_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.bitwise_symbols import bitwise_wire_symbol
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.conjunction_symbols import and_wire_symbol
from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
from flagquantum.drawer.partition_symbols import (
    partition_wire_symbol,
    split_join_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)


def addition_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Reuse existing Add, OutOfPlaceAdder and encoded AddK constructions.

    Symbolic source refusals retain atomic catalog interfaces. Concrete graphs
    reuse their executable owners, including OutOfPlaceAdder's shared source
    wire reconstruction. Physical owners retain execution and allocation.
    """
    if operation not in {"Add", "OutOfPlaceAdder", "AddK"}:
        return None
    width = ports[0].bitsize
    if type(width) is not int:
        return None
    if operation == "Add":
        other = ports[1].bitsize
        if type(other) is not int:
            return None
        return addition_component_program(
            width, other, left_dtype=ports[0].dtype, right_dtype=ports[1].dtype
        )
    if operation == "OutOfPlaceAdder":
        return out_of_place_addition_program(width)
    constant = parameters["k"]
    if type(constant) is not int:
        return None
    return constant_addition_program(ports[0].dtype, constant % (1 << width))


def addition_symbols(
    record: ProgramRecord, parameters: Mapping[str, object]
) -> RegisterWireSymbolSupplier:
    """Retain original immediate child glyphs, including Always's own labels."""
    functions = {f.name: f for f in record.functions}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        function = functions[step.name]
        ports = tuple(r.name for r in function.registers or ())
        opcodes = {op.opcode for b in function.blocks for op in b.operations}
        if ports == ("ctrl", "target"):
            assert function.registers is not None
            if function.registers[0].shape:
                return and_wire_symbol(step, register, index)
            return controlled_not_wire_symbol(step, register, index)
        if ports == ("x",):
            constant = parameters["k"]
            assert type(constant) is int
            return bitwise_wire_symbol(
                step, register, index, operation="xor_constant", constant=constant
            )
        if ports == ("a", "b"):
            if register is None:
                return "Text", {"text": "Add", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        if "allocate_register" in opcodes or "free_register" in opcodes:
            if register is None:
                return "Text", {"text": "", "fontsize": 10}
            return (
                ("LarrowTextBox", {"text": "alloc"})
                if register.side == RegisterSide.RIGHT
                else ("RarrowTextBox", {"text": "free"})
            )
        return split_join_wire_symbol(step, register, index)

    return supply


def addition_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Draw the stored immediate source graph through the existing Drawer."""
    record = addition_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=addition_symbols(record, parameters),
    )
