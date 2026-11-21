# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/controlled_addition.py.

"""Draw CAdd's complete immediate calls using the existing executable owner."""

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from examples.algorithms.addition_decomposition import addition_symbols
from flagquantum.compiler.registers.arithmetic.controlled_addition_component import (
    controlled_addition_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.addition_symbols import addition_wire_symbol
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.conjunction_symbols import and_wire_symbol
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    PortSide,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)


def controlled_addition_catalog_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Select native CAdd, retaining its exact storage and control polarity."""
    if operation != "CAdd":
        return None
    if type(ports[1].bitsize) is not int or type(ports[2].bitsize) is not int:
        return None
    polarity = parameters["cv"]
    if type(polarity) is not int:
        raise ValueError("CAdd catalog requires a concrete control polarity")
    return controlled_addition_program(
        ports[1].dtype, ports[2].dtype, control_value=polarity
    )


def controlled_addition_symbols(
    record: ProgramRecord, polarity: int
) -> RegisterWireSymbolSupplier:
    """Compose source Add/Cast glyphs with existing And and partition glyphs."""
    functions = {f.name: f for f in record.functions}
    fallback = addition_symbols(record, {})

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        function = functions[step.name]
        ports = tuple(r.name for r in function.registers or ())
        if ports == ("ctrl", "target") and (function.registers or ())[0].shape:
            return and_wire_symbol(step, register, index, values=(polarity, 1))
        if ports == ("a", "b"):
            return addition_wire_symbol(
                replace(step, gate="add"),
                None if register is None else register.name,
                None if register is None else cast(PortSide, register.side.name),
            )
        registers = function.registers or ()
        if (
            ports == ("reg", "reg")
            and registers[0].shape_symbolic == registers[1].shape_symbolic
        ):
            if register is None:
                return "Text", {"text": "Cast", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        return fallback(step, register, index)

    return supply


def controlled_addition_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Project all stored CAdd child calls into the existing register Drawer."""
    record = controlled_addition_catalog_program(operation, ports, parameters)
    if record is None:
        return None
    polarity = parameters["cv"]
    assert type(polarity) is int
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=controlled_addition_symbols(record, polarity),
    )
