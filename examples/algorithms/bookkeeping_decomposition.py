# Copyright 2023 Google LLC
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 auto_partition.py and always.py.

"""Draw complete bookkeeping wrappers using existing partition and primitive owners."""

from dataclasses import replace

from examples.algorithms.basic_gate_decomposition import decomposition_symbols
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.control.conjunction_program import (
    and_component_program,
)
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
)
from flagquantum.compiler.registers.primitives.always_component import (
    always_component_program,
)
from flagquantum.compiler.registers.primitives.controlled_swap_component import (
    controlled_swap_program,
)
from flagquantum.compiler.registers.primitives.swap_component import SwapDeclaration
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.conjunction_symbols import and_wire_symbol
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)
from flagquantum.drawer.swap_symbols import swap_wire_symbol


def bookkeeping_program(
    operation: str, ports: Signature, layout: AutoPartition | None
) -> ProgramRecord | None:
    """Retain original wrapper calls and atomic refusal boundaries.

    The original catalog wrappers contain a positive controlled one-bit Swap
    and a positive two-input And. Existing Compiler owners retain both children.
    """
    if operation == "Always":
        return always_component_program(and_component_program((1, 1), name="main"))
    if operation == "AutoPartition" and layout is not None:
        record = auto_partition_component_program(
            controlled_swap_program(1), layout, packed_ports=tuple(ports)
        )
        root = next(f for f in record.functions if f.name == "packed")
        return replace(
            record, functions=(root, *(f for f in record.functions if f != root))
        )
    return None


def bookkeeping_symbols(record: ProgramRecord) -> RegisterWireSymbolSupplier:
    """Compose original partition, controlled Swap and conjunction glyphs."""
    functions = {f.name: f for f in record.functions}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        names = tuple(r.name for r in functions[step.name].registers or ())
        if names == ("ctrl", "target"):
            return and_wire_symbol(step, register, index, values=(1, 1))
        if names == ("ctrl", "x", "y"):
            return swap_wire_symbol(step, register, index, controlled=True)
        return partition_wire_symbol(step, register, index)

    return supply


def bookkeeping_score(
    operation: str, ports: Signature, layout: AutoPartition | None
) -> MusicalScoreData | None:
    """Project the existing complete wrapper without physical state allocation."""
    record = bookkeeping_program(operation, ports, layout)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=bookkeeping_symbols(record),
    )


def bookkeeping_child(
    record: ProgramRecord, entry: str, root: object
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Qualify the controlled Swap's complete source decomposition."""
    template = controlled_swap_program(1)
    if not matches_component_definition(record, entry, template):
        return None
    return (
        "Controlled",
        template,
        decomposition_symbols(template, SwapDeclaration(1, controlled=True)),
    )
