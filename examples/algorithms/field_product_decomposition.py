# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 gf_arithmetic/gf2_multiplication.py.

"""Draw actual field product calls and qualify complete recursive definitions."""

from collections.abc import Mapping

from examples.algorithms.component_ui_dependencies import ChildPresentation
from examples.algorithms.field_product_definitions import (
    field_product_catalog_program,
    field_product_definition,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
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
    adjoint_wire_symbol,
    register_score_data,
)


def field_product_symbols(
    record: ProgramRecord, root_ports: Signature, parameters: Mapping[str, object]
) -> RegisterWireSymbolSupplier:
    """Compose existing typed partition, atomic CCX/CX and allocation glyphs."""
    functions = {f.name: f for f in record.functions}
    root = record.functions[0]
    callees = dict.fromkeys(
        op.callee
        for block in root.blocks
        for op in block.operations
        if op.callee is not None
    )
    labels = {
        name: selected[0]
        for name in callees
        if (selected := field_product_definition(record, name, root_ports, parameters))
        is not None
    }

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        function = functions[step.name]
        ports = tuple(r.name for r in function.registers or ())
        if ports == ("ctrl", "target"):
            return controlled_not_wire_symbol(step, register, index)
        opcodes = {op.opcode for b in function.blocks for op in b.operations}
        if "allocate_register" in opcodes or "free_register" in opcodes:
            if register is None:
                return "Text", {"text": "", "fontsize": 10}
            return (
                ("LarrowTextBox", {"text": "alloc"})
                if register.side == RegisterSide.RIGHT
                else ("RarrowTextBox", {"text": "free"})
            )
        if step.name in labels:
            label = labels[step.name]
            if register is None:
                return "Text", {
                    "text": label if len(label) <= 10 else label[:8] + "..",
                    "fontsize": 10,
                }
            symbol = partition_wire_symbol(step, register, index)
            return adjoint_wire_symbol(*symbol) if label == "GF2MulK†" else symbol
        return split_join_wire_symbol(step, register, index)

    return supply


def field_product_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Project full stored multiplication graphs into the normal catalog score."""
    record = field_product_catalog_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=field_product_symbols(record, ports, parameters),
    )


def field_product_child_decomposition(
    record: ProgramRecord,
    entry: str,
    root_ports: Signature,
    parameters: Mapping[str, object],
) -> ChildPresentation | None:
    """Apply a source class label only after literal full-definition matching."""
    selected = field_product_definition(record, entry, root_ports, parameters)
    if selected is None:
        return None
    label, template = selected
    return label, template, field_product_symbols(template, root_ports, parameters)
