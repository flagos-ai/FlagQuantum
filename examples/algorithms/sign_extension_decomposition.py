# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/conversions/sign_extension.py.

"""Draw sign extension and truncation through the existing executable owner."""

from examples.algorithms.subtraction_decomposition import subtraction_symbols
from flagquantum.compiler.registers.bits.sign_extension_component import (
    sign_extension_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    register_score_data,
)


def sign_extension_catalog_program(
    operation: str, ports: Signature
) -> ProgramRecord | None:
    """Select the exact owner, including the source truncation's output view."""
    if operation == "SignExtend":
        return sign_extension_program(ports[0].dtype, ports[1].dtype)
    if operation == "SignTruncate":
        return sign_extension_program(ports[1].dtype, ports[0].dtype, adjoint=True)
    return None


def sign_extension_score(operation: str, ports: Signature) -> MusicalScoreData | None:
    """Compose existing workspace, partition and fanout glyphs in source order."""
    record = sign_extension_catalog_program(operation, ports)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=subtraction_symbols(record),
    )


def sign_extension_child_decomposition(
    record: ProgramRecord, entry: str
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Qualify consumed/returned sign views by their complete native definition."""
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    if (
        tuple(register.name for register in registers) != ("x", "y")
        or tuple(register.side for register in registers)
        != (RegisterSide.LEFT, RegisterSide.RIGHT)
        or any(
            type(register.bitsize) is not int
            or register.bitsize < 1
            or register.dtype.encoding not in {"signed", "ones_complement"}
            or register.shape_symbolic
            for register in registers
        )
        or registers[0].dtype.encoding != registers[1].dtype.encoding
        or registers[0].bitsize == registers[1].bitsize
    ):
        return None
    label = (
        "SignExtend" if registers[0].bitsize < registers[1].bitsize else "SignTruncate"
    )
    template = sign_extension_catalog_program(label, Signature(registers))
    assert template is not None
    if matches_component_definition(record, entry, template):
        return label, template, subtraction_symbols(template)
    return None
