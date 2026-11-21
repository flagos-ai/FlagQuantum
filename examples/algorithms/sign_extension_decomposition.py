# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/conversions/sign_extension.py.

"""Draw sign extension and truncation through the existing executable owner."""

from examples.algorithms.subtraction_decomposition import subtraction_symbols
from flagquantum.compiler.registers.bits.sign_extension_component import (
    sign_extension_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import MusicalScoreData, register_score_data


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
