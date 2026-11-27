# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 gf2_addition.py and gf2_add_k.py.

"""Present binary field addition through its existing executable typed owner."""

from collections.abc import Mapping
from operator import index
from typing import SupportsIndex

from examples.algorithms.word_arithmetic_decomposition import word_symbols
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_decomposition_program,
    field_addition_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_parameters import is_symbolic_value
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import MusicalScoreData, register_score_data
from flagquantum.simulation.finite_field_metadata import resolve_quantum_field_storage


def field_addition_catalog_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Keep original symbolic refusals and concrete field identities."""
    if operation not in {"GF2Addition", "GF2AddK"}:
        return None
    dtype = ports[0].dtype
    if type(dtype.num_qubits) is not int:
        return None
    if operation == "GF2Addition":
        return field_addition_decomposition_program(
            resolve_quantum_field_storage(dtype)
        )
    constant = parameters["k"]
    if is_symbolic_value(constant) or not isinstance(constant, SupportsIndex):
        return None
    return field_addition_program(
        resolve_quantum_field_storage(dtype), constant=index(constant)
    )


def field_addition_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Reuse CNOT, X and typed Split/Join glyphs for the actual stored graph."""
    record = field_addition_catalog_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=word_symbols(
            record, "Xor" if operation == "GF2Addition" else "XorK"
        ),
    )
