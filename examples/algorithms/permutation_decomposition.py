# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/permutation.py.

"""Present complete permutation-to-cycle calls through existing native owners."""

from collections.abc import Mapping

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.sorting.permutation_component import (
    permutation_program,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    SymbolClass,
    register_score_data,
)


def permutation_catalog_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Reuse finite owners and retain symbolic cycle calls as declarations.

    Unknown cycle counts and original cycle drawing failures keep atomic catalog
    interfaces. Declared child identity follows the root's ordered cycle data.
    """
    if operation != "Permutation":
        return None
    cycles = parameters["cycles"]
    if not isinstance(cycles, tuple):
        return None
    size, width = parameters["N"], ports[0].bitsize
    if type(size) is int and type(width) is int:
        assert all(
            isinstance(cycle, tuple) and all(type(v) is int for v in cycle)
            for cycle in cycles
        )
        return permutation_program(size, cycles, width)
    builder = ComponentProgramBuilder(ports)
    word = builder.inputs["x"]
    for index, _ in enumerate(cycles):
        child = FunctionRecord(f"PermutationCycle_{index}", (), registers=tuple(ports))
        builder.define(child)
        word = builder.call_dict(child.name, {"x": word})["x"]
    return builder.finalize({"x": word})


def _cycle_symbols(
    step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    if register is None:
        return "Text", {"text": "Permutat..", "fontsize": 10}
    return partition_wire_symbol(step, register, index)


def permutation_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Draw the complete immediate cycle layer with source default labels."""
    record = permutation_catalog_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=_cycle_symbols,
    )
