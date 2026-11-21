# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/permutation.py.

"""Present complete permutation-to-cycle calls through existing native owners."""

from collections.abc import Mapping

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.sorting.permutation_component import (
    permutation_cycle_program,
    permutation_program,
)
from flagquantum.compiler.registers.sorting.permutation_graph import (
    permutation_cycle_graph,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
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
    """Retain finite cycles independently of symbolic widths and drawing failures.

    Unknown cycle values retain the original decomposition refusal. Concrete
    widths and bounds reuse executable owners. Symbolic children retain their
    original refusal leaves beneath the shared finite cycle wiring.
    """
    if operation == "PermutationCycle":
        cycle = parameters["cycle"]
        if not isinstance(cycle, tuple) or not all(type(v) is int for v in cycle):
            return None
        size, width = parameters["N"], ports[0].bitsize
        if type(size) is int and type(width) is int:
            return permutation_cycle_program(size, cycle, width)
        return symbolic_cycle_program(ports, cycle)
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
    for index, cycle in enumerate(cycles):
        if isinstance(cycle, tuple) and all(type(v) is int for v in cycle):
            child = symbolic_cycle_program(ports, cycle)
            word = builder.call_program(child, {"x": word})["x"]
        else:
            declared = FunctionRecord(
                f"PermutationCycle_{index}", (), registers=tuple(ports)
            )
            builder.define(declared)
            word = builder.call_dict(declared.name, {"x": word})["x"]
    return builder.finalize({"x": word})


def symbolic_cycle_program(ports: Signature, cycle: tuple[int, ...]) -> ProgramRecord:
    """Supply original symbolic equality/XOR declarations to shared cycle wiring."""
    word = unsigned_register_type(ports[0].bitsize)

    def equality(value: int) -> ProgramRecord:
        return ProgramRecord(
            (
                FunctionRecord(
                    f"EqualsAConstant_{value}",
                    (),
                    registers=tuple(
                        Signature.build_from_dtypes(x=word, target=quantum_bit_type())
                    ),
                ),
            )
        )

    def controlled_xor(value: int) -> ProgramRecord:
        return ProgramRecord(
            (
                FunctionRecord(
                    f"ControlledXorK_{value}",
                    (),
                    registers=tuple(
                        Signature.build_from_dtypes(ctrl=quantum_bit_type(), x=word)
                    ),
                ),
            )
        )

    return permutation_cycle_graph(ports[0].dtype, cycle, equality, controlled_xor)


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
    # Frozen source cycle drawings raise ValueError despite successful decomposition.
    if operation == "PermutationCycle":
        return None
    record = permutation_catalog_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=_cycle_symbols,
    )
