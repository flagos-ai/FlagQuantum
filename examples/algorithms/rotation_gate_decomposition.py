# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotation.py.

"""Project controlled rotations through their native typed Compiler owners."""

from collections.abc import Mapping
from math import pi

from sympy import Symbol

from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_rz_program,
    controlled_z_power_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.parameters import Parameter
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.conjunction_symbols import and_wire_symbol
from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)
from flagquantum.drawer.rotation_symbols import rotation_wire_symbol


def rotation_gate_program(
    operation: str, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Keep both source controlled decompositions and atomic refusal roots."""
    if operation == "CZPowGate":
        exponent = parameters["exponent"]
        assert isinstance(exponent, (int, float))
        return controlled_z_power_program(float(exponent))
    if operation == "CRz":
        angle = parameters["angle"]
        assert isinstance(angle, (int, float, Symbol))
        return controlled_rz_program(
            Parameter(str(angle)) if isinstance(angle, Symbol) else float(angle)
        )
    return None


def rotation_gate_symbols(
    operation: str, parameters: Mapping[str, object]
) -> RegisterWireSymbolSupplier:
    """Compose unchanged phase, conjunction and CNOT wire glyphs."""
    parameter = parameters["angle"] if operation == "CRz" else parameters["exponent"]
    assert isinstance(parameter, (int, float, Symbol))
    powers = (
        {"positive": (parameter / pi) / 2, "negative": -(parameter / pi) / 2}
        if operation == "CRz"
        else {"z_power": parameter}
    )

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if step.name in {"compute", "erase"}:
            return and_wire_symbol(step, register, index)
        if step.name == "cnot":
            return controlled_not_wire_symbol(step, register, index)
        return rotation_wire_symbol(
            step, register, index, axis="z", parameter=powers[step.name], power=True
        )

    return supply


def rotation_gate_score(
    operation: str, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Draw the actual immediate calls without synthesizing a physical circuit."""
    record = rotation_gate_program(operation, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record),
        register_wire_symbol=rotation_gate_symbols(operation, parameters),
    )
