# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 gradient-mediated Rz and ZPow graphs.

"""Present gradient rotations through their existing typed arithmetic owners."""

from collections.abc import Mapping

from sympy import Expr

from examples.algorithms.addition_child_decomposition import (
    addition_child_decomposition,
)
from examples.algorithms.controlled_word_decomposition import (
    controlled_word_child_decomposition,
)
from examples.algorithms.word_arithmetic_decomposition import word_symbols
from flagquantum.compiler.registers.arithmetic.conditional_addition_component import (
    conditional_addition_program,
)
from flagquantum.compiler.registers.bits.bitwise_component_control import (
    controlled_bitwise_word_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.phase.rz_phase_gradient_component import (
    rz_via_phase_gradient_program,
)
from flagquantum.compiler.registers.phase.zpow_phase_gradient import (
    zpow_phase_gradient_constant,
)
from flagquantum.compiler.registers.phase.zpow_phase_gradient_component import (
    zpow_via_phase_gradient_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)


def phase_rotation_program(
    operation: str, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Keep concrete/symbolic-precision graphs and symbolic-angle refusal."""
    if operation == "ZPowConstViaPhaseGradient":
        exponent = parameters["exponent"]
        if not isinstance(exponent, (int, float)):
            return None
        width = parameters["phase_grad_bitsize"]
        assert isinstance(width, (int, Expr))
        return zpow_via_phase_gradient_program(exponent, width)
    if operation == "RzViaPhaseGradient":
        angle, phase = parameters["angle_dtype"], parameters["phasegrad_dtype"]
        assert isinstance(angle, dict) and isinstance(phase, dict)
        return rz_via_phase_gradient_program(
            fixed_point_register_type(angle["bitsize"], angle["num_frac"]),
            fixed_point_register_type(phase["bitsize"], phase["num_frac"]),
        )
    return None


def phase_rotation_symbols(record: ProgramRecord) -> RegisterWireSymbolSupplier:
    """Use native fallback titles where the original controlled title raises."""
    functions = {f.name: f for f in record.functions}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        ports = functions[step.name].registers or ()
        names = tuple(r.name for r in ports)
        if names == ("reg", "reg"):
            if register is None:
                return "Text", {"text": "Cast", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        if names == ("ctrl", "a", "b"):
            if register is None:
                return "Text", {"text": "Controll..", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        return partition_wire_symbol(step, register, index)

    return supply


def phase_rotation_score(
    operation: str, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Draw actual stored roots with separately recorded source title failures."""
    record = phase_rotation_program(operation, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record),
        register_wire_symbol=phase_rotation_symbols(record),
    )


def phase_rotation_child(
    record: ProgramRecord, entry: str, parameters: Mapping[str, object]
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Require literal complete definitions before source arithmetic labels."""
    function = next(f for f in record.functions if f.name == entry)
    ports = function.registers or ()
    names = tuple(r.name for r in ports)
    if names == ("ctrl", "a", "b"):
        for polarity in (True, False):
            template = conditional_addition_program(
                ports[1].dtype, ports[2].dtype, add_when_control_is_on=polarity
            )
            if matches_component_definition(record, entry, template):
                # Its original score fails on Controlled(BitwiseNot)'s title.
                return "ControlledAddOrSubtract", template, partition_wire_symbol
    if (
        names == ("ctrl", "x")
        and "exponent" in parameters
        and type(ports[1].bitsize) is int
    ):
        exponent = parameters["exponent"]
        assert isinstance(exponent, (int, float))
        template = controlled_bitwise_word_program(
            ports[1].dtype,
            "xor_constant",
            pattern=zpow_phase_gradient_constant(exponent, ports[1].bitsize),
        )
        if matches_component_definition(record, entry, template):
            return "Controlled", template, word_symbols(template, "Controlled")
    return controlled_word_child_decomposition(
        record, entry
    ) or addition_child_decomposition(record, entry, parameters)
