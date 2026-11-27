# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source glyphs follow Qualtran v0.7.0 LP, reflection and basic-gate protocols.

"""Present the complete native LP root with its original source wire symbols."""

from math import acos

from flagquantum.algorithms.primitives.luis_perina_state import (
    luis_perina_interim_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    RegisterWireSymbolSupplier,
    SymbolClass,
)
from flagquantum.drawer.rotation_symbols import rotation_wire_symbol


def luis_perina_window_symbols(
    record: ProgramRecord, width: int
) -> RegisterWireSymbolSupplier:
    """Project only the selected complete LP Program, without executing it."""
    functions = {function.name: function for function in record.functions}
    inverse = luis_perina_interim_program(width, adjoint=True)
    inverse_names = {
        function.name
        for function in record.functions
        if tuple(port.name for port in function.registers or ()) == ("m", "anc")
        and matches_component_definition(record, function.name, inverse)
    }

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        ports = functions[step.name].registers or ()
        names = tuple(port.name for port in ports)
        if names == ("reg",):
            if register is None:
                return "Text", {"text": "", "fontsize": 10}
            if ports[0].side == RegisterSide.RIGHT:
                return "LarrowTextBox", {"text": "alloc"}
            return "RarrowTextBox", {"text": "free"}
        if names == ("m", "anc"):
            suffix = "†" if step.name in inverse_names else ""
            if register is None:
                return "Text", {"text": "LPRS" + suffix, "fontsize": 10}
            label = "LPRSInterimPrep" if register.name == "m" else register.name
            return "TextBox", {"text": label + suffix}
        if names == ("reg0_", "reg1_", "reg2_"):
            if register is None:
                return "Text", {"text": "ReflectionUsingPrepare", "fontsize": 10}
            return "TextBox", {"text": "R_L"}
        if names == ("q1", "q2"):
            if register is None:
                return "Text", {"text": "", "fontsize": 10}
            return "Circle", {"filled": True}
        if step.name == "flip":
            if register is None:
                return "Text", {"text": "X", "fontsize": 10}
            return "ModPlus", {}
        if step.name in {"flag_rotate", "flag_unrotate"}:
            angle = acos(1 / (1 + 2**width))
            if step.name == "flag_unrotate":
                angle = -angle
            return rotation_wire_symbol(
                step, register, index, axis="y", parameter=angle
            )
        raise ValueError("Unknown source call in the selected LP Program")

    return supply
