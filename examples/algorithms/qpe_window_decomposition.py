# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 phase_estimation source files.

"""Present stored phase-window and textbook estimation dependency graphs."""

from collections.abc import Mapping
from dataclasses import replace
from math import pi

from sympy import Expr

from examples.algorithms.luis_perina_decomposition import luis_perina_window_symbols
from examples.algorithms.rotation_gate_decomposition import rotation_gate_symbols
from flagquantum.algorithms.primitives.luis_perina_state import (
    luis_perina_interim_program,
    luis_perina_state_program,
)
from flagquantum.algorithms.primitives.phase_estimation_window import (
    rectangular_window_program,
)
from flagquantum.algorithms.primitives.textbook_phase_estimation import (
    textbook_z_phase_estimation_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.phase.phase_gradient_component import (
    phase_gradient_unitary_program,
)
from flagquantum.compiler.registers.phase.textbook_qft_component import (
    textbook_qft_program,
)
from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_rz_program,
    controlled_z_power_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.value_math import SymbolicInt
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.hadamard_symbols import hadamard_wire_symbol
from flagquantum.drawer.on_each_symbols import on_each_wire_symbol
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
from flagquantum.drawer.rotation_symbols import rotation_wire_symbol


def window_program(
    operation: str, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Retain successful factories, symbolic declarations and atomic refusals."""
    width = parameters.get("bitsize")
    if operation == "RectangularWindowState":
        assert isinstance(width, (int, Expr))
        return rectangular_window_program(width)
    if operation == "LPRSInterimPrep" and type(width) is int:
        return luis_perina_interim_program(width)
    if operation == "LPResourceState" and type(width) is int:
        return luis_perina_state_program(width)
    if operation == "TextbookQPE":
        preparation, unitary = parameters["ctrl_state_prep"], parameters["unitary"]
        assert isinstance(preparation, dict) and isinstance(unitary, dict)
        width, exponent = preparation["bitsize"], unitary["exponent"]
        if type(width) is int and isinstance(exponent, (int, float)):
            return textbook_z_phase_estimation_program(width, exponent)
    return None


def _title(text: str) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    return "Text", {
        "text": text if len(text) <= 10 else text[:8] + "..",
        "fontsize": 10,
    }


def window_symbols(
    record: ProgramRecord,
    *,
    width: SymbolicInt,
    exponent: float = 0.468,
    adjoint: bool = False,
) -> RegisterWireSymbolSupplier:
    """Project source glyphs from literal, definition-qualified constructors."""
    functions = {f.name: f for f in record.functions}
    root_ports = tuple(p.name for p in record.functions[0].registers or ())
    lp_root = (
        root_ports == ("qpe_reg",)
        and isinstance(width, int)
        and len(record.functions[0].blocks[0].operations) == 15
        and matches_component_definition(
            record, record.functions[0].name, luis_perina_state_program(width)
        )
    )
    if lp_root:
        assert isinstance(width, int)
        return luis_perina_window_symbols(record, width)
    controlled_powers = [
        op.callee
        for op in record.functions[0].blocks[0].operations
        if op.callee
        and tuple(p.name for p in functions[op.callee].registers or ()) == ("q",)
        and (functions[op.callee].registers or ())[0].shape_symbolic == (2,)
    ]
    gradient_powers = dict(
        zip(
            controlled_powers,
            (-0.5 / 2**i for i in reversed(range(len(controlled_powers)))),
            strict=True,
        )
    )

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        function = functions[step.name]
        ports = function.registers or ()
        names = tuple(p.name for p in ports)
        body = function.blocks[0].operations if function.blocks else ()
        if names == ("reg", "reg"):
            return split_join_wire_symbol(step, register, index)
        if names == ("reg",):
            return (
                ("Text", {"text": "", "fontsize": 10})
                if register is None
                else ("LarrowTextBox", {"text": "alloc"})
            )
        if step.name == "hadamard":
            return hadamard_wire_symbol(step, register, index)
        if step.name in {"interim_rz", "interim_inverse_rz"}:
            assert isinstance(width, int)
            return rotation_wire_symbol(
                step,
                register,
                index,
                axis="z",
                parameter=(1 if adjoint else -1) * 2 * pi / (2**width + 1),
            )
        if names == ("ctrl", "phase_grad"):
            if register is None:
                return "Text", {"text": "PhaseGradientUnitary†", "fontsize": 10}
            return (
                ("Circle", {"filled": True})
                if register.name == "ctrl"
                else ("TextBox", {"text": "Z^0.5/2†"})
            )
        if names == ("x", "y"):
            return (
                _title("TwoBitSwap")
                if register is None
                else partition_wire_symbol(step, register, index)
            )
        if names == ("ctrl", "q"):
            if root_ports == ("m", "anc"):
                assert isinstance(width, int)
                for i in range(width):
                    angle = -2 * pi * 2**i / (2**width + 1)
                    if matches_component_definition(
                        record,
                        step.name,
                        controlled_rz_program(angle, adjoint=adjoint),
                    ):
                        symbol = (
                            _title(f"CRz({angle})")
                            if register is None
                            else partition_wire_symbol(step, register, index)
                        )
                        return adjoint_wire_symbol(*symbol) if adjoint else symbol
                raise ValueError("Unknown interim controlled rotation")
            count = len(body)
            if register is None:
                return _title("C[Power]")
            return (
                ("Circle", {"filled": True})
                if register.name == "ctrl"
                else ("TextBox", {"text": f"Power^{count}"})
            )
        if names == ("qpe_reg",):
            return (
                _title("RectangularWindowState")
                if register is None
                else partition_wire_symbol(step, register, index)
            )
        if names == ("q",) and ports[0].shape_symbolic == (2,):
            value = (
                gradient_powers[step.name]
                if root_ports == ("ctrl", "phase_grad")
                else exponent
            )
            return (
                _title(f"CZ**{value}")
                if register is None
                else partition_wire_symbol(step, register, index)
            )
        if root_ports == ("qpe_reg", "q"):
            return (
                ("Text", {"text": "QFTTextBook†", "fontsize": 10})
                if register is None
                else ("TextBox", {"text": "QFTTextBook†"})
            )
        symbol = on_each_wire_symbol(
            step, register, index, width=width, child_symbol=hadamard_wire_symbol
        )
        return adjoint_wire_symbol(*symbol) if adjoint else symbol

    return supply


def window_score(
    operation: str, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    record = window_program(operation, parameters)
    if record is None:
        return None
    width = parameters.get("bitsize")
    exponent = 0.468
    if operation == "TextbookQPE":
        unitary = parameters["unitary"]
        assert isinstance(unitary, dict) and isinstance(
            unitary["exponent"], (int, float)
        )
        exponent = unitary["exponent"]
        preparation = parameters["ctrl_state_prep"]
        assert isinstance(preparation, dict)
        width = preparation["bitsize"]
    assert isinstance(width, (int, Expr))
    return register_score_data(
        component_register_score(record),
        register_wire_symbol=window_symbols(record, width=width, exponent=exponent),
    )


def window_child(
    record: ProgramRecord,
    entry: str,
    parameters: Mapping[str, object],
    *,
    operation: str,
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    if operation not in {
        "RectangularWindowState",
        "LPRSInterimPrep",
        "LPResourceState",
        "TextbookQPE",
    }:
        return None
    function = next(f for f in record.functions if f.name == entry)
    ports = function.registers or ()
    names = tuple(p.name for p in ports)
    if names == ("m", "anc") and type(ports[0].bitsize) is int:
        interim_width = ports[0].bitsize
        for adjoint in (False, True):
            template = luis_perina_interim_program(interim_width, adjoint=adjoint)
            if matches_component_definition(record, entry, template):
                return (
                    "LPRSInterimPrep" + ("†" if adjoint else ""),
                    template,
                    window_symbols(template, width=interim_width, adjoint=adjoint),
                )
    if names == ("qpe_reg",):
        template = rectangular_window_program(ports[0].bitsize)
        if matches_component_definition(record, entry, template):
            return (
                "RectangularWindowState",
                template,
                window_symbols(template, width=ports[0].bitsize),
            )
    if (
        names == ("q",)
        and not ports[0].shape_symbolic
        and ports[0].dtype.encoding == "uint"
    ):
        template = textbook_qft_program(ports[0].bitsize, adjoint=True)
        if matches_component_definition(record, entry, template):
            return (
                "QFTTextBook†",
                template,
                window_symbols(template, width=ports[0].bitsize),
            )
    if names == ("ctrl", "phase_grad"):
        template = phase_gradient_unitary_program(
            ports[1].bitsize, 0.5, controlled=True, adjoint=True
        )
        if matches_component_definition(record, entry, template):
            return (
                "PhaseGradientUnitary†",
                template,
                window_symbols(template, width=ports[1].bitsize),
            )
    width = parameters.get("bitsize", 3)
    if type(width) is not int:
        return None
    if names == ("ctrl", "q") and "unitary" not in parameters:
        for i in range(width):
            angle = -2 * pi * 2**i / (2**width + 1)
            for adjoint in (False, True):
                template = controlled_rz_program(angle, adjoint=adjoint)
                if matches_component_definition(record, entry, template):
                    return (
                        "CRz" + ("†" if adjoint else ""),
                        template,
                        rotation_gate_symbols("CRz", {"angle": angle}),
                    )
    if names == ("q",) and ports[0].shape_symbolic == (2,):
        for value in (0.468, *(-0.5 / 2**i for i in range(width))):
            template = controlled_z_power_program(value)
            if matches_component_definition(record, entry, template):
                return (
                    "CZPowGate",
                    template,
                    rotation_gate_symbols("CZPowGate", {"exponent": value}),
                )
    if names == ("ctrl", "q") and "unitary" in parameters:
        preparation, unitary = parameters["ctrl_state_prep"], parameters["unitary"]
        assert isinstance(preparation, dict) and isinstance(unitary, dict)
        root_width, root_exponent = preparation["bitsize"], unitary["exponent"]
        assert isinstance(root_width, int) and isinstance(root_exponent, (int, float))
        root = textbook_z_phase_estimation_program(root_width, root_exponent)
        for candidate in root.functions:
            if tuple(p.name for p in candidate.registers or ()) != names:
                continue
            template = replace(
                root,
                functions=(
                    replace(candidate, name="main"),
                    *(
                        f
                        for f in root.functions
                        if f.name not in {candidate.name, "main"}
                    ),
                ),
            )
            if matches_component_definition(record, entry, template):
                return (
                    "Controlled",
                    template,
                    window_symbols(template, width=root_width, exponent=root_exponent),
                )
    return None
