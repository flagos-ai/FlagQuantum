# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 qft source files and wire symbols.

"""Present complete Fourier source graphs over existing typed Program owners."""

from collections.abc import Mapping
from math import pi

from examples.algorithms.rotation_gate_decomposition import rotation_gate_symbols
from flagquantum.algorithms.primitives.fermionic_fourier import (
    two_bit_fermionic_fourier_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.phase.approximate_qft_component import (
    approximate_qft_program,
)
from flagquantum.compiler.registers.phase.gradient_qft_component import (
    gradient_qft_program,
)
from flagquantum.compiler.registers.phase.phase_gradient_component import (
    phase_gradient_unitary_program,
)
from flagquantum.compiler.registers.phase.textbook_qft_component import (
    textbook_qft_program,
)
from flagquantum.compiler.registers.primitives.controlled_rotation_component import (
    controlled_z_power_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
from flagquantum.drawer.hadamard_symbols import hadamard_wire_symbol
from flagquantum.drawer.partition_symbols import (
    partition_wire_symbol,
    split_join_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)
from flagquantum.drawer.rotation_symbols import rotation_wire_symbol


def fourier_program(
    operation: str, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Retain concrete source-success roots and the original symbolic refusal."""
    if operation == "TwoBitFFFT":
        k, n, inverse = parameters["k"], parameters["n"], parameters["is_adjoint"]
        assert isinstance(k, int) and isinstance(n, int) and isinstance(inverse, bool)
        return two_bit_fermionic_fourier_program(k, n, adjoint=inverse)
    width = parameters.get("bitsize")
    if type(width) is not int:
        return None
    reverse = parameters.get("with_reverse", True)
    assert isinstance(reverse, bool)
    if operation == "QFTTextBook":
        return textbook_qft_program(width, with_reverse=reverse)
    if operation == "QFTPhaseGradient":
        return gradient_qft_program(width, with_reverse=reverse)
    if operation == "ApproximateQFT":
        precision = parameters["phase_bitsize"]
        assert isinstance(precision, int)
        return approximate_qft_program(width, precision, with_reverse=reverse)
    return None


def fourier_symbols(
    record: ProgramRecord, *, angle: float = 0, controlled_gradient: bool = False
) -> RegisterWireSymbolSupplier:
    functions = {f.name: f for f in record.functions}
    powers = [
        op.callee
        for op in record.functions[0].blocks[0].operations
        if op.callee is not None
        and tuple(p.name for p in functions[op.callee].registers or ()) == ("q",)
        and (functions[op.callee].registers or ())[0].shape_symbolic == (2,)
    ]
    exponents = {name: 0.5 / 2**i for i, name in enumerate(powers)}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        ports = functions[step.name].registers or ()
        names = tuple(p.name for p in ports)
        if step.name in {"hadamard", "fourier_h"}:
            return hadamard_wire_symbol(step, register, index)
        if step.name == "fourier_cx":
            return controlled_not_wire_symbol(step, register, index)
        if step.name == "fourier_rz":
            return rotation_wire_symbol(
                step, register, index, axis="z", parameter=angle
            )
        if step.name in {"fourier_s", "fourier_sdg", "fourier_t", "fourier_tdg"}:
            if register is None:
                return "Text", {"text": "", "fontsize": 10}
            label = step.name.removeprefix("fourier_").replace("dg", "†").upper()
            return "TextBox", {"text": label}
        if step.name == "swap":
            if register is None:
                return "Text", {"text": "TwoBitSwap", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        if (
            names == ("ctrl", "phase_grad")
            and functions[step.name].blocks[0].operations
        ):
            if register is None:
                return "Text", {"text": "PhaseGradientUnitary", "fontsize": 10}
            if register.name == "ctrl":
                return "Circle", {"filled": True}
            return "TextBox", {"text": "Z^0.5/2"}
        if controlled_gradient and step.name in exponents:
            if register is None:
                label = f"CZ**{exponents[step.name]}"
                return "Text", {
                    "text": label if len(label) <= 10 else label[:8] + "..",
                    "fontsize": 10,
                }
            return partition_wire_symbol(step, register, index)
        if names == ("q", "phase_grad"):
            if register is None:
                return "Text", {"text": "QFTPhaseGradient", "fontsize": 10}
            if register.name == "q":
                return "TextBox", {"text": "QFTPhaseGradient"}
            return partition_wire_symbol(step, register, index)
        if names == ("a", "b", "result"):
            if register is None:
                return "Text", {"text": "result += a*b", "fontsize": 10}
            if register.name == "result":
                return "TextBox", {"text": "c+=a*b"}
            return partition_wire_symbol(step, register, index)
        if names in {("ctrl", "x", "phase_grad"), ("ctrl", "phase_grad")}:
            if register is None:
                return "Text", {"text": "pg+=x>>1", "fontsize": 10}
            if register.name == "ctrl":
                return "TextBox", {"text": "AddIntoPhaseGrad"}
            return partition_wire_symbol(step, register, index)
        if names == ("reg", "reg") and not any(p.shape_symbolic for p in ports):
            if register is None:
                return "Text", {"text": "Cast", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        return split_join_wire_symbol(step, register, index)

    return supply


def fourier_score(
    operation: str, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    record = fourier_program(operation, parameters)
    if record is None:
        return None
    angle = 0.0
    if operation == "TwoBitFFFT":
        k, n = parameters["k"], parameters["n"]
        assert isinstance(k, int) and isinstance(n, int)
        angle = 2 * pi * k / n
    return register_score_data(
        component_register_score(record),
        register_wire_symbol=fourier_symbols(record, angle=angle),
    )


def fourier_child(
    record: ProgramRecord, entry: str, parameters: Mapping[str, object]
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    if "bitsize" not in parameters:
        return None
    function = next(f for f in record.functions if f.name == entry)
    ports = function.registers or ()
    names = tuple(p.name for p in ports)
    if names == ("ctrl", "phase_grad"):
        template = phase_gradient_unitary_program(
            ports[1].bitsize, 0.5, controlled=True
        )
        if matches_component_definition(record, entry, template):
            return (
                "PhaseGradientUnitary",
                template,
                fourier_symbols(template, controlled_gradient=True),
            )
    if names == ("q", "phase_grad"):
        template = gradient_qft_program(ports[0].bitsize, with_reverse=False)
        if matches_component_definition(record, entry, template):
            return "QFTPhaseGradient", template, fourier_symbols(template)
    if names == ("q",) and ports[0].shape_symbolic == (2,):
        width = parameters["bitsize"]
        if type(width) is int:
            for i in range(width):
                exponent = 0.5 / 2**i
                template = controlled_z_power_program(exponent)
                if matches_component_definition(record, entry, template):
                    return (
                        "CZPowGate",
                        template,
                        rotation_gate_symbols("CZPowGate", {"exponent": exponent}),
                    )
    return None
