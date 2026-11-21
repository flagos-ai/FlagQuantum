# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/phase_gradient.py.

"""Publish source-visible phase-gradient calls through their Compiler owners."""

from collections.abc import Mapping

from examples.algorithms.basic_gate_child_decomposition import child_decomposition
from flagquantum.compiler.registers.bits.on_each import on_each_word_program
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.phase.phase_gradient_component import (
    phase_gradient_state_program,
    phase_gradient_unitary_program,
)
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_terms,
)
from flagquantum.compiler.registers.phase.scaled_phase_component import (
    scaled_phase_add_program,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
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
    register_score_data,
)
from flagquantum.drawer.rotation_symbols import rotation_wire_symbol


def gradient_program(
    operation: str, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Retain the three concrete source graphs and both original refusals."""
    width = parameters.get("bitsize")
    if (
        operation in {"PhaseGradientUnitary", "PhaseGradientState"}
        and type(width) is int
    ):
        exponent = parameters["exponent"]
        assert isinstance(exponent, (int, float))
        return (
            phase_gradient_unitary_program
            if operation == "PhaseGradientUnitary"
            else phase_gradient_state_program
        )(width, exponent)
    if operation == "AddScaledValIntoPhaseReg":
        x, gamma = parameters["x_dtype"], parameters["gamma_dtype"]
        assert isinstance(x, dict) and isinstance(gamma, dict)
        target, coefficient = parameters["phase_bitsize"], parameters["gamma"]
        assert isinstance(target, int) and isinstance(coefficient, (int, float))
        return scaled_phase_add_program(
            x["bitsize"],
            x["num_frac"],
            target,
            coefficient,
            gamma["bitsize"],
            gamma["num_frac"],
        )
    return None


def gradient_symbols(
    record: ProgramRecord,
    *,
    exponent: float = 1,
    additions: tuple[tuple[int, int], ...] = (),
) -> RegisterWireSymbolSupplier:
    """Compose fresh source glyphs over the actual immediate stored calls."""
    functions = {f.name: f for f in record.functions}
    addition_labels = dict(
        zip(
            (
                f.name
                for f in record.functions[1:]
                if tuple(r.name for r in f.registers or ()) == ("x", "phase_grad")
            ),
            additions,
            strict=True,
        )
    )

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        function = functions[step.name]
        ports = function.registers or ()
        names = tuple(r.name for r in ports)
        if step.name.startswith("power_"):
            i = int(step.name.rsplit("_", 1)[1])
            return rotation_wire_symbol(
                step, register, index, axis="z", parameter=exponent / 2**i, power=True
            )
        if step.name == "hadamard":
            return hadamard_wire_symbol(step, register, index)
        if names == ("q",):
            return on_each_wire_symbol(
                step,
                register,
                index,
                width=ports[0].bitsize,
                child_symbol=hadamard_wire_symbol,
            )
        if names == ("phase_grad",):
            if register is None:
                return "Text", {"text": "PhaseGradientUnitary", "fontsize": 10}
            return "TextBox", {"text": f"Z^{exponent}/2"}
        if names == ("x", "phase_grad"):
            shift, sign = addition_labels[step.name]
            if register is None:
                label = "pg" + ("+" if sign > 0 else "-") + "=x"
                return "Text", {
                    "text": label + (f">>{shift}" if shift else ""),
                    "fontsize": 10,
                }
            if register.name == "x":
                return "TextBox", {"text": "AddIntoPhaseGrad"}
            return partition_wire_symbol(step, register, index)
        if names == ("reg",) and not function.blocks[0].arguments:
            return (
                ("Text", {"text": "", "fontsize": 10})
                if register is None
                else ("LarrowTextBox", {"text": "alloc"})
            )
        if names == ("reg", "reg") and all(not p.shape_symbolic for p in ports):
            if register is None:
                return "Text", {"text": "Cast", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        return split_join_wire_symbol(step, register, index)

    return supply


def gradient_score(
    operation: str, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Draw source-success roots without allocating simulator state."""
    record = gradient_program(operation, parameters)
    if record is None:
        return None
    exponent = parameters.get("exponent", 1)
    assert isinstance(exponent, (int, float))
    additions: tuple[tuple[int, int], ...] = ()
    if operation == "AddScaledValIntoPhaseReg":
        x, gamma = parameters["x_dtype"], parameters["gamma_dtype"]
        assert isinstance(x, dict) and isinstance(gamma, dict)
        coefficient, width = parameters["gamma"], parameters["phase_bitsize"]
        assert isinstance(coefficient, (int, float)) and isinstance(width, int)
        additions = tuple(
            (shift, -1 if coefficient < 0 else 1)
            for _, shift in scaled_phase_add_terms(
                x["bitsize"],
                x["num_frac"],
                width,
                coefficient,
                gamma["bitsize"],
                gamma["num_frac"],
            )
        )
    return register_score_data(
        component_register_score(record),
        register_wire_symbol=gradient_symbols(
            record, exponent=exponent, additions=additions
        ),
    )


def gradient_child(
    record: ProgramRecord, entry: str, root: object
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Qualify complete gradient and repeated-H bodies by literal matching."""
    function = next(f for f in record.functions if f.name == entry)
    ports = function.registers or ()
    if len(ports) == 1 and type(ports[0].bitsize) is int:
        width = ports[0].bitsize
        if ports[0].name == "phase_grad":
            template = phase_gradient_unitary_program(width, -1)
            if matches_component_definition(record, entry, template):
                return (
                    "PhaseGradientUnitary",
                    template,
                    gradient_symbols(template, exponent=-1),
                )
        if ports[0].name == "q" and not ports[0].shape_symbolic:
            h = bit_gate_function(
                "hadamard",
                (Register("q", quantum_bit_type()),),
                CircuitIR(1, (Instruction("h", (0,)),)),
            )
            template = on_each_word_program(width, h)
            if matches_component_definition(record, entry, template):
                return "OnEach", template, gradient_symbols(template)
    return child_decomposition(record, entry, root)
