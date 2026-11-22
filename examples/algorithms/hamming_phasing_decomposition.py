# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 Hamming and variable phase drawings.

"""Present stored Hamming phase calls through their existing native owners."""

from collections.abc import Mapping

from examples.algorithms.qvr_decomposition import qvr_child
from flagquantum.compiler.registers.bits.hamming_weight_component import (
    hamming_weight_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.phase.hamming_phasing_component import (
    hamming_phasing_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.conjunction_symbols import and_wire_symbol
from flagquantum.drawer.controlled_not_symbols import controlled_not_wire_symbol
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


def hamming_phase_program(
    operation: str, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    if operation not in {
        "HammingWeightPhasing",
        "HammingWeightPhasingViaPhaseGradient",
    }:
        return None
    width, exponent, epsilon = (
        parameters["bitsize"],
        parameters["exponent"],
        parameters["eps"],
    )
    assert (
        isinstance(width, int)
        and isinstance(exponent, (int, float))
        and isinstance(epsilon, (int, float))
    )
    return hamming_phasing_program(
        width, exponent, epsilon, via_gradient=operation.endswith("ViaPhaseGradient")
    )


def hamming_symbols(
    record: ProgramRecord, *, exponent: float = 0
) -> RegisterWireSymbolSupplier:
    functions = {function.name: function for function in record.functions}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        function = functions[step.name]
        ports = function.registers or ()
        names = tuple(port.name for port in ports)
        if step.name.startswith("weight_phase_"):
            position = int(step.name.removeprefix("weight_phase_"))
            return rotation_wire_symbol(
                step,
                register,
                index,
                axis="z",
                parameter=2**position * exponent,
                power=True,
            )
        if names == ("x", "junk", "out"):
            inverse = ports[1].side == RegisterSide.LEFT
            if register is None:
                symbol: tuple[SymbolClass, dict[str, str | int | bool]] = "Text", {
                    "text": "HammingWeightCompute",
                    "fontsize": 10,
                }
            elif register.name == "x":
                symbol = "TextBox", {"text": "HammingWeightCompute"}
            else:
                symbol = partition_wire_symbol(
                    step, register.adjoint() if inverse else register, index
                )
            return adjoint_wire_symbol(*symbol) if inverse else symbol
        if names == ("out", "phase_grad"):
            if register is None:
                return "Text", {"text": "QvrPhaseGradient", "fontsize": 10}
            if register.name == "out":
                return "TextBox", {"text": "QvrPhaseGradient"}
            return partition_wire_symbol(step, register, index)
        if names == ("ctrl", "target"):
            return (
                and_wire_symbol
                if ports[0].shape_symbolic
                else controlled_not_wire_symbol
            )(step, register, index)
        if names == ("reg",):
            if register is None:
                return "Text", {"text": "", "fontsize": 10}
            text = "alloc" if ports[0].side == RegisterSide.RIGHT else "free"
            symbol_name: SymbolClass = (
                "LarrowTextBox"
                if ports[0].side == RegisterSide.RIGHT
                else "RarrowTextBox"
            )
            return symbol_name, {"text": text}
        if names == ("reg", "reg") and not any(port.shape_symbolic for port in ports):
            if register is None:
                return "Text", {"text": "Cast", "fontsize": 10}
            return partition_wire_symbol(step, register, index)
        return split_join_wire_symbol(step, register, index)

    return supply


def hamming_phase_score(
    operation: str, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    record = hamming_phase_program(operation, parameters)
    if record is None:
        return None
    exponent = parameters["exponent"]
    assert isinstance(exponent, (int, float))
    return register_score_data(
        component_register_score(record),
        register_wire_symbol=hamming_symbols(record, exponent=exponent),
    )


def hamming_phase_child(
    record: ProgramRecord, entry: str, parameters: Mapping[str, object]
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    if (
        "cost_reg" in parameters
        or not {"bitsize", "exponent", "eps"} <= parameters.keys()
    ):
        return None
    function = next(function for function in record.functions if function.name == entry)
    ports = function.registers or ()
    if tuple(port.name for port in ports) == ("x", "junk", "out"):
        inverse = ports[1].side == RegisterSide.LEFT
        template = hamming_weight_program(ports[0].bitsize, adjoint=inverse)
        if matches_component_definition(record, entry, template):
            return (
                "HammingWeightCompute" + ("†" if inverse else ""),
                template,
                hamming_symbols(template),
            )
    if (
        tuple(port.name for port in ports)
        not in {("out", "phase_grad"), ("x", "phase_grad")}
        or ports[0].dtype.encoding != "fixed_point"
    ):
        return None
    width, exponent = parameters["bitsize"], parameters["exponent"]
    assert isinstance(width, int) and isinstance(exponent, (int, float))
    if (
        ports[0].bitsize != width.bit_length()
        or dict(ports[0].dtype.parameters)["fractional_width"] != 0
    ):
        return None
    qvr = {
        "cost_reg": {
            "name": "out",
            "dtype": {"bitsize": width.bit_length(), "num_frac": 0, "signed": False},
        },
        "gamma": exponent / 2,
        "eps": parameters["eps"],
    }
    return qvr_child(record, entry, qvr)
