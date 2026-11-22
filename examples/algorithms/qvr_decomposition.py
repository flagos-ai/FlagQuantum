# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 variable rotation and cost phasing graphs.

"""Present complete variable rotation Programs through existing domain owners."""

from collections.abc import Mapping

from examples.algorithms.phase_gradient_decomposition import gradient_symbols
from flagquantum.compiler.registers.arithmetic.integer_square_component import (
    integer_square_program,
)
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.phase.cost_function_component import (
    cost_function_phase_program,
)
from flagquantum.compiler.registers.phase.qvr_component import (
    qvr_phase_gradient_program,
    qvr_zpow_program,
)
from flagquantum.compiler.registers.phase.qvr_phase_gradient import qvr_phase_format
from flagquantum.compiler.registers.phase.qvr_zpow import qvr_zpow_schedule
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_terms,
)
from flagquantum.compiler.registers.phase.scaled_phase_component import (
    scaled_phase_add_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.component_score import component_register_score
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


def qvr_program(
    operation: str, parameters: Mapping[str, object], *, via_gradient: bool = False
) -> ProgramRecord | None:
    """Preserve four concrete source-success factory roots and private costs."""
    if operation == "PhasingViaCostFunction":
        square, phase = parameters["cost_eval_oracle"], parameters["phase_oracle"]
        assert isinstance(square, dict) and isinstance(phase, dict)
        variable = qvr_program("QvrPhaseGradient" if via_gradient else "QvrZPow", phase)
        assert variable is not None
        return cost_function_phase_program(
            integer_square_program(square["bitsize"]),
            variable,
            integer_square_program(square["bitsize"], adjoint=True),
            cost_names=("result",),
        )
    if operation not in {"QvrZPow", "QvrPhaseGradient"}:
        return None
    port = parameters["cost_reg"]
    assert isinstance(port, dict)
    dtype = port["dtype"]
    cost = Register(
        port["name"],
        fixed_point_register_type(
            dtype["bitsize"], dtype["num_frac"], signed=dtype["signed"]
        ),
    )
    coefficient, epsilon = parameters["gamma"], parameters["eps"]
    assert isinstance(coefficient, (int, float)) and isinstance(epsilon, (int, float))
    return (
        qvr_phase_gradient_program
        if operation == "QvrPhaseGradient"
        else qvr_zpow_program
    )(cost, coefficient, epsilon)


def qvr_symbols(
    record: ProgramRecord, parameters: Mapping[str, object]
) -> RegisterWireSymbolSupplier:
    """Reuse source rotation and register glyphs over actual stored calls."""
    functions = {f.name: f for f in record.functions}
    rotations: dict[str, float] = {}
    if "cost_reg" in parameters and len(record.functions[0].registers or ()) == 1:
        port = parameters["cost_reg"]
        assert isinstance(port, dict)
        dtype = port["dtype"]
        coefficient, epsilon = parameters["gamma"], parameters["eps"]
        assert isinstance(coefficient, (int, float)) and isinstance(
            epsilon, (int, float)
        )
        schedule = qvr_zpow_schedule(
            tuple(range(dtype["bitsize"])),
            cost_fraction_bits=dtype["num_frac"],
            coefficient=coefficient,
            epsilon=epsilon,
            signed=dtype["signed"],
        )
        rotations = {
            f"power_{i}": exponent for i, (_, exponent, _) in enumerate(schedule)
        }

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if step.name in rotations:
            return rotation_wire_symbol(
                step,
                register,
                index,
                axis="z",
                parameter=rotations[step.name],
                power=True,
            )
        ports = functions[step.name].registers or ()
        names = tuple(r.name for r in ports)
        if names == ("reg", "reg") or names == ("reg",):
            return split_join_wire_symbol(step, register, index)
        if register is None:
            if names == ("a", "result"):
                label = "a^2"
            elif names == ("x", "phase_grad"):
                label = "AddScaledValIntoPhaseReg"
            else:
                label = "QvrPhaseGradient" if "phase_grad" in names else "QvrZPow"
            return "Text", {"text": label, "fontsize": 10}
        if names == ("x", "phase_grad") and register.name == "x":
            return "TextBox", {"text": "AddScaledValIntoPhaseReg"}
        if (
            names in {("result",), ("result", "phase_grad")}
            and register.name == "result"
        ):
            return "TextBox", {
                "text": "QvrPhaseGradient" if "phase_grad" in names else "QvrZPow"
            }
        return partition_wire_symbol(step, register, index)

    return supply


def qvr_score(
    operation: str, parameters: Mapping[str, object], *, via_gradient: bool = False
) -> MusicalScoreData | None:
    record = qvr_program(operation, parameters, via_gradient=via_gradient)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record),
        register_wire_symbol=qvr_symbols(record, parameters),
    )


def qvr_child(
    record: ProgramRecord, entry: str, parameters: Mapping[str, object]
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Require identical reachable definitions before assigning source classes."""
    phase = parameters.get("phase_oracle", parameters)
    assert isinstance(phase, Mapping)
    if "cost_reg" not in phase:
        return None
    function = next(f for f in record.functions if f.name == entry)
    ports = function.registers or ()
    names = tuple(r.name for r in ports)
    cost = phase["cost_reg"]
    assert isinstance(cost, dict)
    for label in ("QvrZPow", "QvrPhaseGradient"):
        expected = (
            (cost["name"], "phase_grad")
            if label == "QvrPhaseGradient"
            else (cost["name"],)
        )
        if names != expected:
            continue
        template = qvr_program(label, phase)
        assert template is not None
        if matches_component_definition(record, entry, template):
            return label, template, qvr_symbols(template, phase)
    if names != ("x", "phase_grad"):
        return None
    coefficient, epsilon = phase["gamma"], phase["eps"]
    assert isinstance(coefficient, (int, float)) and isinstance(epsilon, (int, float))
    fraction = dict(ports[0].dtype.parameters)["fractional_width"]
    assert isinstance(fraction, int)
    fmt = qvr_phase_format(ports[0].bitsize, fraction, coefficient, epsilon)
    template = scaled_phase_add_program(
        ports[0].bitsize,
        fraction,
        fmt.gradient_width,
        coefficient,
        fmt.coefficient_width,
        fmt.coefficient_fraction_bits,
    )
    if not matches_component_definition(record, entry, template):
        return None
    additions = tuple(
        (shift, -1 if coefficient < 0 else 1)
        for _, shift in scaled_phase_add_terms(
            ports[0].bitsize,
            fraction,
            fmt.gradient_width,
            coefficient,
            fmt.coefficient_width,
            fmt.coefficient_fraction_bits,
        )
    )
    return (
        "AddScaledValIntoPhaseReg",
        template,
        gradient_symbols(template, additions=additions),
    )
