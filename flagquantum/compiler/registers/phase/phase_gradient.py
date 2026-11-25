# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PhaseGradientUnitary, PhaseGradientState and CZPowGate.
# Sources:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/phase_gradient.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/rotation.py

"""Phase-gradient unitaries and state preparation in FlagQuantum IR."""

from math import isfinite, pi
from typing import cast

from ....core.ir import Instruction
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ....core.value_parameters import is_symbolic_value
from ..._component_trace.resource_classification import classify_resource_gate
from ..._component_trace.resource_generalization import (
    ControlledZPowerCall,
    RotationCall,
)
from ..._component_trace.resource_model import ComponentCost, CostCall, GateCost
from ..._component_trace.symbolic_math import SymbolicInt, SymbolicReal, smax
from ..bits.bitwise import _validate_wires
from ..control.control_predicate_resource_model import control_predicate_cost
from ..primitives.controlled_rotation import controlled_z_power_ir
from ..resources.symbolic_resource_model import SymbolicResourceFormula


def source_phase_gradient_unitary_signature(
    width: SymbolicInt, *, controlled: bool = False
) -> Signature:
    """Declare through fractional gradient and optional control ports."""
    ports: tuple[Register, ...] = (
        (Register("ctrl", quantum_bit_type()),) if controlled else ()
    )
    if width != 0:
        ports += (Register("phase_grad", fixed_point_register_type(width, width)),)
    return Signature(ports)


def source_phase_gradient_state_signature(width: SymbolicInt) -> Signature:
    """Declare the prepared right-only fractional register."""
    return Signature(
        (
            Register(
                "phase_grad",
                fixed_point_register_type(width, width),
                side=RegisterSide.RIGHT,
            ),
        )
    )


def source_phase_gradient_unitary_calls(
    width: SymbolicInt,
    exponent: SymbolicReal = 1,
    *,
    controlled: bool = False,
    epsilon: SymbolicReal = 1e-10,
) -> dict[RotationCall | ControlledZPowerCall, SymbolicInt]:
    """Preserve source phase keys, precision division and dictionary coalescing."""
    concrete = isinstance(width, int)
    if concrete and width == 0:
        return {}
    if concrete and width < 0:
        raise ValueError("Phase-gradient width must be nonnegative")
    budget = epsilon / width
    powers = (
        (exponent / 2**i for i in range(width)) if concrete else (exponent / 2**width,)
    )
    return {
        (
            ControlledZPowerCall(power, budget)
            if controlled
            else RotationCall("zpow", power, budget)
        ): (1 if concrete else width)
        for power in powers
    }


def source_phase_gradient_unitary_cost(
    width: int,
    exponent: SymbolicReal = 1,
    *,
    controlled: bool = False,
    epsilon: float = 1e-10,
) -> ComponentCost:
    """Model the source's distinct Z or CZ power declarations."""
    if (
        type(width) is not int
        or width < 1
        or (not is_symbolic_value(exponent) and not isfinite(exponent))
    ):
        raise ValueError(
            "Phase-gradient cost requires positive width and finite exponent"
        )
    logical_width = width + 2 * controlled
    calls = []
    declared = source_phase_gradient_unitary_calls(
        width, exponent, controlled=controlled, epsilon=epsilon
    )
    for call in declared:
        power = (
            call.exponent if isinstance(call, ControlledZPowerCall) else call.parameter
        )
        family = classify_resource_gate(RotationCall("zpow", power, call.precision))
        rotation = ComponentCost(
            f"ZPowGate({power})",
            1,
            GateCost(
                t=int(family == "t"),
                clifford=int(family == "clifford"),
                rotation=int(family == "rotation"),
            ),
        )
        if controlled:
            rotation = ComponentCost(
                f"CZPowGate({power})",
                3,
                calls=(
                    CostCall(control_predicate_cost((1, 1))),
                    CostCall(rotation, bystander_qubits=2),
                    CostCall(control_predicate_cost((1, 1), uncompute=True)),
                ),
            )
        calls.append(
            CostCall(rotation, bystander_qubits=logical_width - rotation.qubits)
        )
    return ComponentCost("PhaseGradientUnitary", logical_width, calls=tuple(calls))


def source_phase_gradient_state_cost(
    width: int, exponent: SymbolicReal = -1
) -> ComponentCost:
    """Model state preparation as a Hadamard layer followed by the unitary."""
    unitary = source_phase_gradient_unitary_cost(width, exponent)
    return ComponentCost(
        "PhaseGradientState",
        width,
        GateCost(clifford=width),
        calls=(CostCall(unitary),),
    )


def source_symbolic_phase_gradient_cost(
    width: SymbolicInt, *, controlled: bool = False
) -> SymbolicResourceFormula:
    """Use the source's single symbolic rotation key and its multiplicity."""
    controlled_calls = width if controlled else 0
    return SymbolicResourceFormula(
        and_bloq=controlled_calls,
        clifford=controlled_calls,
        measurement=controlled_calls,
        rotation=width,
        logical_qubits=cast(
            SymbolicInt, smax(3, width + 1) if controlled else smax(1, width)
        ),
    )


def phase_gradient_unitary_ir(
    gradient: tuple[int, ...],
    *,
    exponent: float = 1.0,
    control: int | None = None,
    work: int | None = None,
    classical_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Apply the source Z powers on a big-endian gradient register.

    The positive-control path implements each CZ power with a logical And,
    a phase on its target, and measured uncomputation. Supply one clean work
    qubit and one classical slot, reused for the full register. Negate the
    exponent for the adjoint. Angles are emitted directly in radians.
    """
    if type(exponent) not in (int, float) or not isfinite(pi * exponent):
        raise ValueError("Phase gradient requires a finite phase angle")
    if control is None:
        if work is not None or classical_bit is not None:
            raise ValueError("Uncontrolled phase gradient requires no workspace")
        _validate_wires(gradient)
        return tuple(
            Instruction("phase", (wire,), params={"theta": pi * (exponent / (1 << i))})
            for i, wire in enumerate(gradient)
        )
    if work is None or classical_bit is None:
        raise ValueError("Controlled phase gradient requires work and a classical slot")
    _validate_wires((*gradient, control, work))
    return tuple(
        gate
        for i, wire in enumerate(gradient)
        for gate in controlled_z_power_ir(
            control, wire, exponent / (1 << i), work, classical_bit
        )
    )


def phase_gradient_state_ir(
    gradient: tuple[int, ...], *, inverse: bool = False, exponent: float = 1.0
) -> tuple[Instruction, ...]:
    """Prepare or unprepare a phase-gradient state on caller-owned qubits.

    The existing FlagQuantum default prepares the positive exponent. Pass
    exponent=-1 for the pinned source state's default. The forward input is
    zero; the inverse of the matching state restores that input.
    """
    phases = phase_gradient_unitary_ir(
        gradient, exponent=-exponent if inverse else exponent
    )
    hadamards = tuple(Instruction("h", (wire,)) for wire in gradient)
    return (*phases, *hadamards) if inverse else (*hadamards, *phases)
