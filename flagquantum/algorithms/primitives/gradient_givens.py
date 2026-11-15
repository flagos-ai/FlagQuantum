# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 givens_bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/quad_fermion/givens_bloq.py

"""Bind real and complex phase-gradient Givens rotations."""

from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.arithmetic.arithmetic_resource_model import source_add_cost
from ...compiler.registers.phase.gradient_givens import gradient_givens_ir
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ...core.ir.program.signatures import Signature
from ._block_binding import _BoundBlockOperation


def gradient_givens_signature(
    width: int, *, complex_rotation: bool = False
) -> Signature:
    """Declare the source's ordered through ports and unsigned fractional words."""
    if type(width) is not int or width < 1:
        raise ValueError("Givens requires a positive integer fraction width")
    fraction = fixed_point_register_type(width, width)
    angles = (
        {"real_rom_data": fraction, "cplx_rom_data": fraction}
        if complex_rotation
        else {"rom_data": fraction}
    )
    return Signature.build_from_dtypes(
        target_i=quantum_bit_type(),
        target_j=quantum_bit_type(),
        **angles,
        phase_gradient=fraction,
    )


def gradient_givens_on(
    i: int,
    j: int,
    real_angle: tuple[int, ...],
    gradient: tuple[int, ...],
    *,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    complex_angle: tuple[int, ...] | None = None,
    adjoint: bool = False,
) -> _BoundBlockOperation:
    """Bind the rotation and reusable clean carries for circuit.append."""
    gates = gradient_givens_ir(
        i,
        j,
        real_angle,
        gradient,
        workspace,
        classical_bits,
        complex_angle=complex_angle,
        adjoint=adjoint,
    )
    wire_count = (
        max((i, j, *real_angle, *(complex_angle or ()), *gradient, *workspace)) + 1
    )
    return _BoundBlockOperation(gates, wire_count)


def gradient_givens_cost(
    width: int, *, complex_rotation: bool = False, adjoint: bool = False
) -> ComponentCost:
    """Return the source call graph with shared phase-gradient work lifetimes."""
    ports = int(gradient_givens_signature(width).n_qubits())
    rz_ports = 1 + 2 * width
    addition = ComponentCost(
        "ControlledAddOrSubtract",
        rz_ports,
        GateCost(clifford=2 * width + 2 * (not adjoint)),
        (CostCall(source_add_cost(width, width), bystander_qubits=1),),
    )
    rz = ComponentCost("RzViaPhaseGradient", rz_ports, calls=(CostCall(addition),))
    cliffords = (
        ("XGate", 1, 2),
        ("SGate", 1, 1 if adjoint else 2),
        ("SGate†", 1, 2 if adjoint else 1),
        ("Hadamard", 1, 2),
        ("CNOT", 2, 5),
    )
    real = ComponentCost(
        "RealGivensRotationByPhaseGradient",
        ports,
        calls=(
            *(
                CostCall(ComponentCost(name, size, GateCost(clifford=1)), count)
                for name, size, count in cliffords
            ),
            CostCall(rz, 2, ports - rz_ports),
        ),
    )
    if not complex_rotation:
        return real
    return ComponentCost(
        "ComplexGivensRotationByPhaseGradient",
        ports + width,
        calls=(
            CostCall(real, bystander_qubits=width),
            CostCall(rz, bystander_qubits=ports + width - rz_ports),
        ),
    )
