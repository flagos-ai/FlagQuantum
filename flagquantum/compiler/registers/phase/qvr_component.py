# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/quantum_variable_rotation.py.

"""Compose variable rotations from shared phase powers and scaled addition."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_compatibility import check_register_types_consistent
from ....core.ir.program.register_types import fixed_point_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_builder import ComponentProgramBuilder
from ..primitives.pauli_rotation import RotationAngle
from ..primitives.pauli_rotation_component import z_power_function
from .phase_addition import source_phase_add_signature
from .qvr_phase_gradient import qvr_phase_format
from .qvr_zpow import qvr_zpow_schedule
from .scaled_phase_addition import scaled_phase_add_terms
from .scaled_phase_component import scaled_phase_add_program


def qvr_zpow_program(
    cost: Register, coefficient: float, epsilon: float, *, adjoint: bool = False
) -> ProgramRecord:
    """Retain source-ordered phases on an MSB-first through cost word.

    The existing schedule owns precision allocation and signed-index admission.
    Phases retain their literal angles; synthesis costs remain separate.
    """
    parameters = dict(cost.dtype.parameters)
    width = cost.bitsize
    fraction, signed = parameters["fractional_width"], parameters["signed"]
    assert (
        isinstance(width, int)
        and isinstance(fraction, int)
        and isinstance(signed, bool)
    )
    schedule = qvr_zpow_schedule(
        tuple(range(width)),
        cost_fraction_bits=fraction,
        coefficient=coefficient,
        epsilon=epsilon,
        signed=signed,
        adjoint=adjoint,
    )
    return qvr_phase_power_program(
        cost, tuple((wire, exponent) for wire, exponent, _ in schedule)
    )


def qvr_phase_power_program(
    cost: Register, schedule: tuple[tuple[int, RotationAngle], ...]
) -> ProgramRecord:
    """Apply an owned finite phase schedule to a named cost word, MSB first.

    Numerical and source-expression schedules share this typed storage owner.
    Formats converts external symbolic scalars before entering the register layer.
    """
    phases = tuple(
        z_power_function(f"power_{i}", exponent)
        for i, (_, exponent) in enumerate(schedule)
    )
    builder = ComponentProgramBuilder(Signature((cost,)), phases)
    word = builder.inputs[cost.name]
    assert isinstance(word, RegisterBinding)
    bits = builder.split(word)
    for function, (wire, _) in zip(phases, schedule, strict=True):
        bits[wire] = builder.call_dict(function.name, {"q": bits[wire]})["q"]
    return builder.finalize({cost.name: builder.join(bits, cost.dtype)})


def qvr_phase_gradient_program(
    cost: Register, coefficient: float, epsilon: float
) -> ProgramRecord:
    """Call the source's scaled addition with its existing concrete sizing.

    The caller owns the through gradient. Each shared addition owns and erases
    its carry workspace. Source signed and unresolved sizing refusals remain.
    """
    parameters = dict(cost.dtype.parameters)
    if parameters["signed"]:
        raise ValueError("Gradient QVR requires an unsigned cost word")
    fraction, width = parameters["fractional_width"], cost.bitsize
    assert isinstance(width, int) and isinstance(fraction, int)
    fmt = qvr_phase_format(width, fraction, coefficient, epsilon)
    terms = scaled_phase_add_terms(
        width,
        fraction,
        fmt.gradient_width,
        coefficient,
        fmt.coefficient_width,
        fmt.coefficient_fraction_bits,
    )
    indices = tuple(range(width))
    first_nonempty = next((part for part, _ in terms if indices[part]), None)
    whole_word_mismatch = (
        first_nonempty is not None
        and indices[first_nonempty] == indices
        and not check_register_types_consistent(
            cost.dtype, fixed_point_register_type(width, width)
        )
    )
    invalid_slice = any(part.stop < part.start for part, _ in terms)
    if whole_word_mismatch or invalid_slice:
        # Original exact-word lookup keeps the nonfractional input dtype.
        # QVR's root succeeds, while this scaled child's first Add refuses.
        addition = ProgramRecord(
            (
                FunctionRecord(
                    "main",
                    (),
                    registers=tuple(
                        source_phase_add_signature(
                            width, fmt.gradient_width, input_fraction_bits=fraction
                        )
                    ),
                ),
            )
        )
    else:
        addition = scaled_phase_add_program(
            width,
            fraction,
            fmt.gradient_width,
            coefficient,
            fmt.coefficient_width,
            fmt.coefficient_fraction_bits,
        )
    return qvr_phase_addition_program(cost, fmt.gradient_width, addition)


def qvr_phase_addition_program(
    cost: Register, gradient_width: SymbolicInt, addition: ProgramRecord
) -> ProgramRecord:
    """Thread the named cost and gradient through a supplied scaled addition.

    Unresolved source children retain their typed declaration. Numerical sizing,
    coefficient encoding and physical execution remain in existing owners.
    """
    builder = ComponentProgramBuilder(
        Signature(
            (
                cost,
                Register(
                    "phase_grad",
                    fixed_point_register_type(gradient_width, gradient_width),
                ),
            )
        )
    )
    result = builder.call_program(
        addition,
        {"x": builder.inputs[cost.name], "phase_grad": builder.inputs["phase_grad"]},
        entry=addition.functions[0].name,
    )
    return builder.finalize(
        {cost.name: result["x"], "phase_grad": result["phase_grad"]}
    )
