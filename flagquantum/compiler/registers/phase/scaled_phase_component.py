# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/phase_gradient.py.

"""Retain scaled phase slices and explicit native arithmetic below atomic leaves."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    fixed_point_register_type,
)
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.addition_component import addition_component_program
from ..arithmetic.controlled_addition_component import controlled_addition_program
from ..component_builder import ComponentProgramBuilder
from ..layout.component_cast import cast_word_function
from ..layout.component_word_views import gather_word_binding
from .phase_addition import source_phase_add_signature
from .scaled_phase_addition import scaled_phase_add_terms


def phase_add_component_program(
    input_width: int,
    gradient_width: int,
    *,
    right_shift: int = 0,
    sign: int = 1,
    controlled: bool = False,
    control_value: int = 1,
) -> ProgramRecord:
    """Implement the atomic source leaf using shared integer Add and its inverse.

    Fractional ports are MSB-first. Retained input bits update the gradient
    prefix. Shared Add owns and erases carry workspace. This graph is native
    execution below an original source decomposition refusal.
    An empty input omits x and retains the remaining through storage unchanged.
    """
    if type(right_shift) is not int or right_shift < 0:
        raise ValueError("Phase addition right shift must be a nonnegative integer")
    if type(sign) is not int:
        raise ValueError("Phase addition multiplier must be an integer")
    if (
        type(input_width) is not int
        or type(gradient_width) is not int
        or input_width < 0
        or gradient_width < 1
    ):
        raise ValueError(
            "Phase addition requires nonnegative input and positive gradient widths"
        )
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Phase addition control value must be zero or one")
    if type(controlled) is not bool:
        raise ValueError("Phase addition controlled mode must be boolean")
    builder = ComponentProgramBuilder(
        source_phase_add_signature(input_width, gradient_width, controlled=controlled)
    )
    if input_width == 0:
        return builder.finalize(builder.inputs)
    x, phase = builder.inputs["x"], builder.inputs["phase_grad"]
    assert isinstance(x, RegisterBinding) and isinstance(phase, RegisterBinding)
    active = min(input_width, max(0, gradient_width - right_shift))
    if not active:
        return builder.finalize(builder.inputs)
    prefix = active + right_shift
    magnitude = abs(sign) % (1 << prefix)
    if not magnitude:
        return builder.finalize(builder.inputs)
    bits, target = builder.split(x), builder.split(phase)
    control = builder.inputs.get("ctrl")
    for weight in range(magnitude.bit_length()):
        if not magnitude & (1 << weight):
            continue
        # Truncate the source once, before applying the integer multiplier.
        # A left shift discards high overflow bits, never the low bits already
        # discarded by the source fixed-point conversion.
        target_width = prefix - weight
        retained = min(active, target_width)
        start = active - retained
        left = builder.join(bits[start:active], RegisterType("uint", retained))
        right = builder.join(target[:target_width], RegisterType("uint", target_width))
        addition = (
            controlled_addition_program(
                RegisterType("uint", retained),
                RegisterType("uint", target_width),
                control_value=control_value,
                adjoint=sign < 0,
            )
            if controlled
            else addition_component_program(retained, target_width, adjoint=sign < 0)
        )
        operands = {"a": left, "b": right}
        if controlled:
            assert isinstance(control, RegisterBinding)
            operands["ctrl"] = control
        result = builder.call_program(addition, operands)
        a, b = result["a"], result["b"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
        bits[start:active], target[:target_width] = builder.split(a), builder.split(b)
        if controlled:
            control = result["ctrl"]
    outputs = {
        "x": builder.join(bits, fixed_point_register_type(input_width, input_width)),
        "phase_grad": builder.join(
            target, fixed_point_register_type(gradient_width, gradient_width)
        ),
    }
    if controlled:
        assert isinstance(control, RegisterBinding)
        outputs["ctrl"] = control
    return builder.finalize(outputs)


def scaled_phase_add_program(
    input_width: int,
    input_fraction_bits: int,
    gradient_width: int,
    coefficient: float,
    coefficient_width: int,
    coefficient_fraction_bits: int,
    *,
    input_signed: bool = False,
    coefficient_signed: bool = False,
) -> ProgramRecord:
    """Call each source slice in coefficient-bit order, preserving all other bits.

    The input fractional metadata is retained. Existing whole-word views are
    reused, with Split, Cast and Join emitted only when a new slice is needed.
    Each atomic addition owns its own clean carry lifetime.
    """
    terms = scaled_phase_add_terms(
        input_width,
        input_fraction_bits,
        gradient_width,
        coefficient,
        coefficient_width,
        coefficient_fraction_bits,
        coefficient_signed=coefficient_signed,
    )
    dtype = fixed_point_register_type(
        input_width, input_fraction_bits, signed=input_signed
    )
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", dtype),
                Register(
                    "phase_grad",
                    fixed_point_register_type(gradient_width, gradient_width),
                ),
            )
        )
    )
    x, phase = builder.inputs["x"], builder.inputs["phase_grad"]
    assert isinstance(x, RegisterBinding)
    indices = tuple(range(input_width))
    words = {indices: x}
    for part, shift in terms:
        wires = indices[part]
        width = len(wires)
        if width == 0:
            result = builder.call_program(
                phase_add_component_program(
                    0,
                    gradient_width,
                    right_shift=shift,
                    sign=-1 if coefficient < 0 else 1,
                ),
                {"phase_grad": phase},
            )
            phase = result["phase_grad"]
            continue
        word = gather_word_binding(
            builder, words, wires, fixed_point_register_type(width, width)
        )
        result = builder.call_program(
            phase_add_component_program(
                width,
                gradient_width,
                right_shift=shift,
                sign=-1 if coefficient < 0 else 1,
            ),
            {"x": word, "phase_grad": phase},
        )
        returned = result["x"]
        assert isinstance(returned, RegisterBinding)
        words[wires] = returned
        phase = result["phase_grad"]
    x = gather_word_binding(builder, words, indices, dtype)
    if x.register.dtype != dtype:
        # Source exact-word lookup can retain the last adder's fractional dtype.
        # Restore the declared storage view below that source refusal boundary.
        cast = cast_word_function("restore_cost_view", x.register.dtype, dtype)
        restored = builder.call_program(
            ProgramRecord((cast,)), {"reg": x}, entry=cast.name
        )["reg"]
        assert isinstance(restored, RegisterBinding)
        x = restored
    return builder.finalize({"x": x, "phase_grad": phase})
