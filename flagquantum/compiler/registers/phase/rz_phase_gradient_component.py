# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/rz_via_phase_gradient.py.

"""Compose fixed-point Rz kickback through shared casts and conditional addition."""

import sympy

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..arithmetic.conditional_addition_component import conditional_addition_program
from ..component_builder import ComponentProgramBuilder
from ..layout.component_cast import cast_word_function
from ..layout.register_cast import cast_signature


def rz_via_phase_gradient_program(
    angle_dtype: RegisterType,
    gradient_dtype: RegisterType,
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain q, angle and phase_grad in source order with MSB-first word views.

    The shared conditional Add/Sub owns and erases carry work. Casts preserve
    both fixed-point descriptors and all raw bits. The inverse exchanges the
    add/subtract control sectors through the existing exact arithmetic owner.
    """
    if type(adjoint) is not bool:
        raise ValueError("Rz phase gradient adjoint must be boolean")
    angle_int = unsigned_register_type(angle_dtype.num_qubits)
    phase_int = unsigned_register_type(gradient_dtype.num_qubits)

    def cast_view(
        name: str, source: RegisterType, target: RegisterType
    ) -> FunctionRecord:
        signature = cast_signature(source, target)
        # A raw Cast needs known wire dimensions, not a numerical value for
        # fractional metadata. Proposal 130 already admits these storage views.
        if any(
            isinstance(dtype.quantum_bits, sympy.Expr)
            and bool(dtype.quantum_bits.free_symbols)
            for dtype in (source, target)
        ):
            return FunctionRecord(name, (), registers=tuple(signature))
        return cast_word_function(name, source, target)

    casts = (
        cast_view("angle_to_int", angle_dtype, angle_int),
        cast_view("phase_to_int", gradient_dtype, phase_int),
        cast_view("angle_from_int", angle_int, angle_dtype),
        cast_view("phase_from_int", phase_int, gradient_dtype),
    )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(
            q=quantum_bit_type(), angle=angle_dtype, phase_grad=gradient_dtype
        ),
        casts,
    )
    angle = builder.call_dict("angle_to_int", {"reg": builder.inputs["angle"]})["reg"]
    phase = builder.call_dict("phase_to_int", {"reg": builder.inputs["phase_grad"]})[
        "reg"
    ]
    addition = (
        ProgramRecord(
            (
                FunctionRecord(
                    "main",
                    (),
                    registers=tuple(
                        Signature.build_from_dtypes(
                            ctrl=quantum_bit_type(), a=angle_int, b=phase_int
                        )
                    ),
                ),
            )
        )
        if angle_int.is_symbolic() or phase_int.is_symbolic()
        else conditional_addition_program(
            angle_int, phase_int, add_when_control_is_on=not adjoint
        )
    )
    result = builder.call_program(
        addition,
        {"ctrl": builder.inputs["q"], "a": angle, "b": phase},
    )
    angle = builder.call_dict("angle_from_int", {"reg": result["a"]})["reg"]
    phase = builder.call_dict("phase_from_int", {"reg": result["b"]})["reg"]
    return builder.finalize({"q": result["ctrl"], "angle": angle, "phase_grad": phase})
