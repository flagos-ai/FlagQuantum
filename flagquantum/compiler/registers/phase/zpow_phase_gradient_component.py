# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/zpow_via_phase_gradient.py.

"""Retain constant-load, addition and unload calls for gradient kickback."""

from math import isfinite

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    fixed_point_register_type,
    quantum_bit_type,
)
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from .phase_addition import source_phase_add_signature
from .scaled_phase_component import phase_add_component_program
from .zpow_phase_gradient import zpow_phase_gradient_constant


def zpow_via_phase_gradient_program(
    exponent: float, width: SymbolicInt, *, adjoint: bool = False
) -> ProgramRecord:
    """Thread one positive control through a clean angle word and its erasure.

    q and phase_grad stay live. Words use MSB-first fractional storage. The
    angle word and the shared Add's carry work start and end at zero. Symbolic
    widths retain the source-success root and original atomic child refusals,
    with no physical binding until concrete source parameters are supplied.
    """
    if type(adjoint) is not bool:
        raise ValueError("Z power phase gradient adjoint must be boolean")
    if type(width) is int:
        zpow_phase_gradient_constant(exponent, width)
    elif type(exponent) not in (int, float) or not isfinite(exponent):
        raise ValueError("Z power requires a finite exponent")
    dtype = fixed_point_register_type(width, width)
    if type(width) is int:
        load = controlled_bitwise_word_program(
            dtype, "xor_constant", pattern=zpow_phase_gradient_constant(exponent, width)
        )
        addition = phase_add_component_program(width, width, sign=-1 if adjoint else 1)
    else:
        load = ProgramRecord(
            (
                FunctionRecord(
                    "load",
                    (),
                    registers=tuple(
                        Signature.build_from_dtypes(ctrl=quantum_bit_type(), x=dtype)
                    ),
                ),
            )
        )
        addition = ProgramRecord(
            (
                FunctionRecord(
                    "add", (), registers=tuple(source_phase_add_signature(width, width))
                ),
            )
        )
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(q=quantum_bit_type(), phase_grad=dtype)
    )
    angle = builder.allocate(dtype=dtype)
    result = builder.call_program(
        load, {"ctrl": builder.inputs["q"], "x": angle}, entry=load.functions[0].name
    )
    summed = builder.call_program(
        addition,
        {"x": result["x"], "phase_grad": builder.inputs["phase_grad"]},
        entry=addition.functions[0].name,
    )
    result = builder.call_program(
        load, {"ctrl": result["ctrl"], "x": summed["x"]}, entry=load.functions[0].name
    )
    cleared_angle = result["x"]
    assert isinstance(cleared_angle, RegisterBinding)
    builder.free(cleared_angle)
    return builder.finalize({"q": result["ctrl"], "phase_grad": summed["phase_grad"]})
