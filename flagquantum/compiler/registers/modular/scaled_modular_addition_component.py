# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_addition.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py

"""Compose the source MSB-first scaled additions and measured conjunction erase."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..control.conjunction_program import and_component_program
from .modular_addition import modular_add_requirements
from .modular_constant_component import controlled_modular_constant_program


def scaled_modular_addition_program(
    width: int, modulus: int, scale: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Preserve ctrl/x and add scale*x modulo modulus into y on control one.

    All words are MSB first. Each source bit owns an And output until the
    matching measured inverse. The loaded constant for each bit is reduced
    modulo modulus exactly at the source call site. Source domain failures
    remain observable. No cross-bit regrouping changes the call sequence.
    Adjoints reverse logical calls while retaining their original constants.
    """
    if type(adjoint) is not bool:
        raise ValueError("Scaled modular direction requires a boolean")
    if type(scale) is not int:
        raise ValueError("Modular scale must be an integer")
    modular_add_requirements(width)
    if type(modulus) is not int or not 1 <= modulus <= 1 << width:
        raise ValueError("Modulus must be a positive integer fitting the operand")
    word = unsigned_register_type(width)
    compute = and_component_program((1, 1), name="scale_and")
    erase = and_component_program((1, 1), name="scale_erase", uncompute=True)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(ctrl=quantum_bit_type(), x=word, y=word)
    )
    x = builder.inputs["x"]
    assert isinstance(x, RegisterBinding)
    xs = builder.split(x)
    ctrl, y = builder.inputs["ctrl"], builder.inputs["y"]
    pairs: list[tuple[ProgramRecord, ProgramRecord]] = [
        (compute, erase),
        (erase, compute),
    ]
    additions: dict[int, str] = {}
    for i in range(width):
        result = builder.call_program(
            compute, {"ctrl": np.array([ctrl, xs[i]])}, entry="scale_and"
        )
        constant = (scale * 2 ** (width - 1 - i)) % modulus
        if constant not in additions:
            addition = controlled_modular_constant_program(width, modulus, constant)
            additions[constant] = builder.import_program(addition)
            if adjoint:
                pairs.append(
                    (
                        addition,
                        controlled_modular_constant_program(
                            width, modulus, constant, adjoint=True
                        ),
                    )
                )
        added = builder.call_dict(
            additions[constant], {"ctrl": result["target"], "x": y}
        )
        restored = builder.call_program(
            erase,
            {"ctrl": result["ctrl"], "target": added["ctrl"]},
            entry="scale_erase",
        )
        ctrl, xs[i] = np.asarray(restored["ctrl"])
        y = added["x"]
    record = builder.finalize({"ctrl": ctrl, "x": builder.join(xs, word), "y": y})
    return adjoint_component_calls(record, tuple(pairs)) if adjoint else record
