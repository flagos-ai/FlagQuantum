# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIterationStep4.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Compose the paired exact controlled exchanges of one inverse iteration."""

from .....core.ir.program import ProgramRecord
from .....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from .....core.ir.program.signatures import Signature
from ...component_builder import ComponentProgramBuilder
from ...primitives.controlled_swap_component import controlled_swap_program


def inverse_exchange_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Preserve four MSB words and swap their pairs under the owned a flag.

    Each source register CSwap retains its exact seven-T bit decomposition.
    The inverse reverses the two child calls. Both directions use no workspace,
    measurement or new classical slots, and preserve arbitrary coherent inputs.
    """
    if type(adjoint) is not bool:
        raise ValueError("Inverse exchange direction requires a boolean")
    swap = controlled_swap_program(width)
    word = montgomery_register_type(width)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(
            u=word, v=word, r=word, s=word, a=quantum_bit_type()
        )
    )
    values = dict(builder.inputs)
    pairs = (("u", "v"), ("r", "s"))
    for left, right in reversed(pairs) if adjoint else pairs:
        result = builder.call_program(
            swap,
            {"ctrl": values["a"], "x": values[left], "y": values[right]},
        )
        values["a"], values[left], values[right] = (
            result["ctrl"],
            result["x"],
            result["y"],
        )
    return builder.finalize(values)
