# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SingleWindowModMul.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Compose one Montgomery window and its specialized measurement-assisted inverse."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..arithmetic.addition_component import addition_component_program
from ..arithmetic.controlled_addition_component import controlled_addition_program
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from ..lookup.clean_lookup_component import clean_lookup_program
from .single_window_mod_mul import single_window_qroam_table


def single_window_program(
    width: int, window: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Accumulate a little-endian input window into an MSB-first target array.

    Forward index starts zero and records the low accumulation bits. Source
    wraparound and target rotation remain explicit. Lookup banks and arithmetic
    work are owned internally. Lookup erasure uses the shared executable
    measurement instrument and has its own primitive resource count.
    """
    table = single_window_qroam_table(width, window, modulus)
    if type(adjoint) is not bool:
        raise ValueError("Montgomery window direction requires a boolean")
    bit = quantum_bit_type()
    word = montgomery_register_type(width)
    index_type = montgomery_register_type(window)
    total_type = montgomery_register_type(width + window)
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("x", bit, shape=(window,)),
                Register("y", montgomery_register_type(width, modulus)),
                Register("target", bit, shape=(width + window,)),
                Register("qrom_index", montgomery_register_type(window, modulus)),
            )
        )
    )
    x = np.asarray(builder.inputs["x"]).copy()
    target = np.asarray(builder.inputs["target"]).copy()
    y, index = builder.inputs["y"], builder.inputs["qrom_index"]
    cadd = controlled_addition_program(
        word, montgomery_register_type(width + 1), adjoint=adjoint
    )
    xor = bitwise_word_program(index_type, "xor")
    add = addition_component_program(
        width + window, left_dtype=total_type, right_dtype=total_type, adjoint=adjoint
    )
    if adjoint:
        target = np.roll(target, -window)
    else:
        for i in range(window):
            start, end = window - 1 - i, width + window - i
            added = builder.call_program(
                cadd,
                {"ctrl": x[i], "a": y, "b": builder.join(target[start:end])},
            )
            x[i], y = added["ctrl"], added["a"]
            value = added["b"]
            assert isinstance(value, RegisterBinding)
            target[start:end] = builder.split(value)
        copied = builder.call_program(
            xor, {"x": builder.join(target[-window:], index_type), "y": index}
        )
        value = copied["x"]
        assert isinstance(value, RegisterBinding)
        target[-window:] = builder.split(value)
        index = copied["y"]
    loaded = builder.call_program(
        clean_lookup_program(table, width + window), {"selection": index}
    )
    corrected = builder.call_program(
        add, {"a": loaded["target0_"], "b": builder.join(target)}
    )
    loaded["target0_"] = corrected["a"]
    erased = builder.call_program(
        clean_lookup_program(table, width + window, erase=True), loaded
    )
    index = erased["selection"]
    value = corrected["b"]
    assert isinstance(value, RegisterBinding)
    target = builder.split(value)
    if adjoint:
        copied = builder.call_program(
            xor, {"x": builder.join(target[-window:], index_type), "y": index}
        )
        value = copied["x"]
        assert isinstance(value, RegisterBinding)
        target[-window:] = builder.split(value)
        index = copied["y"]
        for i in reversed(range(window)):
            start, end = window - 1 - i, width + window - i
            added = builder.call_program(
                cadd,
                {"ctrl": x[i], "a": y, "b": builder.join(target[start:end])},
            )
            x[i], y = added["ctrl"], added["a"]
            value = added["b"]
            assert isinstance(value, RegisterBinding)
            target[start:end] = builder.split(value)
    else:
        target = np.roll(target, window)
    return builder.finalize({"x": x, "y": y, "target": target, "qrom_index": index})
