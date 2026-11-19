# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PermutationCycle and Permutation.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/permutation.py

"""Compose bounded basis permutations from equality and controlled XOR calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    unsigned_register_type,
)
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..comparison.equality_component import equality_constant_program
from ..component_builder import BindingCollection, ComponentProgramBuilder


def _dtype(size: int, width: int) -> RegisterType:
    if (
        type(width) is not int
        or width < 1
        or type(size) is not int
        or not 1 <= size <= 1 << width
    ):
        raise ValueError("Permutation size must fit a positive unsigned width")
    return bounded_unsigned_register_type(width, size)


def permutation_cycle_program(
    size: int, cycle: tuple[int, ...], width: int
) -> ProgramRecord:
    """Apply one source cycle to a bounded MSB-first THRU word.

    Marker allocation belongs to the cycle. Equality owns its temporary carries.
    Marker and carries return clean. Singleton cycles retain their full source
    work. Invalid or empty cycles raise ValueError.
    """
    dtype = _dtype(size, width)
    if (
        not cycle
        or len(set(cycle)) != len(cycle)
        or any(type(value) is not int or not 0 <= value < size for value in cycle)
    ):
        raise ValueError("Permutation cycle requires distinct entries in range")
    builder = ComponentProgramBuilder(Signature.build_from_dtypes(x=dtype))
    marker: BindingCollection = builder.allocate()
    word = builder.inputs["x"]
    for index, value in enumerate(cycle):
        result = builder.call_program(
            equality_constant_program(width, value), {"x": word, "target": marker}
        )
        word, marker = result["x"], result["target"]
        delta = value ^ cycle[(index + 1) % len(cycle)]
        result = builder.call_program(
            controlled_bitwise_word_program(
                unsigned_register_type(width), "xor_constant", pattern=delta
            ),
            {"ctrl": marker, "x": word},
        )
        marker, word = result["ctrl"], result["x"]
    result = builder.call_program(
        equality_constant_program(width, cycle[0]), {"x": word, "target": marker}
    )
    marker = result["target"]
    assert isinstance(marker, RegisterBinding)
    builder.free(marker)
    return builder.finalize({"x": result["x"]})


def permutation_program(
    size: int, cycles: tuple[tuple[int, ...], ...], width: int
) -> ProgramRecord:
    """Apply source-ordered cycles, including an empty identity sequence.

    Cross-cycle overlap retains composition order. Each child releases its marker
    and predicate work before the next cycle. Source dtype views remain explicit.
    """
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=_dtype(size, width))
    )
    word = builder.inputs["x"]
    for cycle in cycles:
        word = builder.call_program(
            permutation_cycle_program(size, cycle, width), {"x": word}
        )["x"]
    return builder.finalize({"x": word})
