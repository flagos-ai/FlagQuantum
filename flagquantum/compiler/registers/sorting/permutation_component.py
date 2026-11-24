# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PermutationCycle and Permutation.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/permutation.py

"""Compose bounded basis permutations from equality and controlled XOR calls."""

from functools import partial

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    unsigned_register_type,
)
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..comparison.equality_component import equality_constant_program
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from .permutation_graph import permutation_cycle_graph


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
    size: int, cycle: tuple[int, ...], width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Apply one source cycle to a bounded MSB-first THRU word.

    Marker allocation belongs to the cycle. Equality owns its temporary carries.
    Marker and carries return clean. Singleton cycles retain their full source
    work. Invalid or empty cycles raise ValueError.
    """
    if type(adjoint) is not bool:
        raise ValueError("Permutation cycle direction must be boolean")
    dtype = _dtype(size, width)
    if (
        not cycle
        or len(set(cycle)) != len(cycle)
        or any(type(value) is not int or not 0 <= value < size for value in cycle)
    ):
        raise ValueError("Permutation cycle requires distinct entries in range")
    record = permutation_cycle_graph(
        dtype,
        cycle,
        partial(equality_constant_program, width),
        lambda delta: controlled_bitwise_word_program(
            unsigned_register_type(width), "xor_constant", pattern=delta
        ),
    )
    if not adjoint:
        return record
    pairs = []
    for value, successor in zip(cycle, (*cycle[1:], cycle[0]), strict=True):
        pairs.append(
            (
                equality_constant_program(width, value),
                equality_constant_program(width, value, adjoint=True),
            )
        )
        xor = controlled_bitwise_word_program(
            unsigned_register_type(width), "xor_constant", pattern=value ^ successor
        )
        pairs.append((xor, xor))
    return adjoint_component_calls(record, tuple(pairs))


def permutation_program(
    size: int, cycles: tuple[tuple[int, ...], ...], width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Apply source-ordered cycles, including an empty identity sequence.

    Cross-cycle overlap retains composition order. Each child releases its marker
    and predicate work before the next cycle. Source dtype views remain explicit.
    """
    if type(adjoint) is not bool:
        raise ValueError("Permutation direction must be boolean")
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(x=_dtype(size, width))
    )
    word = builder.inputs["x"]
    pairs = []
    for cycle in cycles:
        forward = permutation_cycle_program(size, cycle, width)
        word = builder.call_program(forward, {"x": word})["x"]
        if adjoint:
            pairs.append(
                (forward, permutation_cycle_program(size, cycle, width, adjoint=True))
            )
    record = builder.finalize({"x": word})
    return adjoint_component_calls(record, tuple(pairs)) if adjoint else record
