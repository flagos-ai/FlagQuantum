# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BitonicMerge.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Compose source cross-comparisons and offset rounds with owned histories."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..comparison.compare_exchange_component import (
    adjoint_comparison_network,
    compare_exchange_program,
)
from ..comparison.parallel_comparison_component import parallel_comparison_program
from ..component_builder import ComponentProgramBuilder
from .bitonic_merge import bitonic_merge_history_count


def bitonic_merge_program(
    half_length: int, width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Merge two sorted unsigned halves, retaining source-ordered logical history.

    Forward consumes xs/ys and returns result/junk. Inverse consumes the valid
    result/history and restores both input arrays. Half length is a positive
    power of two. Unsorted halves execute the same source network.
    """
    if type(adjoint) is not bool:
        raise ValueError("Bitonic merge direction requires a boolean")
    count = bitonic_merge_history_count(half_length)
    comparator = compare_exchange_program(width)
    word, bit = unsigned_register_type(width), quantum_bit_type()
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("xs", word, (half_length,), side=RegisterSide.LEFT),
                Register("ys", word, (half_length,), side=RegisterSide.LEFT),
                Register("result", word, (2 * half_length,), side=RegisterSide.RIGHT),
                Register("junk", bit, (count,), side=RegisterSide.RIGHT),
            )
        )
    )
    xs, ys = (
        np.asarray(builder.inputs["xs"]).copy(),
        np.asarray(builder.inputs["ys"]).copy(),
    )
    first = []
    for index in range(half_length):
        other = half_length - 1 - index
        compared = builder.call_program(comparator, {"a": xs[index], "b": ys[other]})
        xs[index], ys[other] = compared["a"], compared["b"]
        first.append(compared["out"])
    result = np.concatenate((xs, ys))
    histories = [np.array(first, dtype=object)]
    for log_offset in reversed(range((half_length - 1).bit_length())):
        compared = builder.call_program(
            parallel_comparison_program(2 * half_length, 1 << log_offset, width),
            {"xs": result},
        )
        result = np.asarray(compared["xs"])
        histories.append(np.asarray(compared["junk"]))
    record = builder.finalize({"result": result, "junk": np.concatenate(histories)})
    return adjoint_comparison_network(record, width) if adjoint else record
