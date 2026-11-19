# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BitonicSort.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Compose the original recursive sort and ordered left/right/merge histories."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..comparison.compare_exchange_component import (
    adjoint_comparison_network,
    compare_exchange_program,
)
from ..component_builder import ComponentProgramBuilder
from .bitonic_merge_component import bitonic_merge_program
from .bitonic_sort import bitonic_sort_history_count


def bitonic_sort_program(
    count: int, width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Sort a power-of-two array while retaining every source comparison history.

    One word returns empty history. The inverse consumes a complete valid
    result/history and restores the original word array with histories cleared.
    Temporary work belongs to the compare/exchange children.
    """
    if type(adjoint) is not bool:
        raise ValueError("Bitonic sort direction requires a boolean")
    histories = bitonic_sort_history_count(count)
    comparator = compare_exchange_program(width)
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("xs", unsigned_register_type(width), (count,)),
                Register(
                    "junk", quantum_bit_type(), (histories,), side=RegisterSide.RIGHT
                ),
            )
        )
    )
    xs = np.asarray(builder.inputs["xs"]).copy()
    junk = np.array([], dtype=object)
    if count == 2:
        result = builder.call_program(comparator, {"a": xs[0], "b": xs[1]})
        xs[0], xs[1] = result["a"], result["b"]
        junk = np.array([result["out"]], dtype=object)
    elif count > 2:
        half = count // 2
        child = bitonic_sort_program(half, width)
        left = builder.call_program(child, {"xs": xs[:half]})
        right = builder.call_program(child, {"xs": xs[half:]})
        merged = builder.call_program(
            bitonic_merge_program(half, width), {"xs": left["xs"], "ys": right["xs"]}
        )
        xs = np.asarray(merged["result"])
        junk = np.concatenate(
            (
                np.asarray(left["junk"]),
                np.asarray(right["junk"]),
                np.asarray(merged["junk"]),
            )
        )
    record = builder.finalize({"xs": xs, "junk": junk})
    return adjoint_comparison_network(record, width) if adjoint else record
