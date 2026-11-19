# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ParallelComparators.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/sorting.py

"""Compose the existing offset pair order with directional history arrays."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from .compare_exchange_component import (
    adjoint_comparison_network,
    compare_exchange_program,
)
from .parallel_comparison import parallel_comparison_pairs


def parallel_comparison_program(
    count: int, offset: int, width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Compare offset pairs, retaining one logical history per source-ordered pair.

    Zero-pair blocks return empty junk and require no comparison workspace.
    The inverse consumes histories in reverse source dependency order.
    """
    if type(adjoint) is not bool:
        raise ValueError("Parallel comparison direction requires a boolean")
    pairs = parallel_comparison_pairs(count, offset)
    comparator = compare_exchange_program(width)
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("xs", unsigned_register_type(width), (count,)),
                Register(
                    "junk", quantum_bit_type(), (len(pairs),), side=RegisterSide.RIGHT
                ),
            )
        )
    )
    xs = np.asarray(builder.inputs["xs"]).copy()
    histories = []
    for i, j in pairs:
        result = builder.call_program(comparator, {"a": xs[i], "b": xs[j]})
        xs[i], xs[j] = result["a"], result["b"]
        histories.append(result["out"])
    record = builder.finalize({"xs": xs, "junk": np.array(histories, dtype=object)})
    return adjoint_comparison_network(record, width) if adjoint else record
