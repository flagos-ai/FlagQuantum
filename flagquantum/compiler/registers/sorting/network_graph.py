# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 arithmetic/sorting.py.

"""Compose finite sorting arrays around a supplied typed comparator definition."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..comparison.parallel_comparison import parallel_comparison_pairs
from ..component_builder import ComponentProgramBuilder
from .bitonic_merge import bitonic_merge_history_count
from .bitonic_sort import bitonic_sort_history_count


def empty_comparison_program(count: int, width: int, *, adjoint: bool) -> ProgramRecord:
    """Retain xs elements and a directional empty history without child calls."""
    if type(width) is not int or width < 0:
        raise ValueError("Empty comparison width must be a nonnegative integer")
    side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("xs", unsigned_register_type(width), (count,)),
                Register("junk", quantum_bit_type(), (0,), side=side),
            )
        )
    )
    outputs = {"xs": builder.inputs["xs"]}
    if not adjoint:
        outputs["junk"] = np.empty(0, dtype=object)
    return builder.finalize(outputs)


def parallel_comparison_graph(
    count: int, offset: int, comparator: ProgramRecord
) -> ProgramRecord:
    """Retain source pair order and one RIGHT logical history per comparison.

    The comparator entry has a/b THRU words and one RIGHT out bit. Its existing
    definition owns temporary work. Array dimensions are finite even when the
    word dtype contains an unresolved width. This builder owns no physical gate
    or resource estimate.
    """
    pairs = parallel_comparison_pairs(count, offset)
    registers = comparator.functions[0].registers
    assert registers is not None
    word = registers[0].dtype
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("xs", word, (count,)),
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
    return builder.finalize({"xs": xs, "junk": np.array(histories, dtype=object)})


def bitonic_merge_graph(half_length: int, comparator: ProgramRecord) -> ProgramRecord:
    """Consume both halves, returning the ordered result and source histories."""
    count = bitonic_merge_history_count(half_length)
    registers = comparator.functions[0].registers
    assert registers is not None
    word, bit = registers[0].dtype, quantum_bit_type()
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
            parallel_comparison_graph(2 * half_length, 1 << log_offset, comparator),
            {"xs": result},
        )
        result = np.asarray(compared["xs"])
        histories.append(np.asarray(compared["junk"]))
    return builder.finalize({"result": result, "junk": np.concatenate(histories)})


def bitonic_sort_graph(count: int, comparator: ProgramRecord) -> ProgramRecord:
    """Retain recursive left/right/merge order and all logical history outputs."""
    histories = bitonic_sort_history_count(count)
    registers = comparator.functions[0].registers
    assert registers is not None
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("xs", registers[0].dtype, (count,)),
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
        child = bitonic_sort_graph(half, comparator)
        left = builder.call_program(child, {"xs": xs[:half]})
        right = builder.call_program(child, {"xs": xs[half:]})
        merged = builder.call_program(
            bitonic_merge_graph(half, comparator), {"xs": left["xs"], "ys": right["xs"]}
        )
        xs = np.asarray(merged["result"])
        junk = np.concatenate(
            (
                np.asarray(left["junk"]),
                np.asarray(right["junk"]),
                np.asarray(merged["junk"]),
            )
        )
    return builder.finalize({"xs": xs, "junk": junk})
