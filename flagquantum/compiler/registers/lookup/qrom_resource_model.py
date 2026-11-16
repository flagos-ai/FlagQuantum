# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROM and UnaryIterationGate call graphs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/unary_iteration_bloq.py

"""Concrete logical costs for source unary QROM tables."""

from collections import Counter
from math import prod

from ..._component_trace.resource_model import ComponentCost, GateCost
from .source_qrom import _uniform


def _source_qrom_traversal_counts(
    control_count: int,
    selection_widths: tuple[int, ...],
    shape: tuple[int, ...],
    target_layout: tuple[tuple[int, int], ...],
    datasets: tuple[tuple[int, ...], ...],
) -> tuple[int, int, bool]:
    """Return split count, target bit toggles and whether zero selection applies."""
    if type(control_count) is not int or control_count < 0:
        raise ValueError("QROM control count must be nonnegative")
    if len(shape) != len(selection_widths) or any(
        type(width) is not int
        or width < 0
        or type(extent) is not int
        or not 1 <= extent <= 1 << width
        or (extent == 1 and width != 0)
        for width, extent in zip(selection_widths, shape, strict=True)
    ):
        raise ValueError("QROM selection widths must fit the source axes")
    if not target_layout or len(target_layout) != len(datasets):
        raise ValueError("QROM requires one target group per dataset")
    rows = prod(shape)
    for (words, width), data in zip(target_layout, datasets, strict=True):
        if (
            type(words) is not int
            or type(width) is not int
            or words < 1
            or width < 1
            or len(data) != rows * words
            or any(
                type(value) is not int or not 0 <= value < 1 << width for value in data
            )
        ):
            raise ValueError(
                "QROM dataset must fit its row shape and target word width"
            )

    def uniform(start: int, stop: int) -> bool:
        return all(
            _uniform(data[start * words : stop * words])
            for (words, _), data in zip(target_layout, datasets, strict=True)
        )

    def load(row: int) -> int:
        return sum(
            word.bit_count()
            for (words, _), data in zip(target_layout, datasets, strict=True)
            for word in data[row * words : (row + 1) * words]
        )

    if not any(selection_widths) or uniform(0, rows):
        return 0, load(0), True
    if not all(selection_widths):
        raise ValueError(
            "Source QROM cannot index mixed singleton axes in a nonuniform lookup"
        )

    splits, loads = 0, 0

    def axis(level: int, row: int, controlled: bool) -> None:
        nonlocal loads
        if level == len(shape):
            loads += load(row)
            return
        extent = shape[level]
        stride = prod(shape[level + 1 :])

        def visit(start: int, stop: int) -> None:
            nonlocal splits
            if stop <= extent and (
                stop - start == 1 or uniform(row + start * stride, row + stop * stride)
            ):
                axis(level + 1, row + start * stride, True)
                return
            middle = (start + stop) // 2
            if extent <= middle:
                visit(start, middle)
                return
            splits += 1
            visit(start, middle)
            visit(middle, stop)

        capacity = 1 << selection_widths[level]
        if controlled:
            visit(0, capacity)
        else:
            while extent <= capacity // 2:
                capacity //= 2
            visit(0, capacity // 2)
            visit(capacity // 2, capacity)

    axis(0, 0, control_count > 0)
    return splits, loads, False


QromCall = tuple[str, tuple[int, ...], bool]


def source_qrom_calls(
    control_count: int,
    selection_widths: tuple[int, ...],
    shape: tuple[int, ...],
    target_layout: tuple[tuple[int, int], ...],
    datasets: tuple[tuple[int, ...], ...],
) -> dict[QromCall, int]:
    """Declare immediate source calls with literal parameters and adjoint flags.

    The parameter tuple stores control values for And/MultiAnd and
    (word width, constant, control count) for XorK.
    """
    splits, loads, zero_selection = _source_qrom_traversal_counts(
        control_count, selection_widths, shape, target_layout, datasets
    )
    calls: Counter[QromCall] = Counter()
    if control_count > 1:
        name = "And" if control_count == 2 else "MultiAnd"
        values = (1,) * control_count
        calls[(name, values, False)] += 1
        calls[(name, values, True)] += 1
    if zero_selection:
        for (words, width), data in zip(target_layout, datasets, strict=True):
            for value in data[:words]:
                calls[("XorK", (width, value, min(1, control_count)), False)] += 1
    else:
        if control_count == 0:
            calls[("X", (), False)] += 2
        calls[("And", (1, 0), False)] += splits
        calls[("And", (1, 1), True)] += splits
        calls[("CNOT", (), False)] += splits + loads
    return dict(calls)


def source_qrom_cost(
    control_count: int,
    selection_widths: tuple[int, ...],
    shape: tuple[int, ...],
    target_layout: tuple[tuple[int, int], ...],
    datasets: tuple[tuple[int, ...], ...],
    *,
    legacy_shims: bool = False,
) -> ComponentCost:
    """Estimate QROM from flattened row-major data and (word count, width) pairs."""
    splits, loads, zero_selection = _source_qrom_traversal_counts(
        control_count, selection_widths, shape, target_layout, datasets
    )
    reduction = max(0, control_count - 1)
    erase_cliffords = 3 if legacy_shims else 1
    target_width = sum(words * bits for words, bits in target_layout)
    peak = target_width + 2 * (sum(selection_widths) + control_count) - 1
    # Each split has a negative control on compute and a positive erase.
    split_cliffords = 2 + erase_cliffords + 1 if legacy_shims else 2
    return ComponentCost(
        "QROM",
        peak,
        GateCost(
            and_bloq=splits + reduction,
            clifford=(
                loads
                + split_cliffords * splits
                + erase_cliffords * reduction
                + 2 * (control_count == 0 and not zero_selection)
            ),
            measurement=splits + reduction,
        ),
    )
