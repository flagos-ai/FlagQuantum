# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROM and UnaryIterationGate call graphs.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/unary_iteration_bloq.py

"""Concrete logical costs for source unary QROM tables."""

from math import prod

from .._component_trace.resource_model import ComponentCost, GateCost
from .source_qrom import _uniform


def source_qrom_cost(
    control_count: int,
    selection_widths: tuple[int, ...],
    shape: tuple[int, ...],
    target_layout: tuple[tuple[int, int], ...],
    datasets: tuple[tuple[int, ...], ...],
) -> ComponentCost:
    """Estimate QROM from flattened row-major data and (word count, width) pairs."""
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

    reduction = max(0, control_count - 1)
    width = sum(selection_widths)
    target_width = sum(words * bits for words, bits in target_layout)
    peak = target_width + 2 * (width + control_count) - 1

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

    if not width or uniform(0, rows):
        return ComponentCost(
            "QROM",
            peak,
            GateCost(
                and_bloq=reduction, clifford=load(0) + reduction, measurement=reduction
            ),
        )
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
    return ComponentCost(
        "QROM",
        peak,
        GateCost(
            and_bloq=splits + reduction,
            clifford=loads + 2 * splits + reduction + 2 * (control_count == 0),
            measurement=splits + reduction,
        ),
    )
