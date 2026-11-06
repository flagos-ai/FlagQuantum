# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROM and UnaryIterationGate.
# Sources:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qrom.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/unary_iteration_bloq.py

"""Bounded unary XOR lookup using the original measured And tree."""

from collections.abc import Callable
from math import prod

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires
from .multi_control import multi_and_ir


def _uniform(words: tuple[int, ...]) -> bool:
    return all(word == words[0] for word in words)


def source_qrom_workspace_size(
    control_count: int,
    selection_widths: tuple[int, ...],
    datasets: tuple[tuple[int, ...], ...],
) -> int:
    """Return the source allocation size for concrete row-major datasets."""
    if not any(selection_widths) or all(_uniform(data) for data in datasets):
        return max(0, control_count - 1)
    return sum(selection_widths) + control_count - 1


def _append_unary_tree(
    gates: list[Instruction],
    control: int | None,
    selection: tuple[int, ...],
    extent: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    uniform: Callable[[int, int], bool],
    load: Callable[[int, int], None],
) -> None:
    """Visit the source's bounded tree with one effective control."""

    def visit(predicate: int, depth: int, start: int, stop: int) -> None:
        if stop <= extent and (stop - start == 1 or uniform(start, stop)):
            load(start, predicate)
            return
        middle = (start + stop) // 2
        if extent <= middle:
            visit(predicate, depth + 1, start, middle)
            return
        wire, work = selection[depth], workspace[depth]
        gates.extend(and_compute_ir(predicate, wire, work, values=(1, 0)))
        visit(work, depth + 1, start, middle)
        gates.append(Instruction("cx", (predicate, work)))
        visit(work, depth + 1, middle, stop)
        gates.extend(and_uncompute_ir(predicate, wire, work, classical_bits[depth]))

    capacity = 1 << len(selection)
    if control is not None:
        visit(control, 0, 0, capacity)
        return
    # The first address bit that distinguishes populated halves is the predicate.
    while extent <= capacity // 2:
        selection = selection[1:]
        capacity //= 2
    predicate, selection = selection[0], selection[1:]
    middle = capacity // 2
    gates.append(Instruction("x", (predicate,)))
    visit(predicate, 0, 0, middle)
    gates.append(Instruction("x", (predicate,)))
    visit(predicate, 0, middle, capacity)


def source_qrom_ir(
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    shape: tuple[int, ...],
    targets: tuple[tuple[tuple[int, ...], ...], ...],
    datasets: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """XOR concrete datasets into target words and return the And pool to zero.

    Each dataset is flattened in row-major order, with each selected row
    containing one word per corresponding target word. Singleton axes have
    empty physical selections. The source channel assumes an in-range index.
    """
    if len(selections) != len(shape) or any(
        type(extent) is not int
        or not 1 <= extent <= 1 << len(wires)
        or (extent == 1 and bool(wires))
        for extent, wires in zip(shape, selections, strict=True)
    ):
        raise ValueError("QROM selection widths must fit the source axes")
    if not datasets or len(datasets) != len(targets):
        raise ValueError("QROM requires one target group per dataset")
    for words, data in zip(targets, datasets, strict=True):
        if (
            not words
            or not words[0]
            or len(data) != prod(shape) * len(words)
            or any(len(word) != len(words[0]) for word in words)
            or any(
                type(value) is not int or not 0 <= value < 1 << len(words[0])
                for value in data
            )
        ):
            raise ValueError(
                "QROM dataset must fit its row shape and target word width"
            )

    widths = tuple(map(len, selections))
    constant = all(_uniform(data) for data in datasets)
    if any(widths) and not all(widths) and not constant:
        raise ValueError(
            "Source QROM cannot index mixed singleton axes in a nonuniform lookup"
        )
    size = source_qrom_workspace_size(len(controls), widths, datasets)
    if len(workspace) != size:
        raise ValueError(f"QROM requires {size} clean work wires")
    if (
        len(classical_bits) != size
        or len(set(classical_bits)) != size
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError("QROM requires one distinct classical slot per work wire")
    wires = (
        *controls,
        *(wire for axis in selections for wire in axis),
        *(wire for group in targets for word in group for wire in word),
        *workspace,
    )
    _validate_wires(wires)

    gates: list[Instruction] = []
    reduction_size = max(0, len(controls) - 1)
    reduction = workspace[:reduction_size]
    control = controls[0] if controls else None
    if reduction:
        gates.extend(multi_and_ir(controls, (1,) * len(controls), reduction))
        control = reduction[-1]

    def load(row: int, predicate: int | None) -> None:
        for group, data in zip(targets, datasets, strict=True):
            offset = row * len(group)
            for word, value in zip(
                group, data[offset : offset + len(group)], strict=True
            ):
                for bit, wire in enumerate(word):
                    if value & (1 << (len(word) - 1 - bit)):
                        gates.append(
                            Instruction("x", (wire,))
                            if predicate is None
                            else Instruction("cx", (predicate, wire))
                        )

    def visit_axis(axis: int, row: int, predicate: int | None, offset: int) -> None:
        if axis == len(shape):
            load(row, predicate)
            return
        stride = prod(shape[axis + 1 :])
        work_end = offset + len(selections[axis]) - int(predicate is None)

        def uniform(start: int, stop: int) -> bool:
            return all(
                _uniform(
                    data[
                        (row + start * stride)
                        * len(group) : (row + stop * stride)
                        * len(group)
                    ]
                )
                for group, data in zip(targets, datasets, strict=True)
            )

        _append_unary_tree(
            gates,
            predicate,
            selections[axis],
            shape[axis],
            workspace[offset:work_end],
            classical_bits[offset:work_end],
            uniform,
            lambda index, leaf: visit_axis(
                axis + 1, row + index * stride, leaf, work_end
            ),
        )

    if not any(widths) or constant:
        load(0, control)
    else:
        visit_axis(0, 0, control, reduction_size)
    if reduction:
        gates.extend(
            multi_and_ir(
                controls,
                (1,) * len(controls),
                reduction,
                classical_bits=classical_bits[:reduction_size],
                uncompute=True,
            )
        )
    return tuple(gates)
