# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROM and UnaryIterationGate.
# Sources:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qrom.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/unary_iteration_bloq.py

"""Bounded unary XOR lookup using the original measured And tree."""

from math import prod

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..control.control_predicate import ControlRegister
from ..control.controlled_component import controlled_component_ir
from ..control.multi_control import multi_and_ir
from .unary_iteration import append_bounded_unary_tree


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


def source_qrom_ir(
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    shape: tuple[int, ...],
    targets: tuple[tuple[tuple[int, ...], ...], ...],
    datasets: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    outer_control: int | None = None,
) -> tuple[Instruction, ...]:
    """XOR concrete datasets into target words and return the And pool to zero.

    Each dataset is flattened in row-major order, with each selected row
    containing one word per corresponding target word. Singleton axes have
    empty physical selections. The source channel assumes an in-range index.
    An optional positive outer control follows the generic source decomposition
    for loads containing only X and CX.
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
    _validate_wires(wires if outer_control is None else (*wires, outer_control))

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

        append_bounded_unary_tree(
            gates,
            predicate,
            selections[axis],
            shape[axis],
            workspace[offset:work_end],
            classical_bits[offset:work_end],
            lambda index, leaf: visit_axis(
                axis + 1, row + index * stride, leaf, work_end
            ),
            break_early=uniform,
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
    if outer_control is None:
        return tuple(gates)
    if any(gate.name not in ("x", "cx") for gate in gates):
        raise ValueError("Pinned controlled QROM cannot decompose controlled And")
    return tuple(
        Instruction("cx" if gate.name == "x" else "ccx", (outer_control, *gate.wires))
        for gate in gates
    )


def control_source_qrom_ir(
    outer_controls: tuple[tuple[int, ...], ...],
    registers: tuple[ControlRegister, ...],
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    shape: tuple[int, ...],
    targets: tuple[tuple[tuple[int, ...], ...], ...],
    datasets: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    predicate_workspace: tuple[int, ...],
    predicate_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Apply a shared outer predicate to an executable generic controlled lookup."""
    _validate_wires(
        (
            *controls,
            *(wire for group in (*outer_controls, *selections) for wire in group),
            *(wire for group in targets for word in group for wire in word),
            *workspace,
            *predicate_workspace,
        )
    )
    return controlled_component_ir(
        outer_controls,
        registers,
        lambda control: source_qrom_ir(
            controls,
            selections,
            shape,
            targets,
            datasets,
            workspace,
            classical_bits,
            outer_control=control,
        ),
        predicate_workspace,
        predicate_bits,
    )
