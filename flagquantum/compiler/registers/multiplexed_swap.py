# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiplexedCSwap.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/swap_network/multiplexed_cswap.py

"""Multiplex an exact controlled exchange over bounded register indices."""

from math import prod

from ...core.ir import Instruction
from .._component_trace.resource_model import ComponentCost, GateCost
from .bitwise import _validate_wires
from .controlled_swap import controlled_swap_ir
from .multi_control import multi_and_ir
from .unary_iteration import append_bounded_unary_tree


def multiplexed_cswap_ir(
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    shape: tuple[int, ...],
    targets: tuple[tuple[int, ...], ...],
    output: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Exchange the indexed row with the output and clear traversal workspace.

    Each selection word is big-endian. Targets are flattened in row-major
    order and the output can start in any state. The source contract assumes
    in-range selections, so padded addresses follow the bounded gate stream.
    """
    widths = tuple(map(len, selections))
    _validate_selection(shape, widths, len(controls))
    if (
        not output
        or len(targets) != prod(shape)
        or any(len(word) != len(output) for word in targets)
    ):
        raise ValueError("Multiplexed swap requires one equal-width word per row")
    size = sum(widths) + len(controls) - 1
    if len(workspace) != size:
        raise ValueError(f"Multiplexed swap requires {size} clean work wires")
    if (
        len(classical_bits) != size
        or len(set(classical_bits)) != size
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            "Multiplexed swap requires one distinct classical slot per work wire"
        )
    _validate_wires(
        (
            *controls,
            *(wire for group in (*selections, *targets) for wire in group),
            *output,
            *workspace,
        )
    )

    gates: list[Instruction] = []

    def visit(axis: int, row: int, control: int | None, offset: int) -> None:
        selection = selections[axis]
        end = offset + len(selection) - int(control is None)
        stride = prod(shape[axis + 1 :])

        def leaf(index: int, predicate: int) -> None:
            selected = row + index * stride
            if axis + 1 == len(shape):
                gates.extend(controlled_swap_ir(predicate, targets[selected], output))
            else:
                visit(axis + 1, selected, predicate, end)

        append_bounded_unary_tree(
            gates,
            control,
            selection,
            shape[axis],
            workspace[offset:end],
            classical_bits[offset:end],
            leaf,
        )

    reduction_size = max(0, len(controls) - 1)
    reduction = workspace[:reduction_size]
    values = (1,) * len(controls)
    if reduction:
        gates.extend(multi_and_ir(controls, values, reduction))
    control = reduction[-1] if reduction else next(iter(controls), None)
    visit(0, 0, control, reduction_size)
    if reduction:
        gates.extend(
            multi_and_ir(
                controls,
                values,
                reduction,
                classical_bits=classical_bits[:reduction_size],
                uncompute=True,
            )
        )
    return tuple(gates)


def source_multiplexed_cswap_cost(
    shape: tuple[int, ...],
    selection_widths: tuple[int, ...],
    control_count: int,
    word_width: int,
) -> ComponentCost:
    """Estimate source logical gates and peak width without emitting a circuit."""
    _validate_selection(shape, selection_widths, control_count)
    if type(word_width) is not int or word_width < 1:
        raise ValueError("Multiplexed swap requires a positive integer word width")

    rows = prod(shape)
    has_control = control_count > 0
    # A bounded binary tree has one fewer fork than leaves. Without an
    # explicit control, two Xs on the first live address bit replace its root.
    tree_ands = rows - 1 - int(not has_control)
    reduction_ands = max(control_count - 1, 0)
    ands = tree_ands + reduction_ands
    cliffords = 2 * tree_ands + reduction_ands + 2 * int(not has_control)

    # The longest live path visits the complete left half at each first
    # nontrivial split. Leading padded bits do not allocate active work.
    active_work = sum((extent - 1).bit_length() for extent in shape) + control_count - 1
    ports = control_count + sum(selection_widths) + (rows + 1) * word_width
    return ComponentCost(
        "MultiplexedCSwap",
        ports + active_work,
        GateCost(
            cswap=rows * word_width,
            and_bloq=ands,
            clifford=cliffords,
            measurement=ands,
        ),
    )


def _validate_selection(
    shape: tuple[int, ...], widths: tuple[int, ...], control_count: int
) -> None:
    if (
        not shape
        or len(shape) != len(widths)
        or any(type(width) is not int or width < 1 for width in widths)
    ):
        raise ValueError("Multiplexed swap requires nonempty source selection axes")
    if any(
        type(extent) is not int or extent < 1 or (extent - 1).bit_length() > width
        for extent, width in zip(shape, widths, strict=True)
    ):
        raise ValueError("Multiplexed swap extent must fit its selection")
    if type(control_count) is not int or control_count < 0:
        raise ValueError("Multiplexed swap requires a nonnegative control count")
    if not control_count and shape[0] == 1:
        raise ValueError(
            "Source multiplexed swap needs a control for a singleton first axis"
        )
