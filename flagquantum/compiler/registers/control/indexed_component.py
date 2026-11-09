# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ApplyLthBloq and ApplyGateToLthQubit.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers

"""Dispatch controlled components through shared bounded register traversal."""

from collections.abc import Callable
from math import prod
from typing import Literal

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..lookup.unary_iteration import append_nested_bounded_unary_tree
from .multi_control import multi_and_ir


def indexed_component_workspace(
    axis_widths: tuple[int, ...], control_count: int
) -> int:
    """Return reserved clean parent work, excluding the child's own storage."""
    if not axis_widths or any(
        type(width) is not int or width < 1 for width in axis_widths
    ):
        raise ValueError("Indexed components require positive concrete axis widths")
    if type(control_count) is not int or control_count < 0:
        raise ValueError("Indexed components require a nonnegative control count")
    return sum(axis_widths) + control_count - 1


def indexed_component_ir(
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    emit_controlled: Callable[[int, int], tuple[Instruction, ...]],
    *,
    controls: tuple[int, ...] = (),
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply one child per selected tuple, preserving the source bounded tree.

    Controls are positive. The callback receives the flattened C-order index
    and a positive predicate; it owns child binding and must preserve that
    predicate, the input axes and this parent's workspace. For adjoint traversal,
    it must emit the child adjoint; the parent does not invert child IR.
    """
    required = indexed_component_workspace(tuple(map(len, axes)), len(controls))
    if type(adjoint) is not bool:
        raise ValueError("Indexed traversal direction must be a boolean")
    if len(axes) != len(extents) or any(
        type(size) is not int or size < 1 or size > 1 << len(axis)
        for axis, size in zip(axes, extents, strict=True)
    ):
        raise ValueError("Every axis needs a concrete extent fitting its word")
    if not controls and extents[0] == 1:
        raise ValueError("Source uncontrolled singleton first axis has no traversal")
    if len(workspace) != required:
        raise ValueError(f"Indexed components require {required} clean work wires")
    if (
        len(classical_bits) != required
        or len(set(classical_bits)) != required
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            "Indexed components require one distinct classical slot per work wire"
        )
    _validate_wires((*controls, *(wire for axis in axes for wire in axis), *workspace))
    prefix = max(0, len(controls) - 1)
    predicate = (
        controls[0] if len(controls) == 1 else workspace[prefix - 1] if prefix else None
    )
    gates: list[Instruction] = []
    if prefix:
        gates.extend(multi_and_ir(controls, (1,) * len(controls), workspace[:prefix]))

    def visit(index: int, active: int) -> None:
        gates.extend(emit_controlled(index, active))

    append_nested_bounded_unary_tree(
        gates,
        predicate,
        axes,
        extents,
        workspace[prefix:],
        classical_bits[prefix:],
        visit,
        adjoint=adjoint,
    )
    if prefix:
        gates.extend(
            multi_and_ir(
                controls,
                (1,) * len(controls),
                workspace[:prefix],
                classical_bits=classical_bits[:prefix],
                uncompute=True,
            )
        )
    return tuple(gates)


def indexed_pauli_targets_ir(
    axes: tuple[tuple[int, ...], ...],
    extents: tuple[int, ...],
    targets: tuple[int, ...],
    paulis: tuple[Literal["i", "x", "y", "z"], ...],
    *,
    controls: tuple[int, ...] = (),
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply the selected Pauli to the selected target in C-order layout."""
    if (
        len(targets) != prod(extents)
        or len(paulis) != len(targets)
        or any(pauli not in ("i", "x", "y", "z") for pauli in paulis)
    ):
        raise ValueError(
            "Target-position dispatch needs one Pauli and target per address"
        )
    _validate_wires(
        (*targets, *controls, *(wire for axis in axes for wire in axis), *workspace)
    )

    def emit(index: int, predicate: int) -> tuple[Instruction, ...]:
        gate = paulis[index]
        return (
            ()
            if gate == "i"
            else (Instruction("c" + gate, (predicate, targets[index])),)
        )

    return indexed_component_ir(
        axes,
        extents,
        emit,
        controls=controls,
        workspace=workspace,
        classical_bits=classical_bits,
        adjoint=adjoint,
    )
