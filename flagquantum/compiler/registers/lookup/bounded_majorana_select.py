# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 SelectedMajoranaFermion.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/selected_majorana_fermion.py

"""Bounded Majorana selection with a clean Jordan–Wigner accumulator."""

from math import prod

from ....core.ir import Instruction
from ..._component_trace.resource_model import ComponentCost, GateCost
from ..bits.bitwise import _validate_wires
from .unary_iteration import (
    append_bounded_unary_tree,
    append_nested_bounded_unary_tree,
    unary_iteration_workspace,
)


def bounded_majorana_workspace(
    spin_orbitals: int,
    *,
    controlled: bool,
    axis_extents: tuple[int, ...] = (),
    selection_width: int | None = None,
) -> int:
    """Return accumulator and unary-tree work for the declared address."""
    if selection_width is not None:
        if (
            axis_extents
            or type(spin_orbitals) is not int
            or spin_orbitals < (1 if controlled else 2)
            or type(selection_width) is not int
            or selection_width < max(1, (spin_orbitals - 1).bit_length())
        ):
            raise ValueError(
                "Flat Majorana requires a concrete word covering the bounded targets"
            )
        return 1 + unary_iteration_workspace(selection_width, 1 if controlled else None)
    if type(spin_orbitals) is not int or spin_orbitals < 4 or spin_orbitals % 2:
        raise ValueError("Bounded Majorana requires at least two spatial orbitals")
    if axis_extents and (
        any(type(n) is not int or n < 2 for n in axis_extents)
        or prod(axis_extents) != spin_orbitals
    ):
        raise ValueError("Bounded Majorana axes must cover the target register")
    width = (
        sum((n - 1).bit_length() for n in axis_extents)
        if axis_extents
        else (spin_orbitals - 1).bit_length()
    )
    return 1 + unary_iteration_workspace(width, 1 if controlled else None)


def bounded_majorana_ir(
    selection: tuple[int, ...],
    target: tuple[int, ...],
    control: int | None,
    pauli: str,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    axis_extents: tuple[int, ...] = (),
    selection_width: int | None = None,
) -> tuple[Instruction, ...]:
    """Emit the source flat or orbital-major traversal and its feedback.

    Workspace starts with the accumulator and is followed by tree predicates.
    Each predicate owns one classical slot. Spatial padding follows the
    source's bounded traversal, so callers must restrict spatial addresses
    to populated orbitals to obtain the Majorana selection contract.
    An explicit selection_width admits a flat register with arbitrary
    concrete target count and preserves extra physical address bits.
    """
    needed = bounded_majorana_workspace(
        len(target),
        controlled=control is not None,
        axis_extents=axis_extents,
        selection_width=selection_width,
    )
    width = needed - int(control is not None)
    allowed = ("X", "Y") if selection_width is None else ("X", "Y", "Z")
    if len(selection) != width or pauli not in allowed:
        raise ValueError(
            "Bounded Majorana requires a matching selection word and supported Pauli"
        )
    if len(workspace) != needed or len(classical_bits) != needed - 1:
        raise ValueError("Bounded Majorana requires matching work wires and slots")
    _validate_wires(
        (*selection, *target, *workspace, *((control,) if control is not None else ()))
    )
    _validate_wires(classical_bits)
    accumulator, *predicates = workspace
    gates = [
        (
            Instruction("x", (accumulator,))
            if control is None
            else Instruction("cx", (control, accumulator))
        )
    ]

    def visit(index: int, predicate: int) -> None:
        gates.extend(
            (
                Instruction("cx", (predicate, accumulator)),
                Instruction("c" + pauli.lower(), (predicate, target[index])),
                Instruction("cz", (accumulator, target[index])),
            )
        )

    if axis_extents:
        axes = []
        start = 0
        for extent in axis_extents:
            stop = start + (extent - 1).bit_length()
            axes.append(selection[start:stop])
            start = stop
        append_nested_bounded_unary_tree(
            gates,
            control,
            tuple(axes),
            axis_extents,
            tuple(predicates),
            classical_bits,
            visit,
        )
    else:
        # The full spin axis is one bit and can be flattened with the orbital tree.
        append_bounded_unary_tree(
            gates,
            control,
            selection,
            len(target),
            tuple(predicates),
            classical_bits,
            visit,
        )
    return tuple(gates)


def source_bounded_majorana_cost(
    spin_orbitals: int,
    pauli: str,
    *,
    controlled: bool,
    selection_width: int | None = None,
) -> ComponentCost:
    """Count the source calls for one bounded Majorana selection."""
    bounded_majorana_workspace(
        spin_orbitals, controlled=controlled, selection_width=selection_width
    )
    active_width = (spin_orbitals - 1).bit_length()
    width = active_width if selection_width is None else selection_width
    work = 1 + unary_iteration_workspace(active_width, 1 if controlled else None)
    allowed = ("X", "Y") if selection_width is None else ("X", "Y", "Z")
    if pauli not in allowed:
        raise ValueError("Bounded Majorana requires a supported Pauli")
    controls = int(controlled)
    ands = spin_orbitals - 2 + controls
    cliffords = 3 * spin_orbitals + 2 * ands + 1 + 2 * (1 - controls)
    return ComponentCost(
        f"SelectedMajoranaFermion({pauli})",
        spin_orbitals + width + controls + work,
        GateCost(and_bloq=ands, clifford=cliffords, measurement=ands),
        family="multiplexers",
    )
