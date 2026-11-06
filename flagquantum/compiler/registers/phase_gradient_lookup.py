# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PRGAViaPhaseGradient.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_via_rotation.py

"""Programmable phase kickback through a QROM and an external gradient."""

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .phase_addition import phase_add_ir, phase_add_workspace_size
from .source_qrom import source_qrom_ir, source_qrom_workspace_size


def phase_gradient_lookup_requirements(
    control_count: int, selection_width: int, phase_width: int, words: tuple[int, ...]
) -> tuple[int, int]:
    """Return clean qubits (including the angle word) and reusable slots."""
    if (
        type(control_count) is not int
        or control_count < 0
        or type(selection_width) is not int
        or selection_width < 0
        or len(words) != 1 << selection_width
    ):
        raise ValueError("Phase lookup requires a complete table for the selection")
    add = phase_add_workspace_size(phase_width, phase_width)
    lookup = source_qrom_workspace_size(control_count, (selection_width,), (words,))
    scratch = max(add, lookup)
    return phase_width + scratch, scratch


def phase_gradient_lookup_ir(
    controls: tuple[int, ...],
    selection: tuple[int, ...],
    gradient: tuple[int, ...],
    words: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Load the selected angle under all-one controls, add, and erase the angle.

    All words are big endian. The first part of workspace holds the angle;
    QROM and addition reuse the remaining clean scratch and classical slots.
    The source lookup is its own adjoint.
    """
    width = len(gradient)
    work_size, slots = phase_gradient_lookup_requirements(
        len(controls), len(selection), width, words
    )
    if len(workspace) != work_size or len(classical_bits) != slots:
        raise ValueError("Phase lookup requires its declared workspace and slots")
    _validate_wires((*controls, *selection, *gradient, *workspace))
    angle, scratch = workspace[:width], workspace[width:]
    lookup_size = source_qrom_workspace_size(len(controls), (len(selection),), (words,))
    lookup = source_qrom_ir(
        controls,
        (selection,),
        (len(words),),
        ((angle,),),
        (words,),
        scratch[:lookup_size],
        classical_bits[:lookup_size],
    )
    add_size = phase_add_workspace_size(width, width)
    add = phase_add_ir(angle, gradient, scratch[:add_size], classical_bits[:add_size])
    return (*lookup, *add, *lookup)
