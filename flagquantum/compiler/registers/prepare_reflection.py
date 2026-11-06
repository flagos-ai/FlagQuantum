# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ReflectionUsingPrepare.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/reflections/reflection_using_prepare.py

"""Reflection conjugated by an already bound preparation and its inverse."""

from cmath import phase
from math import isclose

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .multi_control import multi_control_z_ir


def prepare_reflection_requirements(width: int, *, controlled: bool) -> tuple[int, int]:
    """Return reflection-only clean work and reusable classical slots."""
    if type(width) is not int or width < 0 or type(controlled) is not bool:
        raise ValueError(
            "Reflection requires a nonnegative width and boolean controlled flag"
        )
    prefix = max(0, width - 1)
    return prefix + int(not controlled), prefix


def prepare_reflection_ir(
    selection: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    prepare: tuple[Instruction, ...] = (),
    unprepare: tuple[Instruction, ...] = (),
    control: int | None = None,
    control_value: int = 1,
    global_phase: complex = 1,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Conjugate the pinned zero reflection by the supplied preparation.

    The caller supplies an inverse pair on selection and its preparation
    registers. Any logical preparation junk remains under that pair's
    contract. The reflection marks zero on selection only and restores
    all reflection work. The adjoint conjugates the supplied unit phase.
    """
    controlled = control is not None
    work, slots = prepare_reflection_requirements(len(selection), controlled=controlled)
    if len(workspace) != work or len(classical_bits) != slots:
        raise ValueError(
            f"Reflection requires {work} clean work qubits and {slots} slots"
        )
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Reflection control value must be zero or one")
    if type(adjoint) is not bool:
        raise ValueError("Reflection adjoint must be boolean")
    if not isclose(abs(global_phase), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Reflection global phase must be a unit coefficient")

    target = control if control is not None else workspace[0]
    prefix = workspace if controlled else workspace[1:]
    _validate_wires((*selection, target, *prefix))
    if bool(prepare) != bool(unprepare) or any(
        target in instruction.wires
        for block in (prepare, unprepare)
        for instruction in block
    ):
        raise ValueError(
            "Reflection requires a preparation pair independent of its phase target"
        )

    mark = multi_control_z_ir(
        selection, (0,) * len(selection), target, prefix, classical_bits
    )
    flip = () if controlled and control_value else (Instruction("x", (target,)),)
    angle = phase(global_phase)
    shift = (
        (
            Instruction(
                "phase", (target,), params={"theta": -angle if adjoint else angle}
            ),
        )
        if global_phase != 1
        else ()
    )
    return (*unprepare, *flip, *mark, *shift, *flip, *prepare)
