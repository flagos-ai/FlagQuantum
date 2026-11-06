# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PhaseGradientUnitary, PhaseGradientState and CZPowGate.
# Sources:
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/phase_gradient.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/rotation.py

"""Phase-gradient unitaries and state preparation in FlagQuantum IR."""

from math import isfinite, pi

from ...core.ir import Instruction
from .and_compute import and_compute_ir, and_uncompute_ir
from .bitwise import _validate_wires


def phase_gradient_unitary_ir(
    gradient: tuple[int, ...],
    *,
    exponent: float = 1.0,
    control: int | None = None,
    work: int | None = None,
    classical_bit: int | None = None,
) -> tuple[Instruction, ...]:
    """Apply the source Z powers on a big-endian gradient register.

    The positive-control path implements each CZ power with a logical And,
    a phase on its target, and measured uncomputation. Supply one clean work
    qubit and one classical slot, reused for the full register. Negate the
    exponent for the adjoint. Angles are emitted directly in radians.
    """
    if type(exponent) not in (int, float) or not isfinite(pi * exponent):
        raise ValueError("Phase gradient requires a finite phase angle")
    if control is None:
        if work is not None or classical_bit is not None:
            raise ValueError("Uncontrolled phase gradient requires no workspace")
        _validate_wires(gradient)
        return tuple(
            Instruction("phase", (wire,), params={"theta": pi * (exponent / (1 << i))})
            for i, wire in enumerate(gradient)
        )
    if work is None or classical_bit is None:
        raise ValueError("Controlled phase gradient requires work and a classical slot")
    _validate_wires((*gradient, control, work))
    gates: list[Instruction] = []
    for i, wire in enumerate(gradient):
        gates.extend(and_compute_ir(control, wire, work))
        gates.append(
            Instruction("phase", (work,), params={"theta": pi * (exponent / (1 << i))})
        )
        gates.extend(and_uncompute_ir(control, wire, work, classical_bit))
    return tuple(gates)


def phase_gradient_state_ir(
    gradient: tuple[int, ...], *, inverse: bool = False, exponent: float = 1.0
) -> tuple[Instruction, ...]:
    """Prepare or unprepare a phase-gradient state on caller-owned qubits.

    The existing FlagQuantum default prepares the positive exponent. Pass
    exponent=-1 for the pinned source state's default. The forward input is
    zero; the inverse of the matching state restores that input.
    """
    phases = phase_gradient_unitary_ir(
        gradient, exponent=-exponent if inverse else exponent
    )
    hadamards = tuple(Instruction("h", (wire,)) for wire in gradient)
    return (*phases, *hadamards) if inverse else (*hadamards, *phases)
