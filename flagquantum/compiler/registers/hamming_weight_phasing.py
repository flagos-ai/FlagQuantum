# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HammingWeightPhasing and HammingWeightPhasingViaPhaseGradient.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/hamming_weight_phasing.py

"""Phase a word by its coherently computed Hamming weight."""

from math import isfinite, pi

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .hamming_weight import hamming_weight_ir
from .qvr_phase_gradient import qvr_phase_gradient_ir


def hamming_weight_phasing_ir(
    source: tuple[int, ...],
    junk: tuple[int, ...],
    weight: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    exponent: float,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply a Z power to each source bit through the shared count tree.

    The caller owns the clean count and junk wires. Only the measured
    erase uses the classical slots. The forward phase order is least
    significant first, and the adjoint reverses and negates those phases.
    The output preserves the source bits and clears all work.
    """
    if type(adjoint) is not bool:
        raise ValueError("Hamming phasing adjoint must be boolean")
    if type(exponent) not in (int, float) or not isfinite(exponent):
        raise ValueError("Hamming phasing requires a finite exponent")
    compute = hamming_weight_ir(source, junk, weight)
    erase = hamming_weight_ir(
        source, junk, weight, adjoint=True, classical_bits=classical_bits
    )
    terms = tuple(enumerate(reversed(weight)))
    order = reversed(terms) if adjoint else iter(terms)
    sign = -1 if adjoint else 1
    phases = tuple(
        Instruction(
            "phase", (wire,), params={"theta": pi * (sign * exponent * (1 << i))}
        )
        for i, wire in order
    )
    return (*compute, *phases, *erase)


def hamming_weight_gradient_phasing_ir(
    source: tuple[int, ...],
    junk: tuple[int, ...],
    weight: tuple[int, ...],
    gradient: tuple[int, ...],
    addition_work: tuple[int, ...],
    count_bits: tuple[int, ...],
    addition_bits: tuple[int, ...],
    *,
    exponent: float,
    epsilon: float,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Kick back a Hamming-weight phase through a caller-owned gradient.

    The source QVR treats the count as an unsigned integer and scales it
    by exponent/2. Supply the selected negative-exponent gradient state.
    Count, junk and addition work start and end at zero. The two slot
    pools may overlap because addition finishes before count erasure.
    """
    _validate_wires((*source, *junk, *weight, *gradient, *addition_work))
    compute = hamming_weight_ir(source, junk, weight)
    erase = hamming_weight_ir(
        source, junk, weight, adjoint=True, classical_bits=count_bits
    )
    phase = qvr_phase_gradient_ir(
        weight,
        gradient,
        addition_work,
        addition_bits,
        cost_fraction_bits=0,
        coefficient=exponent / 2,
        epsilon=epsilon,
        adjoint=adjoint,
    )
    return (*compute, *phase, *erase)
