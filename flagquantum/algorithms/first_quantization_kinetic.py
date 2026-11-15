# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 kinetic PREPARE signatures and composition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/prepare_t.py

"""Register contracts and executable coordinate preparation of kinetic PREPARE."""

from ..compiler.registers.preparation.uniform_preparation import uniform_preparation_ir
from ..core.ir.program.signatures import Signature
from .primitives._block_binding import _BoundBlockOperation

KINETIC_COORDINATE_COUNT = 3


def prepare_power_two_signature(bitsize: int) -> Signature:
    """Return the locked power-of-two declaration's through word."""
    if type(bitsize) is not int or bitsize < 1:
        raise ValueError("First-quantized dimensions must be positive integers")
    return Signature.build(r=bitsize)


def prepare_t_signature(num_bits_p: int) -> Signature:
    """Return the coordinate and two power-of-two through words."""
    width = prepare_power_two_signature(num_bits_p).get_left("r").bitsize
    return Signature.build(
        w=(KINETIC_COORDINATE_COUNT - 1).bit_length(), r=width, s=width
    )


def prepare_kinetic_coordinate_on(
    target: tuple[int, ...],
    *,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
) -> _BoundBlockOperation:
    """Bind the uniform coordinate substep on w for circuit.append."""
    gates = uniform_preparation_ir(
        KINETIC_COORDINATE_COUNT, target, workspace, classical_bits, adjoint=adjoint
    )
    return _BoundBlockOperation(gates, max((*target, *workspace)) + 1)
