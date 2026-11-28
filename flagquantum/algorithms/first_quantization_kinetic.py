# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 kinetic PREPARE and SELECT register contracts.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/prepare_t.py

"""Kinetic register contracts and executable coordinate preparation."""

from ..compiler.registers.preparation.uniform_preparation import uniform_preparation_ir
from ..core.ir.program.register_types import opaque_register_type
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from .primitives._block_binding import _BoundBlockOperation

KINETIC_COORDINATE_COUNT = 3


def prepare_power_two_signature(bitsize: int) -> Signature:
    """Return the locked power-of-two declaration's through word."""
    if type(bitsize) is not int or bitsize < 1:
        raise ValueError("First-quantized dimensions must be positive integers")
    return _kinetic_declaration(bitsize, power=True)


def prepare_t_signature(num_bits_p: int) -> Signature:
    """Return the coordinate and two power-of-two through words."""
    prepare_power_two_signature(num_bits_p)
    return _kinetic_declaration(num_bits_p)


def select_t_signature(num_bits_p: int) -> Signature:
    """Return the kinetic control flags, coordinate indices, and three axes."""
    return _select_t_declaration(num_bits_p)


def _select_t_declaration(num_bits_p: SymbolicInt) -> Signature:
    ports = Signature.build(flag_T=1, plus=1, w=3)
    coordinate = opaque_register_type(num_bits_p)
    return Signature(
        (
            *ports,
            Register("r", coordinate),
            Register("s", coordinate),
            Register("p", coordinate, shape=(3,)),
        )
    )


def projectile_prepare_power_two_signature(bitsize_n: int, bitsize_p: int) -> Signature:
    """Return the projectile unary word while checking both momentum widths."""
    if type(bitsize_p) is not int or bitsize_p < 1:
        raise ValueError("First-quantized dimensions must be positive integers")
    signature = prepare_power_two_signature(bitsize_n)
    if bitsize_n < bitsize_p:
        raise ValueError(
            "Projectile momentum width must be at least the electron width"
        )
    return signature


def projectile_prepare_t_signature(num_bits_p: int, num_bits_n: int) -> Signature:
    """Add the projectile mean-coordinate word to the through kinetic ports."""
    projectile_prepare_power_two_signature(num_bits_n, num_bits_p)
    return _kinetic_declaration(num_bits_n, projectile=True)


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


def _kinetic_declaration(
    width: SymbolicInt, *, projectile: bool = False, power: bool = False
) -> Signature:
    """Retain source Signature.build zero omission and opaque symbolic ports."""
    if power:
        return Signature.build(r=width)
    return (
        Signature.build(w=2, w_mean=2, r=width, s=width)
        if projectile
        else Signature.build(w=2, r=width, s=width)
    )
