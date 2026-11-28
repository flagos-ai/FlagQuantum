# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 momentum-transfer PREPARE signatures.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/prepare_nu.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/projectile/prepare_nu.py

"""Register contracts and executable uniform word of momentum-transfer PREPARE."""

from ..compiler.registers.preparation.uniform_preparation import uniform_preparation_ir
from ..core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ..core.ir.program.registers import Register, RegisterSide
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from .primitives._block_binding import _BoundBlockOperation


def _sample_width(num_bits_p: int, m_param: int) -> int:
    if any(type(n) is not int or n < 1 for n in (num_bits_p, m_param)):
        raise ValueError("Momentum width and m_param must be positive integers")
    return (m_param - 1).bit_length()


def _nu_declarations(
    width: SymbolicInt, sample_width: SymbolicInt, *, adjoint: bool = False
) -> dict[str, Signature]:
    """Share source logical formulas without claiming physical width admission."""
    mu = Register("mu", opaque_register_type(width))
    nu = Register("nu", opaque_register_type(width + 1), shape=(3,))
    m = Register("m", opaque_register_type(sample_width))
    produced = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
    consumed = RegisterSide.RIGHT if adjoint else RegisterSide.LEFT
    bit = quantum_bit_type()
    return {
        "PrepareNuState": Signature((mu, nu, m, Register("flag_nu", bit))),
        "PrepareMuUnaryEncodedOneHot": Signature(
            (mu, Register("flag", bit, side=produced))
        ),
        "PrepareNuSuperPositionState": Signature((mu, nu)),
        "FlagZeroAsFailure": Signature(
            (nu, Register("flag_minus_zero", bit, side=produced))
        ),
        "TestNuLessThanMu": Signature(
            (mu, nu, Register("flag_nu_lt_mu", bit, side=produced))
        ),
        "TestNuInequality": Signature(
            (
                mu,
                nu,
                m,
                Register("flag_minus_zero", bit, side=consumed),
                Register("flag_mu_prep", bit, side=consumed),
                Register("flag_ineq", bit, side=consumed),
                Register("succ", bit),
            )
        ),
    }


def prepare_nu_signature(num_bits_p: int, m_param: int) -> Signature:
    """Return the source through words, including the externally owned success bit."""
    return _nu_declarations(num_bits_p, _sample_width(num_bits_p, m_param))[
        "PrepareNuState"
    ]


def prepare_nu_leaf_signatures(
    num_bits_p: int, m_param: int, *, adjoint: bool = False
) -> dict[str, Signature]:
    """Describe the five declaration-only children and directional flag ownership."""
    declarations = _nu_declarations(
        num_bits_p, _sample_width(num_bits_p, m_param), adjoint=adjoint
    )
    del declarations["PrepareNuState"]
    return declarations


def projectile_prepare_mu_signature(num_bits_p: int, num_bits_n: int) -> Signature:
    """Return the projectile μ word and the forward-produced success flag."""
    if any(type(width) is not int or width < 1 for width in (num_bits_p, num_bits_n)):
        raise ValueError("Momentum widths must be positive integers")
    if num_bits_n < num_bits_p:
        raise ValueError(
            "Projectile momentum width must be at least the electron width"
        )
    return _nu_declarations(num_bits_n, 0)["PrepareMuUnaryEncodedOneHot"]


def projectile_prepare_nu_leaf_signatures(
    num_bits_p: int, num_bits_n: int, m_param: int, *, adjoint: bool = False
) -> dict[str, Signature]:
    """Reuse the shared ν contracts and reverse the projectile μ flag ownership."""
    mu = projectile_prepare_mu_signature(num_bits_p, num_bits_n)
    leaves = prepare_nu_leaf_signatures(num_bits_n, m_param, adjoint=adjoint)
    del leaves["PrepareMuUnaryEncodedOneHot"]
    return {
        "PrepareMuUnaryEncodedOneHotWithProj": mu.adjoint() if adjoint else mu,
        **leaves,
    }


def prepare_nu_m_on(
    m_param: int,
    target: tuple[int, ...],
    *,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool = False,
) -> _BoundBlockOperation:
    """Bind the composite's executable m-register stage for circuit.append."""
    gates = uniform_preparation_ir(
        m_param, target, workspace, classical_bits, adjoint=adjoint
    )
    return _BoundBlockOperation(gates, max((*target, *workspace), default=-1) + 1)
