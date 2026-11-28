# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 nuclear index and potential PREPARE/SELECT signatures.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization

"""Register contracts for potential energy and nuclear-index preparation."""

from ..core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from .first_quantization_nu import prepare_nu_signature


def prepare_zeta_signature(num_atoms: int) -> Signature:
    """Keep the through nuclear index, including the single-atom empty word."""
    if type(num_atoms) is not int or num_atoms < 1:
        raise ValueError("First-quantized dimensions must be positive integers")
    return _zeta_declaration(num_atoms)


def _zeta_declaration(num_atoms: int) -> Signature:
    """Retain source index metadata independently of executable atom admission."""
    return Signature(
        (Register("l", opaque_register_type((num_atoms - 1).bit_length())),)
    )


def prepare_uv_signature(num_bits_p: int, num_atoms: int, m_param: int) -> Signature:
    """Compose the ν and nuclear-index words in the source's UV order."""
    mu, nu, m, flag = prepare_nu_signature(num_bits_p, m_param)
    (index,) = prepare_zeta_signature(num_atoms)
    return Signature((mu, nu, m, index, flag))


def select_uv_signature(
    num_bits_p: int, num_atoms: int, num_bits_nuc_pos: int
) -> Signature:
    """Return the potential flags, nuclear words, and three oriented axis banks."""
    prepare_zeta_signature(num_atoms)
    return _select_uv_declaration(num_bits_p, num_atoms, num_bits_nuc_pos)


def _select_uv_declaration(
    num_bits_p: SymbolicInt, num_atoms: int, num_bits_nuc_pos: SymbolicInt
) -> Signature:
    (index,) = _zeta_declaration(num_atoms)
    return Signature(
        (
            Register("flag_tuv", quantum_bit_type()),
            Register("flag_uv", quantum_bit_type()),
            index,
            Register("rl", opaque_register_type(num_bits_nuc_pos)),
            Register("nu", opaque_register_type(num_bits_p + 1), shape=(3,)),
            Register("p", opaque_register_type(num_bits_p), shape=(3,)),
            Register("q", opaque_register_type(num_bits_p), shape=(3,)),
        )
    )


def apply_nuclear_phase_signature(num_bits_p: int, num_bits_nuc: int) -> Signature:
    """Keep the source's capitalized Rl and distinct nuclear and momentum widths."""
    return _nuclear_phase_declaration(num_bits_p, num_bits_nuc)


def _nuclear_phase_declaration(
    num_bits_p: SymbolicInt, num_bits_nuc: SymbolicInt
) -> Signature:
    position = opaque_register_type(num_bits_nuc)
    return Signature(
        (
            Register("l", position),
            Register("Rl", position),
            Register("nu", opaque_register_type(num_bits_p), shape=(3,)),
        )
    )
