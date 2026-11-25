# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Signatures follow locked Qualtran v0.7.0 Hubbard and TwoBitFFFT declarations.

"""Shared source word and mode layouts, independent of physical lattice binding."""

from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt


def spin_word_signature(length: SymbolicInt) -> Signature:
    """Declare two source spin words of side width L, rather than L**2 modes."""
    return Signature((Register("system", opaque_register_type(length), shape=(2,)),))


def four_mode_signature() -> Signature:
    """Declare the plaquette's ordered four-mode array."""
    return Signature((Register("qubits", quantum_bit_type(), shape=(4,)),))


def fermionic_pair_signature() -> Signature:
    """Declare the source Fourier pair in x/y order."""
    return Signature(
        (Register("x", quantum_bit_type()), Register("y", quantum_bit_type()))
    )
