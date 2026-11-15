# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 nuclear index and potential PREPARE signatures.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization

"""Register contracts for potential-energy and nuclear-index preparation."""

from ..core.ir.program.register_types import opaque_register_type
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature


def prepare_zeta_signature(num_atoms: int) -> Signature:
    """Keep the through nuclear index, including the single-atom empty word."""
    if type(num_atoms) is not int or num_atoms < 1:
        raise ValueError("First-quantized dimensions must be positive integers")
    return Signature(
        (Register("l", opaque_register_type((num_atoms - 1).bit_length())),)
    )
