# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Coefficient normalization follows locked Qualtran v0.7.0 SelectPauliLCU.

"""Convert external Pauli metadata using the original sign-of-real rule."""

from collections.abc import Iterable
from math import isnan
from typing import SupportsInt


def sign_of_real_pauli_word(mask: Iterable[SupportsInt], coefficient: complex) -> str:
    """Discard coefficient magnitude and imaginary part before unitary admission.

    Infinite nonzero real parts have a well-defined sign. Imaginary metadata,
    including NaN and infinity, is discarded by the original nth_operation.
    NaN or zero real parts produce nonunitary source leaves and remain refused.
    """
    real = coefficient.real
    if isnan(real):
        raise ValueError("a non-finite coefficient")
    if real == 0:
        raise ValueError("zero real coefficient and is non-unitary")
    return ("-" if real < 0 else "") + "".join("IXYZ"[int(axis)] for axis in mask)
