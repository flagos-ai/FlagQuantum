# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Coefficient normalization follows locked Qualtran v0.7.0 SelectPauliLCU.

"""Convert external Pauli metadata using the original sign-of-real rule."""

from collections.abc import Iterable
from math import isnan
from operator import index as integer_index
from typing import SupportsInt


def pauli_select_width(value: object, name: str) -> int:
    """Normalize source Python integer and concrete symbolic integer widths.

    Booleans use the original integer protocol. Concrete SymPy integers execute
    in the source SDK generator even though its stored graph refuses them.
    NumPy integer widths and floating widths fail in the original QAny owner.
    """
    message = f"Cirq Pauli SELECT requires a nonnegative {name} width"
    if isinstance(value, int):
        width = integer_index(value)
    else:
        from sympy import Integer
        if not isinstance(value, Integer):
            raise ValueError(message)
        width = int(value)
    if width < 0:
        raise ValueError(message)
    return width


def pauli_select_control(value: object) -> int | None:
    """Use the existing concrete binary activation owner at the SDK boundary."""
    if value is None:
        return None
    from ..compiler.formats.legacy_conjunction_factories import _control_value
    from ..errors import SerializationError

    message = "Cirq Pauli SELECT control value must be zero, one or None"
    try:
        result = _control_value(value)
    except SerializationError:
        raise ValueError(message) from None
    if type(result) is not int:
        raise ValueError(message)
    return result


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
