# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Coefficient selection follows Qualtran v0.7.0 SelectPauliLCU.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/select_pauli_lcu.py

"""Translate external Cirq Pauli SELECT objects into native operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from ...algorithms.primitives.multiplexers.select_pauli_lcu import SelectPauliLCU
from .._pauli_select_rows import sign_of_real_pauli_word


def select_pauli_lcu_from_cirq(source: object) -> SelectPauliLCU:
    """Convert a Qualtran-style SELECT object containing Cirq Pauli rows.

    The source operation selects the sign of each coefficient's real part and
    discards its magnitude and imaginary part. A zero real part creates a
    non-unitary source leaf, which the native unitary SELECT cannot represent.
    """
    import cirq

    external = cast(Any, source)
    try:
        selection_bitsize = external.selection_bitsize
        target_bitsize = external.target_bitsize
        select_unitaries = external.select_unitaries
        control_val = external.control_val
    except AttributeError as exc:
        raise TypeError("Cirq Pauli SELECT source is missing required fields") from exc

    if type(selection_bitsize) is not int or selection_bitsize < 0:
        raise ValueError("Cirq Pauli SELECT requires a nonnegative selection width")
    if type(target_bitsize) is not int or target_bitsize < 0:
        raise ValueError("Cirq Pauli SELECT requires a nonnegative target width")
    if control_val is not None and (
        type(control_val) is not int or control_val not in (0, 1)
    ):
        raise ValueError("Cirq Pauli SELECT control value must be zero, one or None")
    if not isinstance(select_unitaries, Sequence) or not select_unitaries:
        raise ValueError("Cirq Pauli SELECT requires at least one Pauli row")

    words: list[str] = []
    for index, row in enumerate(select_unitaries):
        if not isinstance(row, cirq.DensePauliString):
            raise TypeError(f"Cirq Pauli SELECT row {index} is not a DensePauliString")
        if len(row) != target_bitsize:
            raise ValueError(
                f"Cirq Pauli SELECT row {index} has width {len(row)}, "
                f"expected {target_bitsize}"
            )
        try:
            words.append(
                sign_of_real_pauli_word(row.pauli_mask, complex(row.coefficient))
            )
        except ValueError as error:
            raise ValueError(f"Cirq Pauli SELECT row {index} has {error}") from error

    if selection_bitsize < (len(words) - 1).bit_length():
        raise ValueError("Cirq Pauli SELECT selection width cannot address every row")

    return SelectPauliLCU(selection_bitsize, target_bitsize, tuple(words), control_val)
