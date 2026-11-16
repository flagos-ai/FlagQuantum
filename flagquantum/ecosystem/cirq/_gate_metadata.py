# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 CirqGateAsBloq and wire/tensor hooks.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/cirq_interop/_cirq_to_bloq.py

"""Read Cirq gate metadata and convert it to owned register records."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ._register_export import _register_shape
from .conversion import _cirq

if TYPE_CHECKING:
    from ...drawer.register_score_data import SymbolClass


def cirq_gate_signature(gate: object) -> Signature:
    """Derive the raw SDK contract using concrete binary through ports."""
    cirq = _cirq()
    if not isinstance(gate, cirq.Gate):
        raise TypeError("Cirq metadata requires a gate")
    shape = cirq.qid_shape(gate)
    if any(dimension != 2 for dimension in shape):
        raise ValueError("Cirq metadata requires binary quantum ports")
    width = len(shape)
    if width == 0:
        return Signature(())
    return Signature(
        (Register("q", quantum_bit_type(), (width,) if width > 1 else ()),)
    )


def cirq_gate_tensor(
    gate: object, signature: Signature | None = None
) -> NDArray[np.complex128]:
    """Return the unitary tensor on ordered output bits followed by input bits."""
    from ...simulation.tensor_network.register_tensor import register_tensor_from_matrix

    raw = cirq_gate_signature(gate)
    sig = raw if signature is None else signature
    for register in sig:
        _register_shape(register)
    cirq = _cirq()
    if not cirq.has_unitary(gate):
        raise NotImplementedError("Cirq gate tensors require a unitary gate")
    matrix = np.asarray(cirq.unitary(gate), dtype=np.complex128)
    data = register_tensor_from_matrix(matrix, sig)
    n_bits = sum(
        prod(_register_shape(register))
        for side in (sig.rights(), sig.lefts())
        for register in side
    )
    return np.asarray(data).reshape((2,) * n_bits)


def cirq_gate_wire_symbol(
    gate: object,
    signature: Signature,
    wire_register: Register,
    index: tuple[int, ...] = (),
) -> tuple[SymbolClass, dict[str, str | int | bool]]:
    """Select the source wire label and translate it to a native glyph."""
    labels = _cirq().circuit_diagram_info(gate).wire_symbols
    offset = 0
    symbol = wire_register.name
    if index:
        symbol += "[" + ", ".join(map(str, index)) + "]"
    for register in signature:
        *shape, width = _register_shape(register)
        elements = prod(shape)
        if register == wire_register:
            item = (
                0 if elements == 1 else int(np.ravel_multi_index(index, tuple(shape)))
            )
            # The locked diagram uses the first element-count slice even for words.
            symbol = labels[offset + item]
            break
        offset += elements * width
    if symbol in ("@", "@(0)"):
        return "Circle", {"filled": symbol == "@"}
    if symbol == "X":
        return "ModPlus", {}
    if wire_register.side == RegisterSide.LEFT:
        return "RarrowTextBox", {"text": symbol}
    if wire_register.side == RegisterSide.RIGHT:
        return "LarrowTextBox", {"text": symbol}
    return "TextBox", {"text": symbol}
