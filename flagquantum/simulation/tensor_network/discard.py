# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Trace labels adapted from Qualtran v0.7.0 basic_gates/discard.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/discard.py

"""Supply classical and quantum discard indices to mixed component wiring."""

from collections.abc import Callable, Mapping

from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import classical_bit_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide


def discard_tensor_indices(
    function: FunctionRecord,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    _fresh_label: Callable[[], int],
) -> tuple[int, ...]:
    """Yield the consumed scalar bit's existing numerical trace label.

    The caller explicitly binds this supplier to each discard function in
    ComponentTensorProviders. A matching signature alone never discovers a
    discard, since projections and releases can have the same typed ports.
    Mixed wiring owns conjugate tracing and direct-input effects. Pure wiring
    retains its existing discard refusal. No tensor, label or physical resource
    is allocated here.
    """
    ports = function.registers
    if ports not in (
        (Register("c", classical_bit_type(), side=RegisterSide.LEFT),),
        (Register("q", quantum_bit_type(), side=RegisterSide.LEFT),),
    ):
        raise ValueError("discard requires one scalar CBit c or QBit q LEFT port")
    name = ports[0].name
    if set(incoming) != {name} or outgoing or len(incoming[name]) != 1:
        raise ValueError("discard tensor bindings must match the consumed scalar port")
    word = incoming[name][0]
    if len(word) != 1 or type(word[0]) is not int or word[0] < 0:
        raise ValueError("discard requires one nonnegative integer bit label")
    return word
