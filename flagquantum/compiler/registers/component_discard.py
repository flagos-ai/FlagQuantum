# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Signature adapted from Qualtran v0.7.0 basic_gates/discard.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/discard.py

"""Declare quantum discard for existing typed mixed-channel consumers."""

from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide


def discard_qubit_function(name: str) -> FunctionRecord:
    """Consume one QBit with the source's signature-only quantum declaration.

    Bind its numerical trace explicitly through ComponentTensorProviders.
    No classical action or physical decomposition is supplied, matching the
    original DiscardQ. Existing physical slot reset is a separate lowering.
    """
    return FunctionRecord(
        name,
        (),
        registers=(Register("q", quantum_bit_type(), side=RegisterSide.LEFT),),
    )
