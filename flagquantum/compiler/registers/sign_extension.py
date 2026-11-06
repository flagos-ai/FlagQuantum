# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from pinned Qualtran v0.7.0 SignExtend, SignTruncate and MultiTargetCNOT.
# Sources: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/conversions/sign_extension.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_target_cnot.py
# Changes: native IR and explicit prefix ownership; fanout extracted from subtraction.

"""Sign-extension/truncation gates on a caller-owned prefix."""

from ...core.ir import Instruction
from .bitwise import fanout_ir


def sign_extend_ir(
    source: tuple[int, ...], prefix: tuple[int, ...]
) -> tuple[Instruction, ...]:
    """Copy a sign into zero prefix bits, or clear a correct redundant prefix.

    The extended big-endian logical output is prefix+source. Applying this same
    source fanout to a correctly sign-extended value implements SignTruncate,
    returning prefix to zero. Arbitrary high bits cannot be discarded safely.
    """
    wires = (*source, *prefix)
    if not source or not prefix:
        raise ValueError("Sign extension requires nonempty source and prefix")
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Sign extension requires distinct nonnegative quantum wires")
    return fanout_ir(source[0], prefix)
