# Adapted from Qualtran v0.7.0, qualtran/bloqs/data_loading/qrom.py.
# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs/data_loading
# Changes: FlagQuantum IR, one controlled address,.

"""Controlled XOR lookup and paired-block measurement-assisted erasure."""

from collections.abc import Sequence

from ...core.ir import CircuitIR, Instruction
from .qroam import paired_lookup_ir
from .unary_iteration import append_unary_iteration


def controlled_qrom_ir(
    width: int,
    data_width: int,
    words: Sequence[int],
    *,
    measurement_bits: tuple[int, ...] = (),
    blocked: bool = False,
    inverse: bool = False,
) -> CircuitIR:
    """Emit lookup, or erase loaded data using X measurement and feedback.

    Wires are the enable, big-endian address, data, optional junk bank, and
    clean traversal scratch. Blocked mode loads two words and swaps the
    addressed word into the data bank. Its inverse swaps before XOR lookup.
    Erasure requires the data to still hold the word selected by the same
    enable and address. Unlisted addresses load zero. Classical bits are
    supplied by the caller so the fragment can be placed on any qubits.
    """
    if width < 1 or data_width < 1 or not 0 < len(words) <= 1 << width:
        raise ValueError(
            "QROM requires a nonempty table that fits its address register"
        )
    if any(not 0 <= word < 1 << data_width for word in words):
        raise ValueError("QROM words must fit the data register")
    banks = 2 if blocked else 1
    if measurement_bits and (
        len(measurement_bits) != banks * data_width
        or len(set(measurement_bits)) != banks * data_width
    ):
        raise ValueError("QROM erasure requires a distinct classical bit per data wire")

    if blocked or measurement_bits:
        return paired_lookup_ir(
            width,
            data_width,
            words,
            measurement_bits=measurement_bits,
            blocked=blocked,
            inverse=inverse,
        )

    data_start = 1 + width
    operand_count = data_start + data_width
    instructions: list[Instruction] = []

    def load(address: int, predicate: int | None) -> None:
        assert predicate is not None
        for offset in range(data_width):
            if words[address] & (1 << (data_width - offset - 1)):
                instructions.append(Instruction("cx", (predicate, data_start + offset)))

    append_unary_iteration(
        instructions,
        operand_count=operand_count,
        width=width,
        entries=len(words),
        control_value=1,
        apply_branch=load,
        coherent=True,
    )
    return CircuitIR(operand_count + width, tuple(instructions))
