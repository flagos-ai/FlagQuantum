# Adapted from Qualtran v0.7.0, qualtran/bloqs/data_loading/qrom.py and qroam_clean.py.
# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs/data_loading
# Changes: FlagQuantum IR, one controlled address, and two-row blocks.

"""Controlled XOR lookup and paired-block measurement-assisted erasure."""

from collections.abc import Sequence

from ..core.ir import CircuitIR, Instruction
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

    data_start = 1 + width
    operand_count = data_start + banks * data_width
    instructions: list[Instruction] = []
    if measurement_bits:
        for offset, bit in enumerate(measurement_bits):
            wire = data_start + offset
            instructions.extend(
                (
                    Instruction("h", (wire,)),
                    Instruction(
                        "measure",
                        (wire,),
                        metadata={"is_dynamic": True, "classical_bit": bit},
                    ),
                    Instruction("x", (wire,), metadata={"conditions": ((bit, 1),)}),
                )
            )

    swaps = (
        [
            Instruction("cswap", (width, data_start + i, data_start + data_width + i))
            for i in range(data_width)
        ]
        if blocked
        else []
    )
    if inverse and not measurement_bits:
        instructions.extend(swaps)

    def apply_block(block: int, predicate: int | None) -> None:
        assert predicate is not None  # The enable is always a quantum control.
        row = banks * block if not measurement_bits else 2 * block
        even = words[row]
        odd = words[row + 1] if row + 1 < len(words) else 0
        if not measurement_bits:
            values = (even, odd) if blocked else (even,)
            for bank, word in enumerate(values):
                for offset in range(data_width):
                    if word & (1 << (data_width - offset - 1)):
                        wire = data_start + bank * data_width + offset
                        instructions.append(Instruction("cx", (predicate, wire)))
            return

        # The low address bit swaps the two banks. Its phase correction
        # is the difference (XOR) between the unswapped and swapped words.
        for offset, bit in enumerate(measurement_bits):
            bank, position = divmod(offset, data_width)
            mask = 1 << (data_width - position - 1)
            condition = {"conditions": ((bit, 1),)}
            base_word = odd if bank else even
            if base_word & mask:
                instructions.append(Instruction("z", (predicate,), metadata=condition))
            if (even ^ odd) & mask:
                instructions.append(
                    Instruction("cz", (predicate, width), metadata=condition)
                )

    paired = blocked or bool(measurement_bits)
    append_unary_iteration(
        instructions,
        operand_count=operand_count,
        width=width - int(paired),
        entries=(len(words) + 1) // 2 if paired else len(words),
        control_value=1,
        apply_branch=apply_block,
        coherent=True,
    )
    if not inverse and not measurement_bits:
        instructions.extend(swaps)
    return CircuitIR(operand_count + width, tuple(instructions))
