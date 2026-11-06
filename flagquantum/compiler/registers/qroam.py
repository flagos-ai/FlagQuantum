# Adapted from Qualtran v0.7.0, qualtran/bloqs/data_loading/qroam_clean.py.
# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/quantumlib/Qualtran/tree/v0.7.0/qualtran/bloqs/data_loading
# Changes: FlagQuantum IR, one controlled address, and two-row blocks.

"""Paired-block lookup, swap, and measurement-assisted erasure."""

from collections.abc import Sequence

from ...core.ir import CircuitIR, Instruction
from .._component_trace.trace import ComponentCall, ComponentTrace
from .measurement import ParityPhase, measure_x_clear_ir
from .unary_iteration import append_unary_iteration


def paired_lookup_ir(
    width: int,
    data_width: int,
    words: Sequence[int],
    *,
    measurement_bits: tuple[int, ...],
    blocked: bool,
    inverse: bool,
    trace: ComponentTrace | None = None,
    trace_offset: int = 0,
) -> CircuitIR:
    """Lower paired lookup or erasure for a validated QROM table."""
    banks = 2 if blocked else 1
    data_start = 1 + width
    operand_count = data_start + banks * data_width
    instructions: list[Instruction] = []
    if measurement_bits:
        for offset, bit in enumerate(measurement_bits):
            wire = data_start + offset
            instructions.extend(measure_x_clear_ir(wire, bit))

    measurement_stop = len(instructions)
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
        row = 2 * block
        even = words[row]
        odd = words[row + 1] if row + 1 < len(words) else 0
        if not measurement_bits:
            for bank, word in enumerate((even, odd)):
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
            base_word = odd if bank else even
            if base_word & mask:
                instructions.extend(ParityPhase((predicate,), (bit,)).lower())
            if (even ^ odd) & mask:
                instructions.extend(ParityPhase((predicate, width), (bit,)).lower())

    traversal_start = len(instructions)
    append_unary_iteration(
        instructions,
        operand_count=operand_count,
        width=width - 1,
        entries=(len(words) + 1) // 2,
        control_value=1,
        apply_branch=apply_block,
        coherent=True,
    )
    traversal_stop = len(instructions)
    if not inverse and not measurement_bits:
        instructions.extend(swaps)
    ir = CircuitIR(operand_count + width, tuple(instructions))
    if trace is not None:
        children = tuple(
            ComponentCall(name, trace_offset + start, trace_offset + stop)
            for name, start, stop in (
                ("BankMeasurement", 0, measurement_stop),
                ("BankExchange", measurement_stop, traversal_start),
                ("AddressTraversal", traversal_start, traversal_stop),
                ("BankExchange", traversal_stop, len(instructions)),
            )
            if stop > start
        )
        trace.calls.append(
            ComponentCall(
                "PairedErasure" if measurement_bits else "PairedLookup",
                trace_offset,
                trace_offset + len(instructions),
                (width, data_width, tuple(words), inverse, measurement_bits),
                children,
            )
        )
    return ir
