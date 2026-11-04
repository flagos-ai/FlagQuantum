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
) -> CircuitIR:
    """Emit lookup, or erase loaded data using X measurement and feedback.

    Wires are the enable, big-endian address, data, and clean traversal scratch.
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
    if measurement_bits and (
        len(measurement_bits) != data_width or len(set(measurement_bits)) != data_width
    ):
        raise ValueError("QROM erasure requires a distinct classical bit per data wire")

    data_start = 1 + width
    operand_count = data_start + data_width
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

    def apply_block(block: int, predicate: int | None) -> None:
        assert predicate is not None  # The enable is always a quantum control.
        if not measurement_bits:
            for offset in range(data_width):
                if words[block] & (1 << (data_width - offset - 1)):
                    instructions.append(
                        Instruction("cx", (predicate, data_start + offset))
                    )
            return

        # Within each pair the correction is z on the block predicate for
        # the first row, plus cz on predicate and low bit for the row XOR.
        # A measurement bit controls each term, so their phases add mod 2.
        even = words[2 * block]
        odd = words[2 * block + 1] if 2 * block + 1 < len(words) else 0
        for offset, bit in enumerate(measurement_bits):
            mask = 1 << (data_width - offset - 1)
            condition = {"conditions": ((bit, 1),)}
            if even & mask:
                instructions.append(Instruction("z", (predicate,), metadata=condition))
            if (even ^ odd) & mask:
                instructions.append(
                    Instruction("cz", (predicate, width), metadata=condition)
                )

    paired = bool(measurement_bits)
    append_unary_iteration(
        instructions,
        operand_count=operand_count,
        width=width - int(paired),
        entries=(len(words) + 1) // 2 if paired else len(words),
        control_value=1,
        apply_branch=apply_block,
        coherent=True,
    )
    return CircuitIR(operand_count + width, tuple(instructions))
