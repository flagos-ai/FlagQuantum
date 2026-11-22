# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix id/id_tagging.h, Apache-2.0.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/id/id_tagging.h

"""Physical operand tag layout and diagnostic names."""

TAG_MASK = 0xE0000000
PAYLOAD_MASK = 0x1FFFFFFF
MAX_NUM_QUBITS = 500_000_000
MAX_NUM_BITS = 500_000_000
MAX_NUM_REGISTERS = 1_000_000

_TAG_NAMES = {
    0: "BOOLEAN",
    1: "CLASSICAL",
    2: "QUANTUM",
    3: "X_BOOLEAN",
    4: "X_CLASSICAL",
    64: "MIXED_XLIKE",
    128: "MIXED_ZLIKE",
    192: "MIXED",
}


def is_scalar_tag(tag: int) -> bool:
    """Classify an unsigned byte from physical operand metadata."""
    if type(tag) is not int or not 0 <= tag <= 255:
        raise ValueError("Operand tag must be an unsigned byte")
    return tag <= 4


def format_operand_tag(tag: int) -> str:
    """Render scalar, mixed and unknown tags with native diagnostic names."""
    is_scalar_tag(tag)
    name = _TAG_NAMES.get(tag)
    if name is None:
        return f"OperandTag{{{tag}}}"
    return f"OperandTag.{name}"
