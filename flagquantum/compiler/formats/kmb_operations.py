# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix op_type.h and op_type.cc.

"""Operation byte and column definitions for the KMB v2 format."""

# The middle two bits select a quantum operand column group. The high bits
# mark a classical destination and an inline condition.
QUBIT_MASK = 0x30
BIT_TARGET = 0x40
CONDITION = 0x80
INFO_TAG = 0x06
ANGLE = 0x13
PUSH = 0x81
POP = 0x01

_BASE_CODES = {
    "NEG": 0x00,
    "POP_CONDITION": POP,
    "INFO_TAG": INFO_TAG,
    "DEBUG_PRINT_EMPTY": 0x07,
    "Z": 0x10,
    "X": 0x11,
    "R": 0x12,
    "Z_POW": ANGLE,
    "DEBUG_PRINT_Q": 0x17,
    "CZ": 0x20,
    "CX": 0x21,
    "SWAP": 0x22,
    "CCZ": 0x30,
    "CCX": 0x31,
    "CSWAP": 0x32,
    "BIT_STORE0": 0x40,
    "BIT_STORE1": 0x41,
    "BIT_INVERT": 0x42,
    "DEBUG_PRINT_C": 0x47,
    "HMR": 0x50,
}

OPERATION_CODES = {
    **_BASE_CODES,
    "PUSH_CONDITION": PUSH,
    **{
        f"{name}_IF": code | CONDITION
        for name, code in _BASE_CODES.items()
        if code not in (POP, INFO_TAG)
    },
}
OPERATION_NAMES = {code: name for name, code in OPERATION_CODES.items()}

# Each entry lists the packet IDs for one arity, in physical operand order.
QUANTUM_COLUMNS: tuple[tuple[int, ...], ...] = ((), (7,), (5, 6), (2, 3, 4))
