# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix util/arg_parse.cc at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Source integer grammar and directly rounded host C floating conversions."""

import ctypes
import os
from typing import Literal


def parse_int64(text: str) -> int:
    if not text:
        raise ValueError("Integer argument cannot be empty")
    negative = text.startswith("-")
    digits = text[1:] if text.startswith(("-", "+")) else text
    accumulator = 0
    for digit in digits:
        if digit == "_" and accumulator:
            continue
        if not "0" <= digit <= "9":
            raise ValueError(f"Invalid int64 argument {text!r}")
        numeric = ord(digit) - ord("0")
        mask = (1 << 64) - 1
        following = (accumulator * 10 + numeric) & mask
        # Preserve the original uint64 overflow predicate, including wrap aliases.
        if accumulator != ((following - numeric) & mask) // 10:
            raise ValueError(f"Int64 argument overflows {text!r}")
        accumulator = following
    maximum = 1 << 63 if negative else (1 << 63) - 1
    if accumulator > maximum:
        raise ValueError(f"Int64 argument overflows {text!r}")
    return -accumulator if negative else accumulator


def parse_c_float(text: str, precision: Literal[32, 64]) -> float:
    """Use strtof or strtod on an owned NUL-terminated UTF-8 buffer."""
    runtime = ctypes.CDLL("msvcrt" if os.name == "nt" else None)
    function = runtime.strtof if precision == 32 else runtime.strtod
    function.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p))
    function.restype = ctypes.c_float if precision == 32 else ctypes.c_double
    encoded = text.encode("utf-8")
    buffer = ctypes.create_string_buffer(encoded)
    endpoint = ctypes.c_char_p()
    value = function(buffer, ctypes.byref(endpoint))
    if endpoint.value:
        raise ValueError(f"Invalid floating argument {text!r}")
    if not isinstance(value, float):
        raise TypeError("C floating conversion returned an invalid result")
    return value
