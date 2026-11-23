# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix sim/sim.cc SimInitInstruction.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Parse physical initialization text without register binding or execution."""

import re
from dataclasses import dataclass

from ...core._fixed_width_integer import FixedWidthInteger

_EXAMPLES = "\n    Examples of valid init strings: r0=5, r1<0x500, r2=*"


@dataclass(frozen=True)
class RegisterInitializationText:
    """Own an unresolved register index, declared-width value and random flag."""

    register: int
    value: FixedWidthInteger
    randomize: bool

    def __str__(self) -> str:
        operator = "<" if self.randomize else "="
        return f"r{self.register}{operator}{self.value.decimal()}"


def parse_initialization(text: str) -> RegisterInitializationText:
    """Retain absent index digits, uint32 admission and wildcard normalization."""
    match = re.match(r"r([0-9]*)", text)
    if match is None:
        raise ValueError(f"Failed to parse init instruction: '{text}'")
    digits = match[1].lstrip("0") or "0"
    if len(digits) > 10 or int(digits) > (1 << 32) - 1:
        raise ValueError(f"Register id is out of range in '{text}'")
    index = int(digits)
    rest = text[match.end() :]
    if rest == "=*":
        return RegisterInitializationText(index, FixedWidthInteger(), True)
    if not rest.startswith(("=", "<")):
        raise ValueError(f"Failed to parse init instruction: '{text}'")
    return RegisterInitializationText(
        index, FixedWidthInteger.from_text(rest[1:]), rest[0] == "<"
    )


def parse_initializations(text: str) -> tuple[RegisterInitializationText, ...]:
    """Parse comma-separated records, allowing one source trailing separator."""
    try:
        if not text:
            return ()
        return tuple(
            parse_initialization(item) for item in text.removesuffix(",").split(",")
        )
    except ValueError as error:
        raise ValueError(str(error) + _EXAMPLES) from error
