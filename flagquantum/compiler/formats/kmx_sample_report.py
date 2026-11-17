# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix main_sample.cc and big_int_to_str.

"""Render the physical register sample table in source CSV order."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .kmb import KmxRegister


@dataclass(frozen=True)
class RegisterSample:
    """One shot, in declaration order, with the separate sign column."""

    before: tuple[int, ...]
    after: tuple[int, ...]
    negative: bool


def _digits(value: int, width: int, radix: Literal[2, 10, 16]) -> str:
    if radix == 10:
        return str(value)
    if not width:
        return ""
    if radix == 16:
        return format(value, f"0{(width + 3) // 4}X")
    return format(value, f"0{width}b")


def format_sample_report(
    registers: Sequence[KmxRegister],
    rows: Iterable[RegisterSample],
    *,
    radix: Literal[2, 10, 16],
) -> str:
    """Render paired before/after columns with the fixed source padding."""
    names = (reg.name or f"r{i}" for i, reg in enumerate(registers))
    fields = [column for name in names for column in (f"(before){name}", name)]
    lines = [",".join((*fields, "global_phase"))]
    padding = tuple(6 + len(str(i)) for i in range(len(registers)))
    for row in rows:
        halves = zip(registers, padding, row.before, row.after, strict=True)
        values = [
            _digits(value, len(register.members), radix).rjust(pad)
            for register, pad, before, after in halves
            for value in (before, after)
        ]
        sign = -1 if row.negative else 1
        lines.append(",".join((*values, f"{sign:12}")))
    return "\n".join((*lines, ""))
