"""Emit uniform interval preparation in the established register bit order."""

import math

from ....core._composition import controlled_instruction
from ....core.ir import Instruction


def _uniform_interval(
    register: tuple[int, ...], start: int, stop: int
) -> tuple[Instruction, ...]:
    """Split the real amplitudes by the populations in each binary prefix."""
    gates: list[Instruction] = []
    for depth, target in enumerate(register):
        half = 1 << (len(register) - depth - 1)
        controls = register[:depth]
        for prefix in range(start // (2 * half), (stop - 1) // (2 * half) + 1):
            left = 2 * half * prefix
            middle = left + half
            zeros = max(0, min(stop, middle) - max(start, left))
            ones = max(0, min(stop, middle + half) - max(start, middle))
            if not ones:
                continue
            angle = 2 * math.atan2(math.sqrt(ones), math.sqrt(zeros))
            rotation = Instruction("ry", (target,), {"theta": angle})
            flips = tuple(
                Instruction("x", (bit,))
                for index, bit in enumerate(controls)
                if not prefix & (1 << (depth - index - 1))
            )
            gates.extend(flips)
            gates.extend(
                controlled_instruction(rotation, controls) if controls else (rotation,)
            )
            gates.extend(flips)
    return tuple(gates)
