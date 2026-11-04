"""Fourier transform instructions shared by Circuit algorithms and synthesis."""

import math

from ..core.ir import Instruction


def qft_instructions(wires: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Build the exact forward QFT for wires ordered most significant first."""
    gates: list[Instruction] = []
    for index, target in enumerate(wires):
        gates.append(Instruction("h", (target,)))
        for distance, control in enumerate(wires[index + 1 :], start=1):
            gates.append(
                Instruction(
                    "cphase",
                    (control, target),
                    {"theta": math.ldexp(math.pi, -distance)},
                )
            )
    for index in range(len(wires) // 2):
        gates.append(Instruction("swap", (wires[index], wires[-1 - index])))
    return tuple(gates)
