"""Fourier transform instructions shared by Circuit algorithms and synthesis."""

import math
from collections.abc import Iterator

from ...core._composition import inverted_instruction
from ...core.ir import Instruction


def qft_instructions(wires: tuple[int, ...]) -> tuple[Instruction, ...]:
    """Build the exact forward QFT for qubits ordered most significant first."""
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


def _fourier_add(
    source: tuple[int, ...], target: tuple[int, ...], *, subtract: bool = False
) -> Iterator[Instruction]:
    transform = qft_instructions(target)
    yield from transform
    for power, control in enumerate(reversed(source)):
        for position, wire in enumerate(target):
            if power <= position:
                angle = math.ldexp(math.pi, power - position)
                yield Instruction(
                    "cphase", (control, wire), {"theta": -angle if subtract else angle}
                )
    yield from (inverted_instruction(gate) for gate in reversed(transform))


def borrowed_sum_xor(
    source: tuple[int, ...],
    borrowed: tuple[int, ...],
    target: tuple[int, ...],
    *,
    shift: int = 0,
    signed: bool = False,
) -> Iterator[Instruction]:
    """XOR (source << shift) + borrowed, restoring both arbitrary operands.

    The target has one overflow bit above borrowed. Temporarily including that
    arbitrary target bit in a Fourier adder captures carry by XOR. Restoring
    borrowed modulo its own width leaves the carry in place. No clean work is
    required; the exact QFT is emitted without angle truncation.
    """
    wires = (*source, *borrowed, *target)
    if not source or not borrowed or len(target) != len(borrowed) + 1:
        raise ValueError("borrowed sum requires operands and one target overflow bit")
    if any(isinstance(wire, bool) or not isinstance(wire, int) for wire in wires):
        raise TypeError("borrowed sum wires must be Python integers")
    if min(wires) < 0 or len(set(wires)) != len(wires):
        raise ValueError("borrowed sum wires must be nonnegative and disjoint")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError("borrowed sum shift must be a Python integer")
    if not 0 <= shift < len(borrowed) or len(source) + shift > len(borrowed):
        raise ValueError("borrowed sum shifted source must fit the borrowed word")
    if not isinstance(signed, bool):
        raise TypeError("borrowed sum signed flag must be Boolean")
    high = borrowed[: len(borrowed) - shift]
    if signed:
        yield Instruction("cx", (borrowed[0], target[0]))
    yield from _fourier_add(source, (target[0], *high))
    for control, wire in zip(high, target[1 : len(high) + 1], strict=True):
        yield Instruction("cx", (control, wire))
    yield from _fourier_add(source, high, subtract=True)
    for control, wire in zip(
        borrowed[len(high) :], target[len(high) + 1 :], strict=True
    ):
        yield Instruction("cx", (control, wire))
