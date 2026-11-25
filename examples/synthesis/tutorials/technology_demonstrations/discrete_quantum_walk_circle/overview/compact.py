"""Exact zero-work MCX phase parity and the reversible walk cascade."""

import math

import torch

import flagquantum as fq


def _toggle_parity(
    circuit: fq.Circuit, lower: tuple[int, ...], target: int, mask: int
) -> None:
    if mask:
        circuit.cx(lower[mask.bit_length() - 1], target)


def controlled_x(circuit: fq.Circuit, controls: tuple[int, ...], target: int) -> None:
    """Use a Gray-code traversal of the exact AND parity expansion."""
    if len(controls) == 1:
        circuit.cx(controls[0], target)
        return
    wires = (*controls, target)
    angle = math.pi / (1 << len(controls))
    circuit.u3(target, math.pi / 2, 0.0, math.pi)
    for index, anchor in enumerate(wires):
        lower = wires[:index]
        previous = 0
        for word in range(1 << index):
            mask = word ^ (word >> 1)
            _toggle_parity(circuit, lower, anchor, mask ^ previous)
            circuit.u3(anchor, 0.0, 0.0, angle * (-1 if mask.bit_count() % 2 else 1))
            previous = mask
        _toggle_parity(circuit, lower, anchor, previous)
    circuit.u3(target, math.pi / 2, 0.0, math.pi)


def walk(width: int) -> fq.Circuit:
    """Keep exactly the original position wires and the coin."""
    coin = width
    data = tuple(reversed(range(width)))
    circuit = fq.Circuit(width + 1, dtype=torch.complex128)
    circuit.u3(coin, math.pi / 2, 0.0, math.pi)
    circuit.u3(coin, math.pi, 0.0, math.pi)
    for index in reversed(range(width)):
        controlled_x(circuit, (coin, *data[:index]), data[index])
    circuit.u3(coin, math.pi, 0.0, math.pi)
    for index, target in enumerate(data):
        controlled_x(circuit, (coin, *data[:index]), target)
    return circuit
