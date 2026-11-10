"""Independent truth-table phase and exact Circuit Grover reflection."""

from itertools import product

import torch

import flagquantum as fq

WIDTHS = (2, 2, 2, 2, 2)


def marked() -> tuple[int, ...]:
    result = []
    for values in product(*(range(1 << width) for width in WIDTHS)):
        a, b, c, d, e = values
        if (
            a not in {0, 1}
            and a != b
            and b not in {0, 2}
            and b != c
            and b != e
            and c != 0
            and c != d
            and c != e
            and d not in {1, 2}
            and e != 3
        ):
            index = 0
            for value, width in zip(values, WIDTHS, strict=True):
                index = (index << width) | value
            result.append(index)
    return tuple(result)


def reference_circuit(r: int) -> fq.Circuit:
    size = sum(WIDTHS)
    circuit = fq.Circuit(2 * size - 3, dtype=torch.complex128)
    scratch = tuple(range(size, 2 * size - 3))

    def sign(index: int) -> None:
        # Independently constructed AND ladder implements a basis-state sign.
        flips = tuple(bit for bit in range(size) if not index >> (size - 1 - bit) & 1)
        for bit in flips:
            circuit.x(bit)
        circuit.ccx(0, 1, scratch[0])
        for bit in range(2, size - 2):
            circuit.ccx(scratch[bit - 2], bit, scratch[bit - 1])
        circuit.h(size - 1).ccx(scratch[-1], size - 2, size - 1).h(size - 1)
        for bit in reversed(range(2, size - 2)):
            circuit.ccx(scratch[bit - 2], bit, scratch[bit - 1])
        circuit.ccx(0, 1, scratch[0])
        for bit in reversed(flips):
            circuit.x(bit)

    for bit in range(size):
        circuit.h(bit)
    for _ in range(r):
        for index in marked():
            sign(index)
        for bit in range(size):
            circuit.h(bit)
        sign(0)
        circuit.any(0, matrix=-torch.eye(2, dtype=torch.complex128))
        for bit in range(size):
            circuit.h(bit)
    return circuit
