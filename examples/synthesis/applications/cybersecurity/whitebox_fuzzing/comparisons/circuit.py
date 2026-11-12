"""Independent truth-table phase and exact Circuit Grover reflection."""

from itertools import product

import torch

import flagquantum as fq
from examples.synthesis.applications.cybersecurity.whitebox_fuzzing.main import (
    build_program,
)

WIDTHS = (6, 6)


def marked() -> tuple[int, ...]:
    result = []
    for values in product(*(range(1 << width) for width in WIDTHS)):
        x, y = values
        if x + y < 9 and (x % 4) * (y % 2) == 1:
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


def run_example() -> None:
    """Compare absolute amplitudes and independently verify both clean work tails."""
    registers = ("x", "y")
    for r in (0, 1, 2, 10):
        program = build_program(r)
        assert tuple(program.registers) == registers
        expected = (
            fq.run(reference_circuit(r)).statevector().reshape(1 << sum(WIDTHS), -1)
        )
        assert float(expected[:, 1:].abs().square().sum()) < 1e-24
        state = fq.run(program).statevector().flatten()
        width = state.numel().bit_length() - 1
        logical = tuple(wire for name in registers for wire in program.registers[name])
        hidden = tuple(wire for wire in range(width) if wire not in logical)
        ordered = (
            state.reshape((2,) * width)
            .permute((*logical, *hidden))
            .reshape(expected.shape[0], -1)
        )
        assert float(ordered[:, 1:].abs().square().sum()) < 1e-24
        torch.testing.assert_close(ordered[:, 0], expected[:, 0], atol=1e-12, rtol=0)
        print(f"r={r}: native and independent Circuit amplitudes agree.")
        print(program.report["resources"])


if __name__ == "__main__":
    run_example()
