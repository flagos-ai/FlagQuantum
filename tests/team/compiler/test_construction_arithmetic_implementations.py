"""Independent complete-domain checks for reversible AO-02 arithmetic kernels."""

from itertools import product

import pytest
import torch

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum.compiler._arithmetic_kernels import (
    constant_multiply_accumulate,
    square_accumulate,
    xor_add_parts,
)
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["square", "constant3", "constant5", "constant10"])
def test_accumulation_every_input_and_arbitrary_accumulator_and_inverse(
    width: int, kind: str
) -> None:
    coefficient = int(kind.removeprefix("constant")) if kind != "square" else 0
    out_width = (
        2 * width
        if kind == "square"
        else (coefficient * ((1 << width) - 1)).bit_length()
    )
    x = tuple(range(width))
    out = tuple(range(width, width + out_width))
    count = out_width if kind == "square" else max(0, out_width - width) + 1
    scratch = tuple(range(width + out_width, width + out_width + count))
    gates = tuple(
        square_accumulate(x, out, scratch)
        if kind == "square"
        else constant_multiply_accumulate(x, coefficient, out, scratch)
    )
    circuit = CircuitIR(width + out_width + count, gates)
    inverse = CircuitIR(circuit.n_wires, tuple(reversed(gates)))
    for value, target in product(range(1 << width), range(1 << out_width)):
        answer = value * value if kind == "square" else coefficient * value
        initial = encode((value, target), (x, out))
        expected = encode((value, (target + answer) % (1 << out_width)), (x, out))
        assert classical_apply(circuit, initial) == expected
        assert classical_apply(inverse, expected) == initial


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("same", [False, True])
def test_xor_add_every_input_and_output_without_assuming_zero_target(
    width: int, same: bool
) -> None:
    left = tuple(range(width))
    right = left if same else tuple(range(width, 2 * width))
    start = width if same else 2 * width
    out = tuple(range(start, start + width + 1))
    scratch = () if same else tuple(range(start + width + 1, start + 2 * width + 1))
    prefix, writes = xor_add_parts(left, right, out, scratch)
    circuit = CircuitIR(
        start + width + 1 + len(scratch), (*prefix, *writes, *reversed(prefix))
    )
    for x, y, target in product(
        range(1 << width), range(1 if same else 1 << width), range(1 << (width + 1))
    ):
        actual_y = x if same else y
        registers, values = (
            ((left, out), (x, target)) if same else ((left, right, out), (x, y, target))
        )
        initial = encode(values, registers)
        answer = classical_apply(circuit, initial)
        expected_values = (
            (x, target ^ (x + actual_y)) if same else (x, y, target ^ (x + actual_y))
        )
        assert answer == encode(expected_values, registers)
        assert classical_apply(circuit, answer) == initial


@pytest.mark.parametrize("constant", [0, 1, 3, 5, 16])
def test_xor_add_constant_unequal_width_and_all_high_target_bits(constant: int) -> None:
    left, out = (0, 1, 2), tuple(range(3, 11))
    scratch = tuple(range(11, 18))
    prefix, writes = xor_add_parts(left, constant, out, scratch)
    circuit = CircuitIR(18, (*prefix, *writes, *reversed(prefix)))
    for x, target in product(range(8), range(256)):
        assert classical_apply(circuit, encode((x, target), (left, out))) == encode(
            (x, target ^ (x + constant)), (left, out)
        )


@pytest.mark.parametrize("kind", ["square", "constant", "xor_add"])
def test_complete_complex_amplitudes_preserve_inputs_targets_and_zero_scratch(
    kind: str,
) -> None:
    x = (0, 1)
    if kind == "xor_add":
        y, out, scratch = (2, 3), (4, 5, 6), (7, 8)
        prefix, writes = xor_add_parts(x, y, out, scratch)
        gates = (*prefix, *writes, *reversed(prefix))
        registers = (x, y, out)
        values = list(product(range(4), range(4), range(8)))
    else:
        out, scratch = (2, 3, 4, 5), (6, 7, 8, 9) if kind == "square" else (6, 7, 8)
        gates = tuple(
            square_accumulate(x, out, scratch)
            if kind == "square"
            else constant_multiply_accumulate(x, 3, out, scratch)
        )
        registers = (x, out)
        values = list(product(range(4), range(16)))
    n = max(scratch) + 1
    amplitudes = torch.randn(
        len(values),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(20261102),
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)

    def index(items: tuple[int, ...]) -> int:
        encoded = encode(items, registers)
        return int(f"{encoded:0{n}b}"[::-1], 2)

    for row, items in enumerate(values):
        initial[index(items)] = amplitudes[row]
        a, target = items[0], items[-1]
        answer = (
            target ^ (a + items[1])
            if kind == "xor_add"
            else (target + (a * a if kind == "square" else 3 * a)) % 16
        )
        expected[index((*items[:-1], answer))] = amplitudes[row]
    observed = (
        fq.Circuit.from_ir(CircuitIR(n, gates), inputs=initial, dtype=torch.complex128)
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("kernel", ["square", "constant", "xor_add"])
def test_kernels_reject_output_alias_and_incorrect_scratch(kernel: str) -> None:
    def build(
        output: tuple[int, ...], scratch: tuple[int, ...]
    ) -> tuple[Instruction, ...]:
        if kernel == "square":
            return tuple(square_accumulate((0,), output, scratch))
        if kernel == "constant":
            return tuple(constant_multiply_accumulate((0,), 3, output, scratch))
        prefix, writes = xor_add_parts((0,), (1,), output, scratch)
        return (*prefix, *writes, *reversed(prefix))

    with pytest.raises(ValueError):
        build((0,), (2,))
    with pytest.raises(ValueError):
        build((2, 3), ())


@pytest.mark.parametrize("constant", [True, -1, 1.5])
def test_constant_kernel_does_not_coerce_invalid_integers(constant: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        tuple(constant_multiply_accumulate((0,), constant, (1, 2), (3, 4)))
