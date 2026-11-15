# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the tutorial by Noureldin Yosri, DOI 10.5281/zenodo.8384491,
# at Qualtran v0.7.0, arithmetic/t_complexity_of_comparison_gates.ipynb.

"""Bind the tutorial's comparison circuits to native IR and append execution."""

from collections import Counter
from collections.abc import Callable, Iterator

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.comparison.comparison_tree import (
    bi_qubits_mixer_ir,
    single_qubit_compare_ir,
)
from flagquantum.compiler.registers.comparison.equality import (
    equality_constant_ir,
    equality_ir,
)
from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

Gates = tuple[Instruction, ...]
Flags = tuple[int, int, int]


def constant_equality() -> FieldOperation:
    """Original three-bit equality with 5, one XOR output and two clean flags."""
    return FieldOperation(6, equality_constant_ir((0, 1, 2), 5, 3, (4, 5), (0, 1)))


def register_equality() -> FieldOperation:
    """Original two-word equality using XOR, an all-zero conjunction and XOR."""
    return FieldOperation(6, equality_ir((0, 1), (2, 3), 4, (5,), (0,)))


def _copy_flags(sources: Flags, outputs: Flags) -> Gates:
    return tuple(Instruction("cx", pair) for pair in zip(sources, outputs, strict=True))


def _compare_flags(
    a: int, b: int, flags: tuple[int, int], outputs: Flags, slot: int
) -> Gates:
    less, greater = flags
    return (
        *single_qubit_compare_ir(a, b, less, greater),
        *_copy_flags((less, b, greater), outputs),
        *single_qubit_compare_ir(
            a, b, less, greater, adjoint=True, classical_bits=(slot,)
        ),
    )


def one_bit_comparison() -> FieldOperation:
    """Literal coherent two-CCX construction from the one-bit section."""
    a, b, less, greater = 0, 1, 5, 6
    prepare = (
        Instruction("x", (a,)),
        Instruction("ccx", (a, b, less)),
        Instruction("x", (a,)),
        Instruction("cx", (a, b)),
        Instruction("x", (b,)),
        Instruction("x", (greater,)),
        Instruction("cx", (b, greater)),
        Instruction("cx", (less, greater)),
    )
    return FieldOperation(
        7, (*prepare, *_copy_flags((less, b, greater), (2, 3, 4)), *reversed(prepare))
    )


def simple_mixer(
    left: tuple[int, int] = (0, 1), right: tuple[int, int] = (2, 3), flag: int = 4
) -> Gates:
    """Prepare high-bit inequality and perform both exact controlled swaps."""
    return (
        Instruction("cx", (left[0], flag)),
        Instruction("cx", (right[0], flag)),
        Instruction("cswap", (flag, *left)),
        Instruction("cswap", (flag, *right)),
    )


def two_bit_comparison() -> FieldOperation:
    """Use the illustrative CSWAP mixer and the original measured one-bit leaf."""
    # The original helper reverses both tuples before constructing CSWAP.
    mix = (
        Instruction("cx", (0, 7)),
        Instruction("cx", (2, 7)),
        Instruction("cswap", (7, 1, 0)),
        Instruction("cswap", (7, 3, 2)),
    )
    undo = tuple(reversed(mix))
    return FieldOperation(
        10, (*mix, *_compare_flags(1, 3, (8, 9), (4, 5, 6), 0), *undo)
    )


def comparison_tree(width: int = 3) -> FieldOperation:
    """Use source floor-split recursion, then erase subtrees in reverse order."""
    if type(width) is not int or width < 1:
        raise ValueError("Comparison tree width must be a positive integer")
    left, right = tuple(range(width)), tuple(range(width, 2 * width))
    output = (2 * width, 2 * width + 1, 2 * width + 2)
    pool = iter(range(2 * width + 3, 5 * width + 2))
    slot = iter(range(2 * width - 1))
    inverse: list[Gates] = []

    def mix(a: tuple[int, ...], b: tuple[int, ...]) -> Iterator[Instruction]:
        if len(a) == 1:
            return
        half = len(a) // 2
        yield from mix(a[:half], b[:half])
        yield from mix(a[half:], b[half:])
        x, y = (a[half - 1], a[-1]), (b[half - 1], b[-1])
        aux = (next(pool), next(pool), next(pool))
        yield from bi_qubits_mixer_ir(x, y, aux)
        inverse.append(
            bi_qubits_mixer_ir(
                x, y, aux, adjoint=True, classical_bits=(next(slot), next(slot))
            )
        )

    forward = tuple(mix(left, right))
    leaf = _compare_flags(
        left[-1], right[-1], (next(pool), next(pool)), output, next(slot)
    )
    return FieldOperation(
        5 * width + 2,
        (*forward, *leaf, *(gate for block in reversed(inverse) for gate in block)),
    )


def classical_mixer(
    value: int, *, work: tuple[int, int, int] = (2, 3, 4)
) -> tuple[Gates, Gates]:
    """Port the original two-bit classical mixer, including reversed erasure."""
    if type(value) is not int or not 0 <= value < 4:
        raise ValueError("Classical mixer value must fit two bits")
    flag, q, aux = work
    high, low = value >> 1, value & 1
    before = (
        *((Instruction("x", (q,)),) if low else ()),
        Instruction("cx", (0, flag)),
        *((Instruction("x", (flag,)),) if high else ()),
        Instruction("cx", (1, 0)),
    )
    after = (
        Instruction("cx", (aux, 1)),
        Instruction("cx", (1, 0)),
        *((Instruction("cx", (flag, q)),) if high != low else ()),
    )
    return (
        (*before, *and_compute_ir(flag, 0, aux), *after),
        (*reversed(after), *and_uncompute_ir(flag, 0, aux, 0), *reversed(before)),
    )


def two_bit_classical_comparison(value: int = 2) -> FieldOperation:
    """Compare a quantum two-bit word with a constant using the original mixer."""
    forward, inverse = classical_mixer(value, work=(5, 6, 7))
    leaf = _compare_flags(1, 6, (8, 9), (2, 3, 4), 1)
    return FieldOperation(10, (*forward, *leaf, *inverse))


def linear_constant_comparison() -> FieldOperation:
    """Original n-And prefix chain for comparing a three-bit word with 5."""
    prefix = 4
    initial = Instruction("x", (prefix,))
    forward = [initial]
    inverse: list[Gates] = [(initial,)]
    for bit, value in enumerate((1, 0, 1)):
        successor = prefix + 1
        if value:
            flip = Instruction("x", (bit,))
            update = Instruction("cx", (prefix, successor))
            forward.extend(
                (
                    flip,
                    *and_compute_ir(bit, prefix, successor),
                    Instruction("cx", (successor, 3)),
                    update,
                )
            )
            inverse.append(
                (update, *and_uncompute_ir(bit, prefix, successor, bit), flip)
            )
        else:
            forward.extend(and_compute_ir(prefix, bit, successor, values=(1, 0)))
            inverse.append(and_uncompute_ir(prefix, bit, successor, bit, values=(1, 0)))
        prefix = successor
    return FieldOperation(
        8, (*forward, *(gate for block in reversed(inverse) for gate in block))
    )


def show_counts(operation: FieldOperation, source_t: int | None = None) -> None:
    """Show emitted IR counts separately from an explicitly selected source profile."""
    print(
        "Emitted primitives",
        dict(Counter(gate.name for gate in operation.instructions)),
    )
    if source_t is not None:
        print("Original model T count", source_t)


def check_comparison(
    operation: FieldOperation,
    input_width: int,
    output_width: int,
    predicate: Callable[[int], int],
    *,
    uniform: bool,
) -> None:
    """Check every input or the full uniform state, with zero work at output."""
    labels = torch.arange(1 << input_width, dtype=torch.int64)
    trailing = operation.n_qubits - input_width
    before = labels << trailing
    after = before | (
        torch.tensor([predicate(i) for i in range(len(labels))])
        << (trailing - output_width)
    )
    for indices in (
        (slice(None),)
        if uniform
        else tuple(slice(i, i + 1) for i in range(len(labels)))
    ):
        initial = torch.zeros(1 << operation.n_qubits, dtype=torch.complex128)
        expected = torch.zeros_like(initial)
        initial[before[indices]] = expected[after[indices]] = 1
        expected /= initial.norm()
        initial /= initial.norm()
        circuit = DynamicCircuit(
            operation.n_qubits, inputs=initial, dtype=torch.complex128
        )
        circuit.append(operation)
        samples = run_dynamic(circuit, shots=1, seed=37, strategy="trajectory")
        torch.testing.assert_close(
            samples.final_states.flatten(), expected, atol=1e-12, rtol=0
        )
    print("Uniform complex state verified" if uniform else "All basis outputs verified")
