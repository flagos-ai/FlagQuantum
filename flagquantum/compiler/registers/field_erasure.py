# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, GF2MulMBUC in gf2_multiplication.py.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py
# Changes: explicit qubits, measured-slot cleanup, shared classical-parity phases.

"""Measurement-assisted erasure of a known binary-field product."""

from dataclasses import dataclass

from ...core.classical_dataflow import ClassicalXor
from ...core.ir import Instruction
from .field_multiplication import _remainder, _validate_field
from .measurement import ParityPhase, measure_x_clear_ir
from .polynomial import _validate_registers


@dataclass(frozen=True)
class FieldProductErasure:
    """The source-level measurement order and logical phase correction calls."""

    measurements: tuple[tuple[int, int], ...]
    corrections: tuple[ParityPhase, ...]
    parities: tuple[tuple[int, ...], ...]

    def lower(self, *, parity_bit: int) -> tuple[Instruction, ...]:
        """Emit source calls with one reusable caller-owned classical parity slot."""
        ClassicalXor((), parity_bit)  # Validate without reserving a quantum qubit.
        if any(parity_bit == bit for _, bit in self.measurements):
            raise ValueError("Parity scratch must be distinct from measurement slots")
        instructions = [
            gate
            for wire, bit in self.measurements
            for gate in measure_x_clear_ir(wire, bit)
        ]
        width = len(self.measurements)
        offset = width * (width + 1) // 2
        for correction in self.corrections[:offset]:
            instructions.extend(correction.lower())
        for i, sources in enumerate(self.parities):
            instructions.append(ClassicalXor(sources, parity_bit).instruction())
            count = width - i - 1
            for correction in self.corrections[offset : offset + count]:
                instructions.extend(
                    ParityPhase(correction.qubits, (parity_bit,)).lower()
                )
            offset += count
        return tuple(instructions)


def gf2_product_erasure(
    left: tuple[int, ...],
    right: tuple[int, ...],
    product: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    modulus: int,
) -> FieldProductErasure:
    """Port GF2MulMBUC's measurement and parity-controlled CZ call sequence.

    Registers and their corresponding classical slots are big-endian. Product
    must start in left*right, possibly entangled with the inputs; it ends in zero
    in every branch. Input registers are preserved, including complex phases.
    The classical outcome bits remain available to the caller.
    """
    n = len(left)
    _validate_field(n, modulus)
    _validate_registers(left, right, product)
    if len(right) != n or len(product) != n or len(classical_bits) != n:
        raise ValueError(
            "Product erasure requires equal register and measurement widths"
        )
    if len(set(classical_bits)) != n or any(
        type(b) is not int or b < 0 for b in classical_bits
    ):
        raise ValueError("Product erasure requires distinct nonnegative classical bits")
    x, y, result = left[::-1], right[::-1], classical_bits[::-1]
    corrections: list[ParityPhase] = []
    parities: list[tuple[int, ...]] = []
    for i in range(n):
        for j in range(i + 1):
            corrections.append(ParityPhase((x[j], y[i - j]), (result[i],)))
    for i in range(n):
        column = _remainder(1 << (n + i), modulus)
        parity = tuple(result[k] for k in range(n) if column >> k & 1)
        parities.append(parity)
        for j in range(i + 1, n):
            corrections.append(ParityPhase((x[n - j + i], y[j]), parity))
    return FieldProductErasure(
        tuple(zip(product[::-1], result, strict=True)),
        tuple(corrections),
        tuple(parities),
    )
