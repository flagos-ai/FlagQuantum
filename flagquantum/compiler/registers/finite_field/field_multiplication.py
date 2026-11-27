# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, bloqs/gf_arithmetic/gf2_multiplication.py.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py
# Changes: native IR, explicit scratch, integer polynomial arithmetic, qubit maps.

"""Concrete polynomial-basis multiplication with upstream gate ordering."""

from dataclasses import dataclass
from typing import Literal

from ....core._binary_matrix import BinaryMatrix
from ....core._binary_polynomial import remainder
from ....core.ir import Instruction
from ..primitives.swap import swap_bit_ir
from .polynomial import _PolynomialEmitter, _validate_registers

_remainder = remainder


def _product(left: int, right: int, modulus: int) -> int:
    result = 0
    while right:
        if right & 1:
            result ^= left
        left <<= 1
        right >>= 1
    return _remainder(result, modulus)


def _field_power(base: int, exponent: int, modulus: int) -> int:
    """Raise a field polynomial to a nonnegative exponent by repeated squaring."""
    value = 1
    while exponent:
        if exponent & 1:
            value = _product(value, base, modulus)
        base = _product(base, base, modulus)
        exponent >>= 1
    return value


def _validate_field(width: int, modulus: int) -> None:
    if type(modulus) is not int or width < 1 or modulus.bit_length() != width + 1:
        raise ValueError("Modulus must be a binary polynomial of the register degree")
    # Rabin's criterion; polynomial gcd uses XOR rather than integer subtraction.
    x = _remainder(2, modulus)
    power = x
    for degree in range(1, width + 1):
        power = _product(power, power, modulus)
        if degree <= width // 2:
            a, b = power ^ x, modulus
            while b:
                a, b = b, _remainder(a, b)
            if a != 1:
                raise ValueError("Modulus must be irreducible over GF(2)")
    if power != x:
        raise ValueError("Modulus must be irreducible over GF(2)")


@dataclass(frozen=True)
class FieldDecomposition:
    """Native gates and logical output qubits, including upstream virtual relabeling.

    Scratch excludes logical outputs. Every listed scratch qubit ends in zero.
    The caller owns allocation. Newly allocated result and scratch qubits must
    start in zero; logical input qubits can carry arbitrary quantum states.
    """

    instructions: tuple[Instruction, ...]
    outputs: tuple[tuple[int, ...], ...]
    scratch: tuple[int, ...]


def _field_plu(
    matrix: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Factor a binary matrix with the first available pivot in each column."""
    factors = BinaryMatrix.from_dense(matrix).plu()
    if factors is None:
        raise StopIteration
    permutation, lower, upper = factors
    return (
        upper.dense(),
        lower.dense(),
        [row.bit_length() - 1 for row in permutation.transposed().packed_rows],
    )


def _constant_matrix(width: int, value: int, modulus: int) -> list[list[int]]:
    """Share the polynomial-basis multiplier matrix between gate and graph owners."""
    columns = [_product(value, 1 << i, modulus) for i in range(width)]
    return [[(columns[j] >> i) & 1 for j in range(width)] for i in range(width)]


class _FieldEmitter(_PolynomialEmitter):
    """Field reduction and linear maps over the shared polynomial gate sink."""

    def __init__(self, workspace: tuple[int, ...], modulus: int) -> None:
        super().__init__(workspace)
        self.modulus = modulus

    def linear(
        self, qubits: tuple[int, ...], matrix: list[list[int]]
    ) -> tuple[int, ...]:
        """Qualtran SynthesizeLRCircuit with galois's first-pivot PLU order."""
        upper, lower, permutation = _field_plu(matrix)
        return self._linear_factors(qubits, upper, lower, permutation)

    def linear_with_zero_pivots(
        self, qubits: tuple[int, ...], matrix: list[list[int]]
    ) -> tuple[int, ...]:
        """Emit the original reversible recipe even for a singular input matrix."""
        factors = BinaryMatrix.from_dense(matrix).plu_with_zero_pivots()
        if factors is None:
            raise ValueError("Literal linear synthesis requires a square matrix")
        permutation, lower, upper = factors
        return self._linear_factors(
            qubits,
            upper.dense(),
            lower.dense(),
            [row.bit_length() - 1 for row in permutation.transposed().packed_rows],
        )

    def _linear_factors(
        self,
        qubits: tuple[int, ...],
        upper: list[list[int]],
        lower: list[list[int]],
        permutation: list[int],
    ) -> tuple[int, ...]:
        n = len(qubits)
        for i in range(n):
            for j in range(i + 1, n):
                if upper[i][j]:
                    self.cx(qubits[j], qubits[i])
        for i in reversed(range(n)):
            for j in reversed(range(i)):
                if lower[i][j]:
                    self.cx(qubits[j], qubits[i])
        result = list(qubits)
        column = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if permutation[column[j]] == i:
                    result[i], result[j] = result[j], result[i]
                    column[i], column[j] = column[j], column[i]
        return tuple(result)

    def constant(self, qubits: tuple[int, ...], value: int) -> tuple[int, ...]:
        n = len(qubits)
        matrix = _constant_matrix(n, value, self.modulus)
        return self.linear(qubits, matrix)

    def shift(self, qubits: tuple[int, ...], count: int) -> tuple[int, ...]:
        result = qubits
        if len(qubits) > 1:
            degrees = [i for i in range(1, len(qubits)) if self.modulus >> i & 1]
            for _ in range(count):
                result = result[-1:] + result[:-1]
                for i in degrees:
                    self.cx(result[0], result[i])
        return result

    def inverse_constant(self, qubits: tuple[int, ...], value: int) -> tuple[int, ...]:
        start = len(self.instructions)
        outputs = self.constant(qubits, value)
        gates = self.instructions[start:]
        del self.instructions[start:]
        binding = dict(zip(outputs, qubits, strict=True))
        self.instructions.extend(
            Instruction(g.name, tuple(binding[w] for w in g.wires))
            for g in reversed(gates)
        )
        return tuple(binding[w] for w in qubits)

    def multiply(
        self, x: tuple[int, ...], y: tuple[int, ...], target: tuple[int, ...]
    ) -> tuple[int, ...]:
        """GF2MulViaKaratsuba's complete forward decomposition, little-endian internally."""
        n = len(x)
        if n == 1:
            self.ccx(x[0], y[0], target[0])
            return target
        k = (n + 1) // 2
        for i in range(n - k):
            self.cx(x[k + i], x[i])
        for i in range(n - k):
            self.cx(y[k + i], y[i])
        self.polynomial(x[:k], y[:k], target[: 2 * k - 1])
        for i in range(n - k):
            self.cx(y[k + i], y[i])
        for i in range(n - k):
            self.cx(x[k + i], x[i])
        target = self.inverse_constant(target, 1 | (1 << k))
        self.polynomial(x[k:], y[k:], target[: 2 * (n - k) - 1])
        target = self.shift(target, k)
        self.polynomial(x[:k], y[:k], target[: 2 * k - 1])
        return self.constant(target, 1 | (1 << k))


def gf2_multiplication(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    modulus: int,
    algorithm: Literal["karatsuba", "quadratic"] = "karatsuba",
    plus_equal_prod: bool = False,
) -> FieldDecomposition:
    """Compute a product into a clean target; preserve upstream output qubit maps.

    All registers are big-endian. Quadratic multiplication needs no scratch;
    Karatsuba uses caller-owned clean scratch for odd recursive widths.
    Quadratic plus-equal multiplication admits an arbitrary target and retains
    the source's inverse monomial shift before the two multiplication ladders.
    """
    n = len(left)
    _validate_field(n, modulus)
    _validate_registers(left, right, target, workspace)
    if len(right) != n or len(target) != n:
        raise ValueError("Field multiplication requires equal register widths")
    if type(plus_equal_prod) is not bool:
        raise ValueError("Product accumulation flag must be boolean")
    if plus_equal_prod and algorithm != "quadratic":
        raise ValueError("Source plus-equal multiplication requires quadratic mode")
    emitter = _FieldEmitter(workspace, modulus)
    x, y, result = left[::-1], right[::-1], target[::-1]
    if algorithm == "karatsuba":
        result = emitter.multiply(x, y, result)
    elif algorithm == "quadratic":
        if plus_equal_prod and n > 1:
            degrees = [i for i in range(1, n) if modulus >> i & 1]
            for _ in range(n):
                for i in reversed(degrees):
                    emitter.cx(result[0], result[i])
                result = result[1:] + result[:1]
        for i in range(n):
            for j in range(i + 1, n):
                emitter.ccx(x[n - j + i], y[j], result[i])
        result = emitter.shift(result, n)
        for i in range(n):
            for j in range(i + 1):
                emitter.ccx(x[j], y[i - j], result[i])
    else:
        raise ValueError("Unknown field multiplication algorithm")
    return FieldDecomposition(
        tuple(emitter.instructions),
        (left, right, result[::-1]),
        emitter.workspace.used_wires,
    )


def bind_field_outputs(
    decomposition: FieldDecomposition, targets: tuple[tuple[int, ...], ...]
) -> tuple[Instruction, ...]:
    """Materialize upstream virtual permutations at a fixed-qubit consumer boundary."""
    current = [w for reg in decomposition.outputs for w in reg]
    desired = [w for reg in targets for w in reg]
    if len(current) != len(desired) or set(current) != set(desired):
        raise ValueError("Output binding must be a permutation of logical output wires")
    gates = list(decomposition.instructions)
    for i, wire in enumerate(desired):
        j = current.index(wire)
        if i != j:
            gates.extend(swap_bit_ir(current[i], current[j]))
            current[i], current[j] = current[j], current[i]
    return tuple(gates)
