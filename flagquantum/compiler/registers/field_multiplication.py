# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, bloqs/gf_arithmetic/gf2_multiplication.py.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py
# Changes: native IR, explicit scratch, integer polynomial arithmetic, wire maps.

"""Concrete polynomial-basis multiplication with upstream gate ordering."""

from dataclasses import dataclass
from typing import Literal

from ...core.ir import Instruction


def _remainder(value: int, modulus: int) -> int:
    while value.bit_length() >= modulus.bit_length():
        value ^= modulus << (value.bit_length() - modulus.bit_length())
    return value


def _product(left: int, right: int, modulus: int) -> int:
    result = 0
    while right:
        if right & 1:
            result ^= left
        left <<= 1
        right >>= 1
    return _remainder(result, modulus)


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
    """Native gates and logical output wires, including upstream virtual relabeling.

    Scratch excludes logical outputs. Every listed scratch wire ends in zero.
    The caller owns allocation and must initialize all output/scratch wires to zero.
    """

    instructions: tuple[Instruction, ...]
    outputs: tuple[tuple[int, ...], ...]
    scratch: tuple[int, ...]


class _FieldEmitter:
    """Compile-only allocator and gate sink shared by field call chains."""

    def __init__(self, workspace: tuple[int, ...], modulus: int) -> None:
        self.available = list(reversed(workspace))
        self.used: set[int] = set()
        self.instructions: list[Instruction] = []
        self.modulus = modulus

    def allocate(self, count: int) -> tuple[int, ...]:
        if count > len(self.available):
            raise ValueError("Insufficient clean field workspace")
        wires = tuple(self.available.pop() for _ in range(count))
        self.used.update(wires)
        return wires

    def free(self, wires: tuple[int, ...]) -> None:
        self.available.extend(reversed(wires))

    def cx(self, source: int, target: int) -> None:
        self.instructions.append(Instruction("cx", (source, target)))

    def ccx(self, left: int, right: int, target: int) -> None:
        self.instructions.append(Instruction("ccx", (left, right, target)))

    def linear(
        self, wires: tuple[int, ...], matrix: list[list[int]]
    ) -> tuple[int, ...]:
        """Qualtran SynthesizeLRCircuit with galois's first-pivot PLU order."""
        n = len(wires)
        upper = [row[:] for row in matrix]
        lower = [[0] * n for _ in range(n)]
        permutation = list(range(n))
        for i in range(n):
            pivot = next(j for j in range(i, n) if upper[j][i])
            upper[i], upper[pivot] = upper[pivot], upper[i]
            lower[i], lower[pivot] = lower[pivot], lower[i]
            permutation[i], permutation[pivot] = permutation[pivot], permutation[i]
            lower[i][i] = 1
            for j in range(i + 1, n):
                if upper[j][i]:
                    lower[j][i] = 1
                    upper[j] = [a ^ b for a, b in zip(upper[j], upper[i], strict=True)]
        for i in range(n):
            for j in range(i + 1, n):
                if upper[i][j]:
                    self.cx(wires[j], wires[i])
        for i in reversed(range(n)):
            for j in reversed(range(i)):
                if lower[i][j]:
                    self.cx(wires[j], wires[i])
        result = list(wires)
        column = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if permutation[column[j]] == i:
                    result[i], result[j] = result[j], result[i]
                    column[i], column[j] = column[j], column[i]
        return tuple(result)

    def constant(self, wires: tuple[int, ...], value: int) -> tuple[int, ...]:
        n = len(wires)
        columns = [_product(value, 1 << i, self.modulus) for i in range(n)]
        matrix = [[(columns[j] >> i) & 1 for j in range(n)] for i in range(n)]
        return self.linear(wires, matrix)

    def shift(self, wires: tuple[int, ...], count: int) -> tuple[int, ...]:
        result = wires
        if len(wires) > 1:
            degrees = [i for i in range(1, len(wires)) if self.modulus >> i & 1]
            for _ in range(count):
                result = result[-1:] + result[:-1]
                for i in degrees:
                    self.cx(result[0], result[i])
        return result

    def inverse_constant(self, wires: tuple[int, ...], value: int) -> tuple[int, ...]:
        start = len(self.instructions)
        outputs = self.constant(wires, value)
        gates = self.instructions[start:]
        del self.instructions[start:]
        binding = dict(zip(outputs, wires, strict=True))
        self.instructions.extend(
            Instruction(g.name, tuple(binding[w] for w in g.wires))
            for g in reversed(gates)
        )
        return tuple(binding[w] for w in wires)

    def one_plus(
        self, f: tuple[int, ...], g: tuple[int, ...], h: tuple[int, ...], k: int
    ) -> None:
        n = len(f)
        length = max(0, 2 * n - k - 1)
        if n == 1:
            self.cx(h[k], h[0])
            self.ccx(f[0], g[0], h[k])
            self.cx(h[k], h[0])
            return
        for i in reversed(range(length)):
            self.cx(h[2 * k + i], h[k + i])
        for i in range(k):
            self.cx(h[k + i], h[i])
        self.polynomial(f, g, h[k : 2 * k + length])
        for i in range(k):
            self.cx(h[k + i], h[i])
        for i in range(length):
            self.cx(h[2 * k + i], h[k + i])

    def polynomial(
        self, f: tuple[int, ...], g: tuple[int, ...], h: tuple[int, ...]
    ) -> None:
        n = len(f)
        if n == 1:
            self.ccx(f[0], g[0], h[0])
            return
        k = (n + 1) // 2
        self.one_plus(f[:k], g[:k], h[: 3 * k - 1], k)
        width = 2 * k + max(0, 2 * (n - k) - k - 1)
        delta = max(0, k + width - len(h))
        aux = self.allocate(delta)
        extended = h + aux
        self.one_plus(f[k:], g[k:], extended[k : k + width], k)
        self.free(aux)
        for i in range(n - k):
            self.cx(f[k + i], f[i])
        for i in range(n - k):
            self.cx(g[k + i], g[i])
        self.polynomial(f[:k], g[:k], h[k : 3 * k - 1])
        for i in range(n - k):
            self.cx(g[k + i], g[i])
        for i in range(n - k):
            self.cx(f[k + i], f[i])

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


def _validate_registers(*registers: tuple[int, ...]) -> None:
    wires = tuple(w for register in registers for w in register)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Field registers require distinct nonnegative integer wires")


def gf2_multiplication(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    modulus: int,
    algorithm: Literal["karatsuba", "quadratic"] = "karatsuba",
) -> FieldDecomposition:
    """Compute a product into a clean target; preserve upstream output wire maps.

    All registers are big-endian. Quadratic multiplication needs no scratch;
    Karatsuba uses caller-owned clean scratch for odd recursive widths.
    """
    n = len(left)
    _validate_field(n, modulus)
    _validate_registers(left, right, target, workspace)
    if len(right) != n or len(target) != n:
        raise ValueError("Field multiplication requires equal register widths")
    emitter = _FieldEmitter(workspace, modulus)
    x, y, result = left[::-1], right[::-1], target[::-1]
    if algorithm == "karatsuba":
        result = emitter.multiply(x, y, result)
    elif algorithm == "quadratic":
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
        tuple(sorted(emitter.used)),
    )


def bind_field_outputs(
    decomposition: FieldDecomposition, targets: tuple[tuple[int, ...], ...]
) -> tuple[Instruction, ...]:
    """Materialize upstream virtual permutations at a fixed-wire consumer boundary."""
    current = [w for reg in decomposition.outputs for w in reg]
    desired = [w for reg in targets for w in reg]
    if len(current) != len(desired) or set(current) != set(desired):
        raise ValueError("Output binding must be a permutation of logical output wires")
    gates = list(decomposition.instructions)
    for i, wire in enumerate(desired):
        j = current.index(wire)
        if i != j:
            gates.append(Instruction("swap", (current[i], current[j])))
            current[i], current[j] = current[j], current[i]
    return tuple(gates)
