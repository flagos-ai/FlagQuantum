# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, BinaryPolynomialMultiplication and
# MultiplyPolyByOnePlusXk in bloqs/gf_arithmetic/gf2_multiplication.py.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic/gf2_multiplication.py
# Changes: native IR, explicit clean workspace, big-endian register boundary.

"""Binary polynomial multiply-accumulate networks without field reduction."""

from ...core.ir import Instruction


class _PolynomialEmitter:
    """Shared recursive polynomial gates and caller-owned compile-time workspace."""

    def __init__(self, workspace: tuple[int, ...]) -> None:
        self.available = list(reversed(workspace))
        self.used: set[int] = set()
        self.instructions: list[Instruction] = []

    def allocate(self, count: int) -> tuple[int, ...]:
        if count > len(self.available):
            raise ValueError("Insufficient clean field workspace")
        wires = tuple(self.available.pop() for _ in range(count))
        self.used.update(wires)
        return wires

    def free(self, qubits: tuple[int, ...]) -> None:
        self.available.extend(reversed(qubits))

    def cx(self, source: int, target: int) -> None:
        self.instructions.append(Instruction("cx", (source, target)))

    def ccx(self, left: int, right: int, target: int) -> None:
        self.instructions.append(Instruction("ccx", (left, right, target)))

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


def _validate_registers(*registers: tuple[int, ...]) -> None:
    wires = tuple(w for register in registers for w in register)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError("Field registers require distinct nonnegative integer wires")


def binary_polynomial_workspace_size(width: int) -> int:
    """Return peak clean workspace for either polynomial accumulation variant."""
    if type(width) is not int or width < 1:
        raise ValueError("Polynomial width must be a positive concrete integer")
    if width == 1:
        return 0
    k = (width + 1) // 2
    extension = max(0, 3 * k + max(0, 2 * (width - k) - k - 1) - (2 * width - 1))
    return max(
        binary_polynomial_workspace_size(k),
        extension + binary_polynomial_workspace_size(width - k),
    )


def binary_polynomial_product_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    shift: int | None = None,
) -> tuple[Instruction, ...]:
    """XOR fg or (1+x**shift)fg into a possibly nonzero polynomial target.

    Registers are big-endian coefficients. Inputs have equal positive width n.
    The target width is 2n-1, or 2k+max(0,2n-k-1) for shift k in [1,n+1].
    Caller-owned workspace must start in zero and is restored to zero. The
    network preserves both inputs and uses only CNOT and Toffoli instructions.
    """
    width = len(left)
    required = binary_polynomial_workspace_size(width)
    _validate_registers(left, right, target, workspace)
    if len(right) != width:
        raise ValueError("Polynomial inputs require equal register widths")
    if shift is not None and (type(shift) is not int or not 1 <= shift <= width + 1):
        raise ValueError("Polynomial shift must be a concrete integer in [1, n+1]")
    target_width = (
        2 * width - 1 if shift is None else 2 * shift + max(0, 2 * width - shift - 1)
    )
    if len(target) != target_width:
        raise ValueError(f"Polynomial target requires {target_width} wires")
    if len(workspace) < required:
        raise ValueError(
            f"Polynomial accumulation requires {required} clean work wires"
        )
    emitter = _PolynomialEmitter(workspace)
    if shift is None:
        emitter.polynomial(left[::-1], right[::-1], target[::-1])
    else:
        emitter.one_plus(left[::-1], right[::-1], target[::-1], shift)
    return tuple(emitter.instructions)
