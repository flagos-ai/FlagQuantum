# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0, gf2_inverse.py and gf2_square.py.
# Source: https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/gf_arithmetic
# Changes: native IR, explicit scratch, correct linear adjoints, clean-output wrapper.

"""Itoh--Tsujii inversion and explicit compute-copy-uncompute cleanup."""

from ....core.ir import Instruction
from .field_linear import _square
from .field_multiplication import (
    FieldDecomposition,
    _FieldEmitter,
    _validate_field,
)
from .polynomial import _validate_registers


def gf2_inverse_decomposition(
    source: tuple[int, ...], workspace: tuple[int, ...], *, modulus: int
) -> FieldDecomposition:
    """Port GF2Inverse's raw call chain, including its logical junk outputs.

    Source and outputs are big-endian coefficient words. Zero maps to zero.
    The caller supplies clean workspace for result, junk, and recursive scratch.
    This raw decomposition does not erase the upstream intermediate products.
    """
    n = len(source)
    _validate_field(n, modulus)
    _validate_registers(source, workspace)
    emitter = _FieldEmitter(workspace, modulus)

    def add(left: tuple[int, ...], right: tuple[int, ...]) -> None:
        for a, b in zip(left, right, strict=True):
            emitter.cx(a, b)

    def multiply(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
        target = emitter.workspace.allocate(n).wires
        return emitter.multiply(left[::-1], right[::-1], target[::-1])[::-1]

    if n == 1:
        result = emitter.workspace.allocate(n).wires
        add(source, result)
        return FieldDecomposition(tuple(emitter.instructions), (source, result), ())

    t = (n - 1).bit_count()
    k1 = (n - 1).bit_length() - 1
    k = max(k1 + t - 1, k1 + 1)
    f = [source] + [()] * k
    temporary = emitter.workspace.allocate(n)
    f[k] = temporary.wires
    for i in range(1, k1 + 1):
        add(f[i - 1], f[k])
        f[k] = _square(emitter, f[k], 2 ** (i - 1))
        f[i] = multiply(f[i - 1], f[k])
        f[k] = _square(emitter, f[k], 2 ** (i - 1), adjoint=True)
        add(f[i - 1], f[k])
    bits = [-1] + [i for i in reversed(range(k1 + 1)) if (n - 1) >> i & 1]
    if k1 + t - 1 == k:
        emitter.workspace.free(temporary, order=f[k])
    for s in range(1, t):
        f[k1 + s - 1] = _square(emitter, f[k1 + s - 1], 2 ** bits[s + 1])
        f[k1 + s] = multiply(f[k1 + s - 1], f[bits[s + 1]])
    if t == 1:
        if k1 == 0:
            add(f[0], f[k])
        f[k1], f[k] = f[k], f[k1]
    f[k] = _square(emitter, f[k])
    outputs = (f[0], f[k], *f[1:k])
    logical = {w for reg in outputs for w in reg}
    scratch = tuple(
        wire for wire in emitter.workspace.used_wires if wire not in logical
    )
    return FieldDecomposition(tuple(emitter.instructions), outputs, scratch)


def gf2_inverse_ir(
    source: tuple[int, ...],
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    modulus: int,
) -> tuple[Instruction, ...]:
    """XOR the inverse into target and erase all upstream intermediate products.

    Target need not be zero. Workspace must start in zero and returns to zero.
    Source is preserved on its original physical qubits, including GF(2) and GF(4).
    """
    _validate_registers(source, target, workspace)
    if len(source) != len(target):
        raise ValueError("Field inversion requires equal register widths")
    raw = gf2_inverse_decomposition(source, workspace, modulus=modulus)
    copy = tuple(
        Instruction("cx", (a, b)) for a, b in zip(raw.outputs[1], target, strict=True)
    )
    return (*raw.instructions, *copy, *reversed(raw.instructions))


def gf2_inverse_workspace_size(width: int, *, modulus: int) -> int:
    """Return the exact clean workspace required by this concrete call chain."""
    _validate_field(width, modulus)
    # At most two registers per exponent bit, plus recursive polynomial scratch.
    bound = width * (2 * (width - 1).bit_length() + 2)
    source = tuple(range(width))
    raw = gf2_inverse_decomposition(
        source, tuple(range(width, width + bound)), modulus=modulus
    )
    allocated = set(raw.scratch) | {w for reg in raw.outputs for w in reg}
    return len(allocated - set(source))
