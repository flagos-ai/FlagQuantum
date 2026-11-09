# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModAdd.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py
# Changes: native IR, shared register components and explicit reusable workspace.

"""Modular register addition through the original sign-correction call chain."""

from ....core.ir import Instruction
from ..arithmetic.addition import (
    add_adjoint_ir,
    add_constant_adjoint_ir,
    add_constant_ir,
    add_ir,
)
from ..arithmetic.controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from ..comparison.linear_comparison import linear_greater_ir


def modular_add_requirements(width: int) -> tuple[int, int]:
    """Return explicit clean quantum work and reusable classical slot budgets."""
    if type(width) is not int or width < 1:
        raise ValueError("Modular arithmetic requires a positive concrete width")
    return 2 * width + 2, width


def _validate_modular_layout(
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    requirements: tuple[int, int],
    *,
    extra_qubits: tuple[int, ...] = (),
) -> None:
    """Share operand/domain/layout validation across actual modular consumers."""
    n, (q, c) = len(left), requirements
    if len(right) != n:
        raise ValueError("Modular arithmetic requires equal operand widths")
    if type(modulus) is not int or not 1 <= modulus < 1 << n:
        raise ValueError("Modulus must be a positive integer fitting the operand")
    wires = (*extra_qubits, *left, *right, *workspace)
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError(
            "Modular arithmetic requires distinct nonnegative quantum wires"
        )
    if len(workspace) != q:
        raise ValueError(f"Modular arithmetic requires {q} clean work wires")
    if (
        len(classical_bits) != c
        or len(set(classical_bits)) != c
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            f"Modular arithmetic requires {c} distinct nonnegative classical slots"
        )


def _modular_workspace(
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate and bind one shared layout for modular add/subtract consumers."""
    n = len(left)
    q, c = modular_add_requirements(n)
    _validate_modular_layout(left, right, modulus, workspace, classical_bits, (q, c))
    # The extended-left zero prefix dies after the first Add. Reuse it as
    # the constant MSB only after that component has restored it to zero.
    constant, carry = (workspace[0], *workspace[2 : n + 2]), workspace[n + 2 :]
    # The comparator's last carry reuses the same physical/classical pair as
    # the adder's last carry. Other comparator carries occupy constant[2:].
    compare = constant[:1] if n == 1 else (*constant, carry[-1])
    return workspace[0], workspace[1], constant, carry, compare


def modular_add_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Preserve x and update y=(x+y)%p on x in [0,p], y in [0,p).

    All registers are big-endian. Work starts and ends zero on this promised
    domain; input amplitudes must have no support outside it. The compiler does
    not inspect quantum values. The same layout is reused only after each
    shared component cleans its work; measurement slots hold latest values.
    """
    n = len(left)
    junk, sign, constant, carry, compare = _modular_workspace(
        left, right, modulus, workspace, classical_bits
    )
    extended_left, extended_right = (junk, *left), (sign, *right)
    return (
        *add_ir(extended_left, extended_right, carry, classical_bits),
        *add_constant_ir(extended_right, -modulus, constant, carry, classical_bits),
        *controlled_add_constant_ir(
            sign, right, modulus, constant[:n], carry[: n - 1], classical_bits[: n - 1]
        ),
        *linear_greater_ir(left, right, sign, compare, classical_bits),
        Instruction("x", (sign,)),
    )


def modular_add_adjoint_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Subtract x modulo p using the original ModAdd adjoint component order.

    Big-endian x lies in [0,p], y in [0,p). Reuse the forward work layout and
    original constants. The unsigned comparator's source component palindrome
    is independently checked before reuse. Quantum work starts/ends zero;
    classical slots retain latest outcomes. This is distinct from ModSub.
    """
    n = len(left)
    junk, sign, constant, carry, compare = _modular_workspace(
        left, right, modulus, workspace, classical_bits
    )
    return (
        Instruction("x", (sign,)),
        *linear_greater_ir(left, right, sign, compare, classical_bits),
        *controlled_add_constant_adjoint_ir(
            sign, right, modulus, constant[:n], carry[: n - 1], classical_bits[: n - 1]
        ),
        *add_constant_adjoint_ir(
            (sign, *right), -modulus, constant, carry, classical_bits
        ),
        *add_adjoint_ir((junk, *left), (sign, *right), carry, classical_bits),
    )
