# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModDbl.build_composite_bloq.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py
# Changes: native shared arithmetic and explicit source output/work port aliases.

"""Modular doubling retaining the original logical output-qubit exchange."""

from ...core.ir import Instruction
from .arithmetic import add_constant_ir
from .bitwise import _validate_wires
from .controlled_addition import controlled_add_constant_ir


def modular_double_requirements(width: int) -> tuple[int, int]:
    """Reserve lower/sign, n+2 constant bits, and n+1 shared carries."""
    if type(width) is not int or width < 1:
        raise ValueError("Modular doubling requires a positive concrete width")
    return 2 * width + 5, width + 1


def _double_ports(
    target: tuple[int, ...], workspace: tuple[int, ...]
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    n = len(target)
    q, _ = modular_double_requirements(n)
    if len(workspace) != q:
        raise ValueError(f"Modular doubling requires {q} clean work wires")
    _validate_wires((*target, *workspace))
    return workspace[0], workspace[1], workspace[2 : n + 4], workspace[n + 4 :]


def _double_layout(
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    bits: tuple[int, ...],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    ports = _double_ports(target, workspace)
    n = len(target)
    if type(modulus) is not int or not 0 < modulus < 1 << n or modulus % 2 != 1:
        raise ValueError(
            "Modular doubling modulus must be a fitting positive odd integer"
        )
    if (
        len(bits) != n + 1
        or len(set(bits)) != len(bits)
        or any(type(b) is not int or b < 0 for b in bits)
    ):
        raise ValueError(
            f"Modular doubling requires {n+1} distinct nonnegative classical slots"
        )
    return ports


def modular_double_outputs(
    target: tuple[int, ...], workspace: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return source result aliases and clean work; old lower becomes data."""
    lower, sign, constant, carry = _double_ports(target, workspace)
    return (*target[1:], lower), (target[0], sign, *constant, *carry)


def modular_double_ir(
    target: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Double a valid residue modulo odd p, with explicit returned port roles.

    Read the output/work tuples from modular_double_outputs. Work initially
    zero is returned zero on those new roles; no physical SWAP is appended.
    """
    lower, sign, constant, carry = _double_layout(
        target, modulus, workspace, classical_bits
    )
    n = len(target)
    return (
        *add_constant_ir(
            (sign, *target, lower), -modulus, constant, carry, classical_bits
        ),
        *controlled_add_constant_ir(
            sign,
            (*target, lower),
            modulus,
            constant[: n + 1],
            carry[:n],
            classical_bits[:n],
        ),
        Instruction("x", (lower,)),
        Instruction("cx", (lower, sign)),
        Instruction("x", (lower,)),
    )
