# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 KaliskiModInverse.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: explicit native history/work ownership and physical output roles.

"""In-place Montgomery-scaled modular inverse with retained reversible history."""

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from .inversion.inverse_algorithm import inverse_algorithm_ir, inverse_algorithm_outputs


def modular_inverse_outputs(
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return actual target/clean-work roles; history roles never change.

    Forward target becomes the body's coefficient output. Original target
    becomes the clean auxiliary coefficient word. The inverse consumes these
    physical roles; it does not presume a materialized fixed-qubit output.
    """
    n = len(target)
    if n < 2 or len(workspace) != 5 * n + 8 or type(adjoint) is not bool:
        raise ValueError("Modular inverse requires n>=2, 5n+8 work and boolean adjoint")
    _validate_wires((*target, *workspace))
    u, r, s = workspace[:n], workspace[n : 2 * n], workspace[2 * n : 3 * n]
    f, pool = workspace[3 * n], workspace[3 * n + 1 :]
    coefficient, pool = inverse_algorithm_outputs(
        target if adjoint else r, pool, adjoint=adjoint
    )
    return (
        (r, (*u, *coefficient, *s, f, *pool))
        if adjoint
        else (coefficient, (*u, *target, *s, f, *pool))
    )


def modular_inverse_ir(
    target: tuple[int, ...],
    junk: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Port the original public compute/uncompute wrapper with zero work promises.

    Forward target is zero or a unit modulo positive odd p with p+1<2**n.
    All 4n history bits and 5n+8 work qubits start zero. The result is the
    Montgomery-scaled inverse x^-1*2**(2n) mod p, with zero mapped to zero;
    junk retains 4n logical history bits. Work returns zero on the output roles.
    The inverse consumes the matching result/history and actual output roles,
    clears junk/work and restores x. Supply max(2n-1,n+1) classical slots.
    """
    n = len(target)
    modular_inverse_outputs(target, workspace, adjoint=adjoint)
    if len(junk) != 4 * n:
        raise ValueError("Modular inverse requires 4n logical history wires")
    _validate_wires((*target, *junk, *workspace))
    u, r, s = workspace[:n], workspace[n : 2 * n], workspace[2 * n : 3 * n]
    f, pool = workspace[3 * n], workspace[3 * n + 1 :]
    return inverse_algorithm_ir(
        u,
        r if adjoint else target,
        target if adjoint else r,
        s,
        junk[: 2 * n],
        f,
        junk[2 * n :],
        modulus,
        pool,
        classical_bits,
        adjoint=adjoint,
    )
