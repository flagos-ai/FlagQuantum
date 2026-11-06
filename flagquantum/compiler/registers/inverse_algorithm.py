# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiModInverseImpl.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: native shared layers, explicit history/work and physical output roles.

"""Full initialized binary modular inverse algorithm with retained history."""

from ...core.ir import Instruction
from .arithmetic import add_constant_adjoint_ir, add_constant_ir
from .bitwise import (
    _validate_wires,
    bitwise_not_ir,
    controlled_xor_constant_ir,
    xor_constant_ir,
)
from .inverse_iteration import inverse_iteration_ir, inverse_iteration_outputs


def inverse_algorithm_outputs(
    coefficient: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve the 2n original iterations' coefficient/work physical roles."""
    for _ in range(2 * len(coefficient)):
        coefficient, workspace = inverse_iteration_outputs(
            coefficient, workspace, adjoint=adjoint
        )
    if not coefficient:
        raise ValueError("Inverse algorithm requires at least two coefficient wires")
    return coefficient, workspace


def inverse_algorithm_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    left_coefficient: tuple[int, ...],
    right_coefficient: tuple[int, ...],
    markers: tuple[int, ...],
    active: int,
    terminals: tuple[int, ...],
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Emit original initialization, 2n rounds, coefficient conversion and cleanup.

    Equal n>=2 big-endian words; marker/terminal histories each have 2n bits.
    The forward input right contains x=0 or a unit modulo positive odd modulus;
    all other logical inputs and 2n+7 work qubits start zero. Output coefficient
    is x^-1 * 2**(2n) mod p (zero maps to zero); histories remain logical data.
    The other words and active flag clear on this domain. Use the returned
    roles from inverse_algorithm_outputs, including for the inverse input.
    Supply max(2n-1,n+1) reusable classical slots. The source AddK(p+1) requires
    p+1<2**n; all-ones moduli retain the source's decomposition failure.
    """
    n = len(left)
    if (
        n < 2
        or any(len(word) != n for word in (right, left_coefficient, right_coefficient))
        or len(markers) != 2 * n
        or len(terminals) != 2 * n
        or type(adjoint) is not bool
    ):
        raise ValueError(
            "Inverse algorithm requires equal n>=2 words, 2n histories and boolean adjoint"
        )
    _validate_wires(
        (
            *left,
            *right,
            *left_coefficient,
            *right_coefficient,
            *markers,
            active,
            *terminals,
            *workspace,
        )
    )
    inverse_algorithm_outputs(left_coefficient, workspace, adjoint=adjoint)
    if type(modulus) is not int or not 0 < modulus < (1 << n) - 1 or modulus % 2 != 1:
        raise ValueError(
            "Inverse algorithm requires positive odd modulus with p+1 fitting n bits"
        )
    if (
        len(classical_bits) != max(2 * n - 1, n + 1)
        or len(set(classical_bits)) != len(classical_bits)
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            "Inverse algorithm requires max(2n-1,n+1) distinct classical slots"
        )
    r, work = left_coefficient, workspace
    initialize = (
        (Instruction("x", (active,)),),
        xor_constant_ir(left, modulus),
        xor_constant_ir(right_coefficient, 1),
    )

    def finish(
        coefficient: tuple[int, ...], pool: tuple[int, ...]
    ) -> tuple[Instruction, ...]:
        constant, carries = pool[:n], pool[n : 2 * n - 1]
        addition = add_constant_adjoint_ir if adjoint else add_constant_ir
        update = addition(
            coefficient, modulus + 1, constant, carries, classical_bits[: n - 1]
        )
        complement = bitwise_not_ir(coefficient)
        unload = (xor_constant_ir(left, 1), xor_constant_ir(right_coefficient, modulus))
        cleanup = tuple(
            controlled_xor_constant_ir(terminals[0], target, value)
            for target, value in (
                (coefficient, modulus),
                (left, modulus - 1),
                (right_coefficient, modulus - 1),
            )
        )
        unload_gates = tuple(
            g for stage in (reversed(unload) if adjoint else unload) for g in stage
        )
        cleanup_gates = tuple(
            g for stage in (reversed(cleanup) if adjoint else cleanup) for g in stage
        )
        return (
            (*cleanup_gates, *unload_gates, *update, *complement)
            if adjoint
            else (*complement, *update, *unload_gates, *cleanup_gates)
        )

    gates: list[Instruction] = []
    if adjoint:
        gates.extend(finish(r, work))
    else:
        gates.extend(g for stage in initialize for g in stage)
    indices = reversed(range(2 * n)) if adjoint else range(2 * n)
    for i in indices:
        gates.extend(
            inverse_iteration_ir(
                left,
                right,
                r,
                right_coefficient,
                markers[i],
                active,
                terminals[i],
                modulus,
                work,
                classical_bits,
                adjoint=adjoint,
            )
        )
        r, work = inverse_iteration_outputs(r, work, adjoint=adjoint)
    if adjoint:
        gates.extend(g for stage in reversed(initialize) for g in stage)
    else:
        gates.extend(finish(r, work))
    return tuple(gates)
