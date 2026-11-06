# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _KaliskiIteration.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py
# Changes: compose shared native layers with explicit clean work and output roles.

"""One original binary modular inverse iteration with exact reverse composition."""

from ...core.ir import Instruction
from .bitwise import _validate_wires
from .inverse_arithmetic import inverse_arithmetic_ir
from .inverse_exchange import inverse_exchange_ir
from .inverse_ordering import inverse_ordering_ir
from .inverse_parity import inverse_parity_ir
from .inverse_scaling import inverse_scaling_ir, inverse_scaling_outputs
from .inverse_termination import inverse_termination_ir


def inverse_iteration_outputs(
    coefficient: tuple[int, ...],
    workspace: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the changed coefficient and clean-work roles without extra gates."""
    if len(coefficient) < 2 or len(workspace) != 2 * len(coefficient) + 7:
        raise ValueError("Inverse iteration requires n>=2 and 2n+7 work wires")
    result, work = inverse_scaling_outputs(coefficient, workspace[2:], adjoint=adjoint)
    return result, (*workspace[:2], *work)


def inverse_iteration_ir(
    left: tuple[int, ...],
    right: tuple[int, ...],
    left_coefficient: tuple[int, ...],
    right_coefficient: tuple[int, ...],
    marker: int,
    active: int,
    terminal: int,
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Compose the original six layers, retaining their measured exact adjoints.

    Words are equal-width big-endian; n>=2 and modulus is a fitting positive
    odd integer. Supply 2n+7 clean work and max(2n-1,n+1) reusable classical
    slots. Marker and terminal are fresh zero logical outputs on forward
    algorithm-reachable inputs. Internal parity/branch qubits clear on those
    inputs; arbitrary words/flags need not satisfy this promise. The inverse
    consumes the full forward image and inverse_iteration_outputs' roles.
    There are 8n+5 measurement events; none forcibly resets logical work.
    """
    n = len(left)
    if (
        n < 2
        or any(len(word) != n for word in (right, left_coefficient, right_coefficient))
        or type(adjoint) is not bool
    ):
        raise ValueError(
            "Inverse iteration requires four equal n>=2 words and boolean adjoint"
        )
    _validate_wires(
        (
            *left,
            *right,
            *left_coefficient,
            *right_coefficient,
            marker,
            active,
            terminal,
            *workspace,
        )
    )
    coefficient, returned_work = inverse_iteration_outputs(
        left_coefficient, workspace, adjoint=adjoint
    )
    if (
        len(classical_bits) != max(2 * n - 1, n + 1)
        or len(set(classical_bits)) != len(classical_bits)
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            "Inverse iteration requires max(2n-1,n+1) distinct classical slots"
        )
    parity, branch = workspace[:2]
    work = returned_work[2:] if adjoint else workspace[2:]
    r = coefficient if adjoint else left_coefficient
    stages = (
        inverse_termination_ir(
            right,
            marker,
            active,
            terminal,
            work[:n],
            classical_bits[:n],
            adjoint=adjoint,
        ),
        inverse_parity_ir(
            left,
            right,
            branch,
            parity,
            marker,
            active,
            work[:2],
            classical_bits[:2],
            adjoint=adjoint,
        ),
        inverse_ordering_ir(
            left,
            right,
            branch,
            parity,
            marker,
            active,
            work[: n + 4],
            classical_bits[: max(n, 2)],
            adjoint=adjoint,
        ),
        inverse_exchange_ir(parity, left, right, r, right_coefficient, adjoint=adjoint),
        inverse_arithmetic_ir(
            left,
            right,
            r,
            right_coefficient,
            branch,
            active,
            work[: 2 * n],
            classical_bits[: 2 * n - 1],
            adjoint=adjoint,
        ),
        inverse_scaling_ir(
            left,
            right,
            left_coefficient,
            right_coefficient,
            branch,
            parity,
            marker,
            active,
            modulus,
            workspace[2:],
            classical_bits[: n + 1],
            adjoint=adjoint,
        ),
    )
    return tuple(
        gate for stage in (reversed(stages) if adjoint else stages) for gate in stage
    )
