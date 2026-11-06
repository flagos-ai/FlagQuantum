# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ApproximateQFT.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/approximate_qft.py

"""Fourier transform with precision-limited addition into a phase gradient."""

from math import ceil, isfinite, log2, pi, sqrt

from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.phase_addition import phase_add_ir, phase_add_workspace_size
from ...core.ir import Instruction


def approximate_qft_precision(width: int, epsilon: float) -> int:
    """Compute the pinned concrete upper bound for gradient precision."""
    approximate_qft_workspace_size(width, 1)
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Approximate QFT epsilon must be finite and positive")
    return ceil(log2(width * pi * sqrt(2) / epsilon))


def approximate_qft_workspace_size(width: int, precision: int) -> int:
    """Return clean work and equally many reusable classical slots."""
    if (
        type(width) is not int
        or type(precision) is not int
        or width < 1
        or precision < int(width > 1)
    ):
        raise ValueError("Approximate QFT requires valid data and gradient widths")
    active = max(0, min(width - 1, precision - 1))
    return (
        phase_add_workspace_size(active, active + 1, right_shift=1, controlled=True)
        if active
        else 0
    )


def approximate_qft_ir(
    data: tuple[int, ...],
    gradient: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    with_reverse: bool = True,
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Use the caller's gradient precision and omit smaller controlled phases."""
    required = approximate_qft_workspace_size(len(data), len(gradient))
    if len(workspace) != required:
        raise ValueError(f"Approximate QFT requires {required} clean work qubits")
    if (
        len(classical_bits) != required
        or len(set(classical_bits)) != required
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError("Approximate QFT requires distinct nonnegative slots")
    if type(with_reverse) is not bool or type(adjoint) is not bool:
        raise ValueError("Approximate QFT direction and reversal must be boolean")
    _validate_wires((*data, *gradient, *workspace))

    steps = []
    for i, control in enumerate(data):
        active = max(0, min(i, len(gradient) - 1))
        h = Instruction("h", (control,))
        if active:
            work = phase_add_workspace_size(
                active, active + 1, right_shift=1, controlled=True
            )
            addition = phase_add_ir(
                data[i - active : i][::-1],
                gradient[: active + 1],
                workspace[:work],
                classical_bits[:work],
                right_shift=1,
                sign=-1 if adjoint else 1,
                control=control,
            )
        else:
            addition = ()
        steps.append((h, *addition) if adjoint else (*addition, h))
    swaps = (
        tuple(
            Instruction("swap", (data[i], data[-1 - i])) for i in range(len(data) // 2)
        )
        if with_reverse
        else ()
    )
    if adjoint:
        return (*reversed(swaps), *(g for step in reversed(steps) for g in step))
    return (*(g for step in steps for g in step), *swaps)
