"""Compose the sixteen-row source block from existing native operations."""

import math
from typing import TypedDict

import torch

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    control,
    inplace_prepare_amplitudes,
    phase,
    prepare_amplitudes,
    qfunc,
    select_rotation,
    within,
)


@qfunc
def band_term(
    angles: list[float], offsets: list[int], term: int, flag: QBit, data: QNum[4]
) -> None:
    # RY(2*acos(d)) Z dilates each signed diagonal entry before its cyclic shift.
    phase(flag, math.pi)
    select_rotation(
        "Y",
        [
            angles[16 * term],
            angles[16 * term + 1],
            angles[16 * term + 2],
            angles[16 * term + 3],
            angles[16 * term + 4],
            angles[16 * term + 5],
            angles[16 * term + 6],
            angles[16 * term + 7],
            angles[16 * term + 8],
            angles[16 * term + 9],
            angles[16 * term + 10],
            angles[16 * term + 11],
            angles[16 * term + 12],
            angles[16 * term + 13],
            angles[16 * term + 14],
            angles[16 * term + 15],
        ],
        data,
        flag,
    )
    data -= offsets[term]


@qfunc
def banded_signal(
    weights: list[float],
    angles: list[float],
    offsets: list[int],
    count: int,
    selection: QNum[3],
    flag: QBit,
    data: QNum[4],
) -> None:
    with within():
        inplace_prepare_amplitudes(weights, 0, selection)
    with apply():
        for term in range(count):
            control(
                selection == term,
                lambda: band_term(angles, offsets, term, flag, data),  # noqa: B023
            )


@qfunc
def banded_block(
    amplitudes: list[float],
    weights: list[float],
    angles: list[float],
    offsets: list[int],
    count: int,
    selection: Output[QNum[3]],
    flag: Output[QBit],
    data: Output[QNum[4]],
) -> None:
    allocate(selection)
    allocate(flag)
    prepare_amplitudes(amplitudes, 0, data)
    banded_signal(weights, angles, offsets, count, selection, flag, data)


class BandInputs(TypedDict):
    amplitudes: tuple[float, ...]
    weights: list[float]
    angles: list[float]
    offsets: list[int]
    count: int


def banded_inputs(matrix: torch.Tensor, rhs: torch.Tensor) -> tuple[BandInputs, float]:
    """Derive finite classical tables for the sixteen-row native source block."""
    if (
        matrix.shape != (16, 16)
        or (matrix.is_complex() and bool(matrix.imag.any()))
        or not bool(torch.isfinite(matrix).all())
    ):
        raise ValueError(
            "the native banded profile requires a finite real 16x16 matrix"
        )
    offsets = [
        offset for offset in range(-15, 16) if bool(matrix.diagonal(offset).any())
    ]
    if not 1 <= len(offsets) <= 8:
        raise ValueError(
            "the three-bit selector requires one through eight nonzero bands"
        )
    widths = [float(matrix.diagonal(offset).abs().max()) for offset in offsets]
    alpha = math.fsum(widths)
    angles = []
    for offset, scale in zip(offsets, widths, strict=True):
        for column in range(16):
            row = column - offset
            diagonal = float(matrix[row, column].real / scale) if 0 <= row < 16 else 0.0
            angles.append(2 * math.acos(diagonal))
    inputs: BandInputs = {
        "amplitudes": tuple(float(value) for value in rhs),
        "weights": [(width / alpha) ** 0.5 for width in widths]
        + [0.0] * (8 - len(widths)),
        "angles": angles,
        "offsets": offsets,
        "count": len(offsets),
    }
    return inputs, alpha


def banded_program(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    max_qubits: int = 11,
    max_workspace: int = 3,
    max_operations: int = 1024,
) -> tuple[synth.Program, float]:
    """Build the source block; full native QSVT exceeds the current effect budget."""
    inputs, alpha = banded_inputs(matrix, rhs)
    program = synth.synthesize(
        banded_block,
        classical_inputs=inputs,
        max_qubits=max_qubits,
        max_workspace=max_workspace,
        max_operations=max_operations,
        memory_limit_bytes=64 * 1024**2,
    )
    return program, alpha
