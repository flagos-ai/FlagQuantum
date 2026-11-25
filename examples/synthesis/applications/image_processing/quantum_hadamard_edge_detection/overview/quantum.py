"""Build real-amplitude loading and the two controlled cyclic differences."""

import math
from typing import Annotated

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import H, Output, QArray, QBit, QNum, X, allocate, control, qfunc

AXIS_BITS = 7
PIXEL_BITS = 2 * AXIS_BITS


def walsh_angles(angles: torch.Tensor) -> torch.Tensor:
    """Compute the normalized Walsh coefficients in O(n log n) time."""
    length = angles.numel()
    if angles.ndim != 1 or length == 0 or length & (length - 1):
        raise ValueError("Walsh angles must have nonzero power-of-two length")
    output = angles
    stride = 1
    while stride < length:
        blocks = output.reshape(-1, 2, stride)
        left, right = blocks[:, 0], blocks[:, 1]
        output = torch.stack((left + right, left - right), dim=1).reshape(-1)
        stride *= 2
    return output / length


def prepare_image(amplitudes: torch.Tensor) -> fq.Circuit:
    """Prepare all 16384 real nonnegative amplitudes from the zero state."""
    if amplitudes.shape != (1 << PIXEL_BITS,) or amplitudes.dtype != torch.float64:
        raise ValueError("Expected 16384 float64 amplitudes")
    if not bool(torch.isfinite(amplitudes).all()) or bool((amplitudes < 0).any()):
        raise ValueError("Amplitudes must be finite and nonnegative")
    if not math.isclose(float(amplitudes.norm()), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Amplitudes must carry unit norm")
    circuit = fq.Circuit(PIXEL_BITS, dtype=torch.complex128)
    for wire in range(PIXEL_BITS):
        branches = amplitudes.reshape(1 << wire, 2, -1).norm(dim=2)
        angles = 2 * torch.atan2(branches[:, 1], branches[:, 0])
        coefficients = walsh_angles(angles).tolist()
        previous = 0
        for step in range(1 << wire):
            gray = step ^ (step >> 1)
            if step:
                changed = (previous ^ gray).bit_length() - 1
                circuit.cx(wire - 1 - changed, wire)
            if coefficients[gray] != 0:
                circuit.ry(wire, coefficients[gray])
            previous = gray
        if wire:
            circuit.cx(wire - previous.bit_length(), wire)
    return circuit


@qfunc
def decrement(position: QNum) -> None:
    position -= 1


@qfunc
def detect_edges(
    x: Output[Annotated[QNum, 7]],
    y: Output[Annotated[QNum, 7]],
    flags: Output[QArray[QBit]],
) -> None:
    allocate(x)
    allocate(y)
    allocate(3, flags)
    H(flags[0])
    control(flags[0], lambda: decrement(x))
    H(flags[0])
    H(flags[1])
    control(flags[1], lambda: decrement(y))
    H(flags[1])
    control(flags[0], lambda: X(flags[2]))
    control(flags[1], lambda: X(flags[2]))
    control(flags[0:2], lambda: X(flags[2]))


def build_edges() -> synth.Program:
    return synth.synthesize(
        detect_edges,
        max_qubits=20,
        max_workspace=3,
        max_operations=3000,
        memory_limit_bytes=64 * 1024**2,
    )
