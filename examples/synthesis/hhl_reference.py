"""Independent dense Circuit reference for matrix HHL examples."""

import math

import torch

import flagquantum as fq


def householder(amplitudes: tuple[float, ...]) -> torch.Tensor:
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    identity = torch.eye(len(amplitudes), dtype=torch.complex128)
    delta = identity[:, 0] - vector
    norm = float(delta.abs().square().sum())
    return (
        identity
        if norm < 1e-28
        else identity - 2 * torch.outer(delta, delta.conj()) / norm
    )


def matrix_hhl_circuit(
    physical: int,
    data: tuple[int, ...],
    phase: tuple[int, ...],
    indicator: int,
    unitary: tuple[tuple[complex, ...], ...],
    angles: tuple[float, ...],
    amplitudes: tuple[float, ...],
) -> fq.Circuit:
    circuit = fq.Circuit(physical, dtype=torch.complex128)
    circuit.any(*data, matrix=householder(amplitudes))
    estimation = fq.Circuit(physical, dtype=torch.complex128)
    dimension = 1 << len(data)
    matrix = torch.tensor(unitary, dtype=torch.complex128)
    powers: list[tuple[int, torch.Tensor]] = []
    for bit, wire in enumerate(reversed(phase)):
        controlled = torch.zeros((2 * dimension, 2 * dimension), dtype=torch.complex128)
        controlled[:dimension, :dimension] = torch.eye(
            dimension, dtype=torch.complex128
        )
        controlled[dimension:, dimension:] = torch.linalg.matrix_power(matrix, 1 << bit)
        estimation.h(wire).any(wire, *data, matrix=controlled)
        powers.append((wire, controlled))
    size = 1 << len(phase)
    axis = torch.arange(size, dtype=torch.float64)
    fourier = torch.exp(-2j * math.pi * torch.outer(axis, axis) / size) / math.sqrt(
        size
    )
    estimation.any(*phase, matrix=fourier)
    circuit.compose(estimation)
    table = torch.zeros((2 * size, 2 * size), dtype=torch.complex128)
    for word, angle in enumerate(angles):
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        table[2 * word : 2 * word + 2, 2 * word : 2 * word + 2] = torch.tensor(
            [[c, -s], [s, c]], dtype=torch.complex128
        )
    circuit.any(*phase, indicator, matrix=table)
    # Preserve rounded source entries. Core adjoint validation has a tighter
    # tolerance than source matrix admission, so form the reference inverse
    # explicitly rather than projecting or renormalizing the matrix.
    circuit.any(*phase, matrix=fourier.mH)
    for wire, controlled in reversed(powers):
        circuit.any(wire, *data, matrix=controlled.mH)
    for wire in phase:
        circuit.h(wire)
    return circuit
