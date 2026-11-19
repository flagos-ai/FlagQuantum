"""Independent dense sine/cosine, spectral evolution and inverse Fourier oracle."""

import math

import torch

import flagquantum as fq


def reference_circuit(amplitudes: tuple[float, ...]) -> fq.Circuit:
    """Keep the original 64 source amplitudes, six estimator bits and indicator."""
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    identity = torch.eye(64, dtype=torch.complex128)
    delta = identity[:, 0] - vector
    preparation = identity - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    samples = torch.arange(8, dtype=torch.float64) + 0.5
    kernel = torch.exp(
        1j * math.pi * torch.outer(torch.arange(9, dtype=torch.float64), samples) / 8
    )
    sine, cosine = kernel[1:].imag.clone() / 2, kernel[:-1].real.clone() / 2
    sine[-1] /= math.sqrt(2)
    cosine[0] /= math.sqrt(2)
    energies = torch.tensor(
        [
            4 * math.sin(math.pi * (int(f"{word:06b}"[::-1], 2) % 8 + 1) / 16) ** 2
            + 4 * math.sin(math.pi * (int(f"{word:06b}"[::-1], 2) // 8) / 16) ** 2
            for word in range(64)
        ],
        dtype=torch.float64,
    )
    words = torch.arange(64, dtype=torch.float64)
    fourier = torch.exp(2j * math.pi * torch.outer(words, words) / 64) / 8
    circuit = fq.Circuit(13, dtype=torch.complex128).any(*range(6), matrix=preparation)
    circuit.any(3, 4, 5, matrix=sine.to(torch.complex128)).any(
        0, 1, 2, matrix=cosine.to(torch.complex128)
    )
    for wire in range(6, 12):
        circuit.h(wire)
    for bit in range(6):
        controlled = torch.eye(128, dtype=torch.complex128)
        controlled[64:, 64:] = torch.diag(
            torch.exp(2j * (1 << bit) * energies / math.pi)
        )
        circuit.any(11 - bit, *range(6), matrix=controlled)
    circuit.any(*range(6, 12), matrix=fourier.mH)
    reciprocal = torch.zeros((128, 128), dtype=torch.complex128)
    for word in range(64):
        value = 0.0 if word == 0 else 1 / (64 * word)
        cosine_angle = math.sqrt(1 - value**2)
        reciprocal[2 * word : 2 * word + 2, 2 * word : 2 * word + 2] = torch.tensor(
            [[cosine_angle, -value], [value, cosine_angle]], dtype=torch.complex128
        )
    circuit.any(*range(6, 13), matrix=reciprocal).any(*range(6, 12), matrix=fourier)
    for bit in reversed(range(6)):
        controlled = torch.eye(128, dtype=torch.complex128)
        controlled[64:, 64:] = torch.diag(
            torch.exp(-2j * (1 << bit) * energies / math.pi)
        )
        circuit.any(11 - bit, *range(6), matrix=controlled)
    for wire in range(6, 12):
        circuit.h(wire)
    return circuit.any(0, 1, 2, matrix=cosine.to(torch.complex128).mH).any(
        3, 4, 5, matrix=sine.to(torch.complex128).mH
    )
