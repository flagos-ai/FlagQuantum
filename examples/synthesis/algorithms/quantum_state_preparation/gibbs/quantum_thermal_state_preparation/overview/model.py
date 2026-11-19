"""Regenerate the source Fourier windows, Glauber angles, and Gibbs reference."""

import math

import torch

BETA = 0.2
DELTA = 0.1
ROUNDS = 50
SPECTRUM_BITS = 6
MIXING_BITS = 4
FOURIER_SHOTS = 10000
THERMAL_SHOTS = 1000
ENERGIES = (2, 0, -2, 0)
DTYPE = torch.complex128


def frequency(word: int, bits: int) -> int:
    """Interpret the signed Fourier word after extracting its unsigned bits."""
    half = 1 << (bits - 1)
    return word if word < half else word - 2 * half


def fourier_spacing(bits: int) -> float:
    """Use the source relative energy bound four and its 2.5 safety factor."""
    return 10.0 / (1 << bits)


def evolution_angles(bits: int) -> tuple[float, ...]:
    """Return the negative evolution angles on the powers-of-two controls."""
    step = 2 * math.pi / (fourier_spacing(bits) * (1 << bits))
    return tuple(-step * (1 << bit) for bit in range(bits))


def gaussian(bits: int) -> tuple[float, ...]:
    """Generate the entire finite Gaussian amplitude window."""
    count = 1 << bits
    values = tuple(math.exp(-((-2 + 4 * i / (count - 1)) ** 2)) for i in range(count))
    norm = math.sqrt(math.fsum(value**2 for value in values))
    return tuple(value / norm for value in values)


def glauber_angles(bits: int = MIXING_BITS) -> tuple[float, ...]:
    """Convert each signed source Bohr frequency through the Glauber rule."""
    spacing = BETA * fourier_spacing(bits)
    return tuple(
        2 * math.asin(math.sqrt(1 / (1 + math.exp(spacing * frequency(word, bits)))))
        for word in range(1 << bits)
    )


def gibbs_probabilities() -> tuple[float, ...]:
    """Return the independently normalized theoretical probability vector."""
    weights = tuple(math.exp(-BETA * value) for value in ENERGIES)
    return tuple(value / math.fsum(weights) for value in weights)
