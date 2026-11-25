"""Regenerate the hydrogen energy curve and the exact two-determinant sector."""

from dataclasses import dataclass

import torch

from examples.synthesis.applications.chemistry.molecular_energy_curve.overview.model import (
    build_model,
)

CENTRE = 1.0
DISPLACEMENT = 0.0025
SAMPLE_COUNT = 30


@dataclass(frozen=True)
class Hydrogen:
    distance: float
    scalar: float
    z: float
    x: float
    fci: float


def hydrogen(distance: float) -> Hydrogen:
    generated = build_model(distance)
    block = generated.hamiltonian
    if float(block.imag.abs().max()) >= 1e-12:
        raise AssertionError("Expected the real invariant hydrogen sector")
    if abs(float(torch.linalg.eigvalsh(block)[0]) - generated.fci) >= 1e-11:
        raise AssertionError("The paired-sector and full-space energies differ")
    first, second = float(block[0, 0].real), float(block[1, 1].real)
    return Hydrogen(
        distance,
        (first + second) / 2,
        (first - second) / 2,
        float(block[0, 1].real),
        generated.fci,
    )


def curve() -> tuple[Hydrogen, ...]:
    grid = torch.linspace(0.4, 2.1, SAMPLE_COUNT, dtype=torch.float64)
    return tuple(hydrogen(float(distance)) for distance in grid)


def differences(
    rows: tuple[Hydrogen, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    intervals = tuple(zip(rows[:-1], rows[1:], strict=True))
    return (
        tuple((left.distance + right.distance) / 2 for left, right in intervals),
        tuple(
            (right.fci - left.fci) / (right.distance - left.distance)
            for left, right in intervals
        ),
    )


def displaced() -> tuple[Hydrogen, Hydrogen]:
    return (
        hydrogen(CENTRE - DISPLACEMENT / 2),
        hydrogen(CENTRE + DISPLACEMENT / 2),
    )
