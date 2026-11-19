"""Run all source imaginary-time updates with actual finite samples."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .model import CutModel, RealArray, source_model
from .quantum import ProgramBuilder, probabilities, program_builder

SHOTS = 128
ROUNDS = 14


@dataclass(frozen=True)
class Round:
    """Both independent energy estimates and the signed rounded-partition cost."""

    iteration: int
    energy: float
    measured_energy: float
    partition_energy: float
    partition_word: int
    angles: tuple[float, ...]


@dataclass(frozen=True)
class Run:
    """The fourteen updates and the actual histogram evaluation budget used."""

    model: CutModel
    rounds: tuple[Round, ...]
    measurements: int


def _update(metric: RealArray, derivative: RealArray) -> RealArray:
    left, singular, right = np.linalg.svd(metric, full_matrices=False)
    retained = (singular >= 0.01 * singular[0]) & (singular > 0)
    weighted = (left[:, retained].T @ derivative) / singular[retained]
    return np.asarray(right[retained].T @ weighted, dtype=np.float64)


def _measure(
    build: ProgramBuilder, values: RealArray, generator: torch.Generator
) -> RealArray:
    program = build(tuple(map(float, values)))
    weights = torch.tensor(probabilities(program), dtype=torch.float64)
    draws = torch.multinomial(weights, SHOTS, replacement=True, generator=generator)
    return np.asarray(
        torch.bincount(draws, minlength=16).numpy() / SHOTS, dtype=np.float64
    )


def run_workflow(seed: int = 0) -> Run:
    """Retain the original two-sided shift, SVD cutoff, and readout sequence."""
    model = source_model()
    build = program_builder(model)
    device_rng = torch.Generator().manual_seed(seed)
    angles = np.zeros(6, dtype=np.float64)
    previous_energy = 0.0
    metric = np.empty((7, 6), dtype=np.float64)
    shifts = np.eye(6, dtype=np.float64) * (math.pi / 4)
    records = []
    measurements = 0
    for iteration in range(1, ROUNDS + 1):
        for column, shift in enumerate(shifts):
            positive = _measure(build, angles + shift, device_rng)
            negative = _measure(build, angles - shift, device_rng)
            metric[:, column] = (positive - negative) @ model.parities / 4
            measurements += 2
        current = _measure(build, angles, device_rng)
        derivative = (current * (previous_energy - model.energy)) @ model.parities
        previous_energy = float(current @ model.energy)
        rounded = _measure(build, angles, device_rng)
        measurements += 2
        word = sum(
            (float(value) <= 0) << bit
            for bit, value in enumerate(rounded @ model.spins)
        )
        angles += (0.1 / math.sqrt(iteration)) * _update(metric, derivative)
        records.append(
            Round(
                iteration,
                previous_energy,
                float(rounded @ model.energy),
                float(model.energy[word]),
                word,
                tuple(map(float, angles)),
            )
        )
    return Run(model, tuple(records), measurements)
