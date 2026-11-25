"""Run the original logical-cat demonstration and all105 measured parameter points."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler import estimate_resources

from . import channels, quantum
from .noise import Parameters

SHOTS = 10_000
DISTANCES = tuple(range(3, 16, 2))
PHOTONS = (4, 7, 10, 15, 19)
KAPPA_2 = (1e5, 1e6, 1e7)


@dataclass(frozen=True)
class Observation:
    parameters: Parameters | None
    probability_one: float
    counts: tuple[int, int]
    clocks: int
    physical_instructions: int
    noise_applications: int
    merged_channels: int

    @property
    def observed_error(self) -> float:
        return self.counts[1] / SHOTS


@dataclass(frozen=True)
class Rotation:
    iterations: int
    gates: int
    depth: int
    operator_error: float


@dataclass(frozen=True)
class Result:
    rotation_examples: tuple[Rotation, ...]
    noiseless: Observation
    configured: Observation
    grid: tuple[Observation, ...]
    full_state_preparation_error: float
    maximum_rotation_error: float


def sample(probability: float, seed: int) -> tuple[int, int]:
    if not 0 <= probability <= 1:
        raise ValueError("The executed measurement probability must be physical")
    observations = torch.multinomial(
        torch.tensor([1 - probability, probability], dtype=torch.float64),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(observations, minlength=2)
    return int(counts[0]), int(counts[1])


def observe(circuit: fq.Circuit, parameters: Parameters, seed: int) -> Observation:
    execution = channels.execute(circuit, parameters)
    probability = float(execution.density.diagonal().real[64:].sum())
    return Observation(
        parameters,
        probability,
        sample(probability, seed),
        execution.clocks,
        execution.source_instructions,
        execution.one_qubit_noise_applications,
        execution.merged_channels,
    )


def run() -> Result:
    single = fq.Circuit(1, dtype=torch.complex128).rx(0, theta=0.1)
    rotations = []
    for iterations in (2, 3):
        program, errors = quantum.synthesize(single, iterations)
        resources = estimate_resources(program)
        rotations.append(
            Rotation(iterations, resources.n_operations, resources.depth, max(errors))
        )
    exact = quantum.swap_test()
    state = exact.state().reshape(-1)
    values = torch.tensor(quantum.amplitudes(), dtype=torch.complex128)
    expected = torch.kron(
        torch.tensor([1, 0], dtype=torch.complex128), torch.kron(values, values)
    )
    error = float((state - expected).abs().max())
    if error > 1e-12:
        raise ValueError("The source's complete identical-state preparation changed")
    probability = float(state[64:].abs().square().sum())
    noiseless = Observation(
        None,
        probability,
        sample(probability, 12),
        0,
        len(exact.to_ir().instructions),
        0,
        0,
    )
    compiled, errors = quantum.synthesize(exact, 2)
    configured = observe(compiled, Parameters(31, 19, 1e4), 31)
    grid: list[Observation] = []
    for photons in PHOTONS:
        for distance in DISTANCES:
            for kappa in KAPPA_2:
                point = observe(
                    compiled, Parameters(distance, photons, kappa), 10_000 + len(grid)
                )
                grid.append(point)
                print(f"Executed original cat grid point {len(grid)}/105", flush=True)
    return Result(
        tuple(rotations), noiseless, configured, tuple(grid), error, max(errors)
    )
