"""Optimize the native tapered ansatz for every distance in the source."""

from dataclasses import dataclass
from time import monotonic

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.molecular_energy_curve.main import (
    build_program,
)
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.compiler import ResourceEstimate

from .model import DTYPE, Model, build_model, distances

ITERATION_LIMIT = 500


@dataclass(frozen=True)
class Point:
    distance: float
    model: Model
    energy: float
    angle: float
    evaluations: int
    iterations: int
    converged: bool
    resources: ResourceEstimate
    seconds: float


def evaluate(model: Model, angle: float) -> float:
    """Evaluate the exact expectation of an actually executed native circuit."""
    program = build_program((angle,))
    amplitudes = fq.run(program).statevector().flatten()
    return float(torch.vdot(amplitudes, model.hamiltonian @ amplitudes).real)


def optimize(distance: float) -> Point:
    start = monotonic()
    model = build_model(distance)

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        return torch.tensor(evaluate(model, float(parameters[0])), dtype=DTYPE)

    optimizer = CobylaOptimizer(maxiter=ITERATION_LIMIT, rhobeg=1.0, rhoend=1e-8)
    outcome = optimizer.minimize(objective, torch.zeros(1, dtype=DTYPE))
    angle = float(outcome.parameters[0])
    program = build_program((angle,))
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Point(
        distance,
        model,
        outcome.value,
        angle,
        outcome.evaluations,
        outcome.iterations,
        outcome.converged,
        resources,
        monotonic() - start,
    )


def run_workflow() -> tuple[Point, ...]:
    return tuple(optimize(distance) for distance in distances())
