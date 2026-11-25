"""Exact compact execution of the two source Strang branch evolutions."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.resource_estimation import ResourceEstimate

from .model import Hydrogen

STEP_CEILING = 0.5
DTYPE = torch.complex128


@dataclass(frozen=True)
class Experiment:
    time: float
    repetitions: int
    template: fq.Circuit
    resources: ResourceEstimate


def _powered_branch(model: Hydrogen, time: float, repetitions: int) -> torch.Tensor:
    dt = time / repetitions
    half_z = torch.diag(
        torch.tensor(
            [
                complex(math.cos(dt * model.z / 2), -math.sin(dt * model.z / 2)),
                complex(math.cos(dt * model.z / 2), math.sin(dt * model.z / 2)),
            ],
            dtype=DTYPE,
        )
    )
    cosine, sine = math.cos(dt * model.x), math.sin(dt * model.x)
    x_step = torch.tensor([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=DTYPE)
    scalar = complex(math.cos(dt * model.scalar), -math.sin(dt * model.scalar))
    powered: torch.Tensor = torch.linalg.matrix_power(
        scalar * half_z @ x_step @ half_z, repetitions
    )
    return powered


def build(models: tuple[Hydrogen, Hydrogen], time: float) -> Experiment:
    if not math.isfinite(time) or time <= 0:
        raise ValueError("Evolution time must be positive and finite")
    repetitions = math.ceil(time / STEP_CEILING)
    matrix = torch.zeros((4, 4), dtype=DTYPE)
    for branch, row in enumerate(models):
        start = 2 * branch
        matrix[start : start + 2, start : start + 2] = _powered_branch(
            row, time, repetitions
        )
    circuit = (
        fq.Circuit(2, dtype=DTYPE)
        .x(1)
        .h(0)
        .any(0, 1, matrix=matrix)
        .phase(0, fq.Parameter("delta_phase"))
        .h(0)
    )
    return Experiment(time, repetitions, circuit, estimate_resources(circuit))


def execute(experiment: Experiment, delta: float) -> torch.Tensor:
    if not math.isfinite(delta):
        raise ValueError("Trial energy offset must be finite")
    circuit = experiment.template.bind_parameters(
        {"delta_phase": delta * experiment.time}
    )
    state: torch.Tensor = fq.run(circuit).statevector().flatten()
    return state


def reference(
    models: tuple[Hydrogen, Hydrogen], time: float, delta: float
) -> torch.Tensor:
    """Apply every Strang factor to each vector using independent dense exponentials."""
    z = torch.diag(torch.tensor([1, -1], dtype=DTYPE))
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    count = math.ceil(time / STEP_CEILING)
    dt = time / count
    branches = []
    for model in models:
        identity_phase = complex(
            math.cos(-dt * model.scalar), math.sin(-dt * model.scalar)
        )
        left = torch.linalg.matrix_exp(-0.5j * dt * model.z * z)
        centre = torch.linalg.matrix_exp(-1j * dt * model.x * x)
        vector = torch.tensor([0, 1], dtype=DTYPE)
        for _ in range(count):
            vector = identity_phase * (left @ (centre @ (left @ vector)))
        branches.append(vector)
    phase = complex(math.cos(delta * time), math.sin(delta * time))
    expected: torch.Tensor = (
        torch.cat(
            (branches[0] + phase * branches[1], branches[0] - phase * branches[1])
        )
        / 2
    )
    return expected
