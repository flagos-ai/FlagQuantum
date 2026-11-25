"""Run the original two- and three-oscillator distance sweeps with native VQE."""

from dataclasses import dataclass
from time import monotonic

import numpy as np
import torch

import flagquantum as fq

from .model import RealArray, oscillator_hamiltonian
from .quantum import build_program
from .reference import reference_circuit

ITERATION_BUDGET = 500
LEARNING_RATE = 0.04


@dataclass(frozen=True)
class VariationalRun:
    """Actual objective trace and independently computed finite-basis ground energy."""

    oscillator_count: int
    distance: float
    iterations: int
    evaluations: int
    objective_energies: tuple[float, ...]
    angles: tuple[float, ...]
    energy: float
    reference_energy: float
    seconds: float


def energy_at(
    oscillator_count: int, angles: tuple[float, ...], matrix: RealArray
) -> float:
    """Evaluate the full Hamiltonian expectation from the synthesized state."""
    state = fq.run(build_program(oscillator_count, angles)).statevector()
    amplitudes = state.numpy().flatten()
    return float(np.vdot(amplitudes, matrix @ amplitudes).real)


def optimize_distance(oscillator_count: int, distance: float) -> VariationalRun:
    """Start from source all-ones angles and keep the original iteration budget."""
    started = monotonic()
    matrix = oscillator_hamiltonian(oscillator_count, 4, distance)
    observable = torch.from_numpy(matrix).to(torch.complex128)
    parameters = torch.ones(
        12 if oscillator_count == 2 else 40, dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.Adam([parameters], lr=LEARNING_RATE)
    history: list[float] = []
    for _ in range(ITERATION_BUDGET):
        optimizer.zero_grad()
        state = (
            fq.run(reference_circuit(oscillator_count, parameters))
            .statevector()
            .flatten()
        )
        energy = torch.vdot(state, observable @ state).real
        history.append(float(energy.detach()))
        (parameters.grad,) = torch.autograd.grad(energy, parameters)
        optimizer.step()

    angles = tuple(map(float, parameters.detach()))
    # Read the last trained binding from native synthesis and check the training circuit.
    state = fq.run(build_program(oscillator_count, angles)).statevector().flatten()
    comparison = (
        fq.run(reference_circuit(oscillator_count, angles)).statevector().flatten()
    )
    torch.testing.assert_close(state, comparison, rtol=0, atol=1e-12)
    energy_value = float(torch.vdot(state, observable @ state).real)
    history.append(energy_value)
    return VariationalRun(
        oscillator_count,
        distance,
        ITERATION_BUDGET,
        len(history),
        tuple(history),
        angles,
        energy_value,
        float(np.linalg.eigvalsh(matrix)[0]),
        monotonic() - started,
    )


def sweep(oscillator_count: int) -> tuple[VariationalRun, ...]:
    """Run each of the five original separations for the chosen ansatz."""
    if oscillator_count not in (2, 3):
        raise ValueError("expected two or three oscillators")
    first = 3.3 if oscillator_count == 2 else 3.1
    return tuple(
        optimize_distance(oscillator_count, float(distance))
        for distance in np.linspace(first, 5.5, 5)
    )
