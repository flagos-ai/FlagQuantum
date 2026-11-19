"""Source four-mode fermionic energy and bounded native variational search."""

from dataclasses import dataclass

import flagquantum as fq
from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.main import (
    build_program,
)
from flagquantum.observables import Observable, annihilate, create, jordan_wigner


def hamiltonian() -> Observable:
    """Map the source ordered fermionic monomials with the public Jordan-Wigner API."""
    operator = (
        0.2 * create(0) * annihilate(0)
        - 0.1 * create(0) * create(1) * annihilate(1) * annihilate(0)
        - 0.3 * create(2) * create(3) * annihilate(2) * annihilate(3)
    )
    mapped = jordan_wigner(operator, n_modes=4)
    if not isinstance(mapped, Observable):
        raise TypeError("Jordan-Wigner mapping must return a FlagQuantum Observable.")
    return mapped


@dataclass(frozen=True)
class Optimization:
    """Best actual native energy, all evaluated candidates and explicit termination."""

    parameters: tuple[float, ...]
    initial_energy: float
    best_energy: float
    trace: tuple[float, ...]
    termination: str


def optimize() -> Optimization:
    """Retain twelve zero initial angles and use 200 explicit bounded evaluations."""
    observable = hamiltonian()
    point = [0.0] * 12
    trace: list[float] = []

    def evaluate(parameters: list[float]) -> float:
        program = build_program(parameters)
        result = fq.run(
            program.circuit, outputs=[fq.expectation(observable, name="energy")]
        )
        energy = float(result.expectation("energy").reshape(()))
        trace.append(energy)
        return energy

    value = evaluate(point)
    best_parameters = tuple(point)
    best = value
    step = 0.5
    while len(trace) < 200:
        for coordinate in range(12):
            candidates = []
            for sign in (-1, 1):
                if len(trace) >= 200:
                    break
                trial = point[:]
                trial[coordinate] += sign * step
                candidate = evaluate(trial)
                candidates.append((candidate, trial))
                if candidate < best:
                    best, best_parameters = candidate, tuple(trial)
            if candidates:
                candidate, trial = min(candidates, key=lambda row: row[0])
                if candidate < value:
                    value, point = candidate, trial
            if len(trace) >= 200:
                break
        step *= 0.8
    return Optimization(
        best_parameters,
        trace[0],
        best,
        tuple(trace),
        "200-evaluation budget exhausted; no convergence certification",
    )
