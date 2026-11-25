"""Thirty spin-preserving singles and the original bounded VQE experiment."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.lih_chemistry.overview.variational import (
    ansatz,
)
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.algorithms.fermion.encoding import encode_fermion
from flagquantum.algorithms.fermion.lowering import lift_stabilizer_state
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.core.algebra.pauli import PauliOperator
from flagquantum.observables.fermion import annihilate, create

from .model import OCCUPIED, WIDTH, Model, trim

PAIRS = tuple(
    (2 * occupied + spin, 2 * virtual + spin)
    for occupied in range(OCCUPIED)
    for virtual in range(OCCUPIED, WIDTH // 2)
    for spin in range(2)
)
BUDGET, TOLERANCE = 100, 1e-4


@dataclass(frozen=True)
class Trial:
    parameters: tuple[float, ...]
    energy: float


@dataclass(frozen=True)
class Optimization:
    trace: tuple[Trial, ...]
    selected: Trial
    physical_trimmed_energy: float
    physical_untrimmed_energy: float
    converged: bool
    query_budget_reached: bool


class QueryBudgetError(Exception):
    """Stop before an unapproved one-hundred-and-first actual objective."""


def operators(model: Model) -> tuple[PauliOperator, ...]:
    result = []
    for occupied, virtual in PAIRS:
        excitation = create(occupied) * annihilate(virtual)
        generator = 1j * (excitation - excitation.dagger())
        encoded = encode_fermion(generator, n_modes=WIDTH).operator
        result.append(trim(model.reduction.reduce_operator(encoded), 1e-12))
    return tuple(result)


def circuit(
    model: Model,
    generators: tuple[PauliOperator, ...],
    parameters: tuple[float, ...],
) -> fq.Circuit:
    if len(generators) != 30 or len(parameters) != 30:
        raise ValueError("The original singles ansatz requires thirty angles")
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("The singles parameters must be finite")
    return ansatz(14, model.hf_word, generators, parameters)


def independent_occupation_state(parameters: tuple[float, ...]) -> torch.Tensor:
    """Apply signed two-determinant rotations without Pauli mapping or lowering."""
    indices = torch.arange(1 << WIDTH)
    state = torch.zeros(1 << WIDTH, dtype=torch.complex128)
    state[((1 << 10) - 1) << 6] = 1
    for (occupied, virtual), angle in zip(PAIRS, parameters, strict=True):
        occupied_mask, virtual_mask = 1 << (15 - occupied), 1 << (15 - virtual)
        source = indices[
            ((indices & occupied_mask) == 0) & ((indices & virtual_mask) != 0)
        ]
        target = source ^ occupied_mask ^ virtual_mask
        parity = torch.zeros_like(source)
        for mode in range(occupied + 1, virtual):
            parity ^= (source >> (15 - mode)) & 1
        sign = (1 - 2 * parity).to(torch.float64)
        a, b = state[source].clone(), state[target].clone()
        state[source] = math.cos(angle) * a - sign * math.sin(angle) * b
        state[target] = sign * math.sin(angle) * a + math.cos(angle) * b
    return state


def check_states(
    model: Model, generators: tuple[PauliOperator, ...]
) -> tuple[float, ...]:
    errors = []
    for scale in (0.17, -0.23):
        angles = tuple(scale * (index + 1) / 30 for index in range(30))
        reduced = circuit(model, generators, angles).state().reshape(-1)
        actual = lift_stabilizer_state(model.reduction, reduced)
        expected = independent_occupation_state(angles)
        errors.append(float((actual - expected).abs().max()))
    if max(errors) > 1e-12:
        raise ValueError(
            "The complete singles states disagree with occupation rotations"
        )
    return tuple(errors)


def optimize(model: Model, generators: tuple[PauliOperator, ...]) -> Optimization:
    cost = to_observable(model.trimmed)
    trace: list[Trial] = []

    def evaluate(point: torch.Tensor) -> torch.Tensor:
        if len(trace) >= BUDGET:
            raise QueryBudgetError
        angles = tuple(float(value) for value in point)
        result = fq.run(
            circuit(model, generators, angles), outputs=[fq.expectation(cost)]
        )
        value = torch.as_tensor(result.expectation()).reshape(-1).sum().real
        trace.append(Trial(angles, float(value)))
        return value

    converged = stopped = False
    try:
        result = CobylaOptimizer(maxiter=BUDGET, rhobeg=1, rhoend=TOLERANCE).minimize(
            evaluate, torch.zeros(30, dtype=torch.float64)
        )
        converged = result.converged
    except QueryBudgetError:
        stopped = True
    selected = min(trace, key=lambda trial: trial.energy)
    physical = fq.run(
        circuit(model, generators, selected.parameters),
        outputs=[fq.expectation(to_observable(model.physical_trimmed))],
    )
    energy = float(torch.as_tensor(physical.expectation()).reshape(-1).sum().real)
    full = fq.run(
        circuit(model, generators, selected.parameters),
        outputs=[fq.expectation(to_observable(model.reduction.operator))],
    )
    untrimmed = float(torch.as_tensor(full.expectation()).reshape(-1).sum().real)
    return Optimization(tuple(trace), selected, energy, untrimmed, converged, stopped)
