"""Retain source conditional objectives and compare the sampled solve to a direct one."""

import random
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.vqls.vqls_with_lcu.main import (
    build_program as build_solution,
)
from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer

from .quantum import MATRIX, TARGET, build_objective

SHOTS = 204800
BUDGET = 2000


class _ObjectiveBudgetError(Exception):
    """Stop before exceeding the source query cap."""


@dataclass(frozen=True)
class Query:
    parameters: tuple[float, ...]
    auxiliary_zero: int
    all_zero: int
    cost: float


@dataclass(frozen=True)
class Solve:
    queries: tuple[Query, ...]
    converged: bool
    budget_stop: bool
    solution: synth.Program
    amplitudes: torch.Tensor
    reference: torch.Tensor
    fidelity: float
    normalized_residual: float


def probabilities(program: synth.Program) -> torch.Tensor:
    if dict(program.registers) != {"auxiliary": (0, 1), "system": (2, 3, 4)}:
        raise ValueError("unexpected postselection register order")
    values = fq.run(program).statevector().flatten().abs().square()
    assert isinstance(values, torch.Tensor)
    return values


def sampled_query(parameters: tuple[float, ...], generator: torch.Generator) -> Query:
    draws = torch.multinomial(
        probabilities(build_objective(parameters)),
        SHOTS,
        replacement=True,
        generator=generator,
    )
    auxiliary_zero = int((draws < 8).sum())
    all_zero = int((draws == 0).sum())
    if auxiliary_zero == 0:
        raise ValueError("no auxiliary-zero outcomes in the actual objective shots")
    return Query(parameters, auxiliary_zero, all_zero, 1 - all_zero / auxiliary_zero)


def run_workflow() -> Solve:
    rng = random.Random(1000)
    initial = torch.tensor(
        [rng.randint(0, 3000) / 1000 for _ in range(9)], dtype=torch.float64
    )
    generator = torch.Generator().manual_seed(81102)
    trace: list[Query] = []

    def evaluate(point: torch.Tensor) -> torch.Tensor:
        if len(trace) == BUDGET:
            raise _ObjectiveBudgetError()
        query = sampled_query(tuple(float(x) for x in point), generator)
        trace.append(query)
        return torch.tensor(query.cost, dtype=torch.float64)

    stopped = converged = False
    try:
        optimized = CobylaOptimizer(maxiter=BUDGET, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, initial
        )
        converged = optimized.converged
    except _ObjectiveBudgetError:
        stopped = True
    # The source wrapper returns the last executed parameter binding.
    solved = build_solution(trace[-1].parameters)
    native = fq.run(solved).statevector().flatten()
    final_phase = native[-1]
    if float(final_phase.abs()) <= 1e-12:
        raise ValueError("cannot align to the source last-component phase convention")
    amplitudes = native / (final_phase / final_phase.abs())
    reference = (TARGET / MATRIX).to(torch.complex128)
    reference /= reference.norm()
    fidelity = float(torch.vdot(reference, amplitudes).abs().square())
    transformed = MATRIX * amplitudes
    error = float((transformed / transformed.norm() - TARGET).norm())
    return Solve(
        tuple(trace), converged, stopped, solved, amplitudes, reference, fidelity, error
    )
