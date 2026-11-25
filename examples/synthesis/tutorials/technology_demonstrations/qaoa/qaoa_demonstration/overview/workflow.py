"""Run the native sampled QAOA trace with the original iteration and shot budgets."""

from dataclasses import dataclass
from time import monotonic

import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import Evaluation
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis

from .model import SHOTS, WIDTH, Model, build_model, decode, initial_parameters
from .quantum import build_program, probabilities

MAX_ITERATIONS = 400
MAX_QUERIES = 400


class _QueryBudgetError(Exception):
    """Stop before another native sample would exceed the query budget."""


@dataclass(frozen=True)
class Run:
    model: Model
    trace: tuple[Evaluation, ...]
    iterations: int | None
    budget_reached: bool
    converged: bool
    mean: float
    word: int
    nodes: tuple[int, ...]
    count: int
    clique: bool
    resources: ResourceEstimate
    seconds: float


def run_workflow(
    iterations: int = MAX_ITERATIONS, query_budget: int = MAX_QUERIES
) -> Run:
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError("Use between one and four hundred optimizer iterations")
    if not 1 <= query_budget <= MAX_QUERIES:
        raise ValueError("Use between one and four hundred measured queries")
    start = monotonic()
    model = build_model()
    energy = torch.tensor(model.energies, dtype=torch.float64)
    generator = torch.Generator().manual_seed(2)
    trace: list[Evaluation] = []

    def sample(parameters: tuple[float, ...]) -> tuple[int, ...]:
        weights = torch.tensor(
            probabilities(build_program(model, parameters)), dtype=torch.float64
        )
        draws = torch.multinomial(weights, SHOTS, replacement=True, generator=generator)
        return tuple(map(int, torch.bincount(draws, minlength=1 << WIDTH)))

    def evaluate(parameters: torch.Tensor) -> torch.Tensor:
        if len(trace) >= query_budget:
            raise _QueryBudgetError()
        values = tuple(map(float, parameters))
        count = torch.tensor(sample(values), dtype=torch.float64)
        mean = energy.dot(count) / SHOTS
        trace.append(Evaluation(values, float(mean)))
        if len(trace) % 25 == 0:
            print(
                f"Completed objective queries: {len(trace)}/{query_budget}, "
                f"lowest mean energy {min(entry.cost for entry in trace):.6f}",
                flush=True,
            )
        return mean

    try:
        result = CobylaOptimizer(maxiter=iterations, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate, torch.tensor(initial_parameters(model), dtype=torch.float64)
        )
    except _QueryBudgetError:
        result = None
    best = min(trace, key=lambda entry: entry.cost)
    counts = sample(best.parameters)
    word = min(
        (i for i, count in enumerate(counts) if count), key=lambda i: model.energies[i]
    )
    nodes = decode(word)
    selected = set(nodes)
    clique = not any({left, right} <= selected for left, right in model.conflicts)
    native = build_program(model, best.parameters)
    converted = convert_basis(
        native.circuit, gates=("cx", "u"), max_added_operations=8192
    )
    converted_state = fq.run(converted.program).statevector().flatten()
    native_state = fq.run(native).statevector().flatten()
    torch.testing.assert_close(
        converted_state * torch.vdot(converted_state, native_state),
        native_state,
        atol=1e-12,
        rtol=0,
    )
    return Run(
        model,
        tuple(trace),
        result.iterations if result is not None else None,
        result is None,
        result.converged if result is not None else False,
        best.cost,
        word,
        nodes,
        counts[word],
        clique,
        estimate_resources(converted.program),
        monotonic() - start,
    )
