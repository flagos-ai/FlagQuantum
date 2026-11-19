"""Run all six source QAOA layers with exact local CVaR and finite samples."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import Evaluation
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult

from ..main import main
from .model import (
    DECISIONS,
    PHYSICAL_WIDTH,
    QUANTILE,
    REGISTER_NAMES,
    Model,
    energies,
    exact_solutions,
    fresh_model,
)

LAYERS = 6
ITERATIONS = 70
SHOTS = 1000
INITIAL = tuple(math.pi * (i + 1) / LAYERS for i in range(LAYERS)) + tuple(
    math.pi * (LAYERS - i) / LAYERS for i in range(LAYERS)
)


def build_program(parameters: tuple[float, ...]) -> synth.Program:
    if len(parameters) != 2 * LAYERS or any(
        type(angle) not in (int, float) or not math.isfinite(angle)
        for angle in parameters
    ):
        raise ValueError("the six layers require twelve finite angles")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters},
        max_qubits=PHYSICAL_WIDTH,
        max_workspace=0,
        max_operations=4096,
        memory_limit_bytes=64 * 1024**2,
    )


def distribution(program: synth.Program) -> torch.Tensor:
    """Read masses in joint numeric register order, least significant bit first."""
    if tuple(program.registers) != REGISTER_NAMES:
        raise ValueError("unexpected named decision or slack registers")
    # Named results store each register most significant bit first. Move the
    # decision field to the least significant end without reversing its bits.
    wires = tuple(
        wire for name in reversed(REGISTER_NAMES) for wire in program.registers[name]
    )
    if sorted(wires) != list(range(PHYSICAL_WIDTH)):
        raise ValueError(
            "the original encoding requires twenty logical and physical bits"
        )
    state = fq.run(program).statevector()
    probabilities = (
        state.reshape((2,) * PHYSICAL_WIDTH).permute(*wires).reshape(-1).abs().square()
    )
    assert isinstance(probabilities, torch.Tensor)
    return probabilities


def lower_cvar(probabilities: torch.Tensor, costs: torch.Tensor) -> float:
    """Use the exact full joint distribution, including partially selected tie bins."""
    if (
        probabilities.shape != costs.shape
        or abs(float(probabilities.sum()) - 1.0) > 1e-10
    ):
        raise ValueError("cost and normalized probability vectors must match")
    mass = torch.bincount(costs, weights=probabilities)
    prior = mass.cumsum(0) - mass
    take = torch.minimum(mass, (QUANTILE - prior).clamp_min(0))
    return float(take @ torch.arange(mass.numel(), dtype=mass.dtype)) / QUANTILE


@dataclass(frozen=True)
class Sample:
    word: int
    decisions: int
    cost: int
    count: int
    feasible: bool


def sample(
    probabilities: torch.Tensor, costs: torch.Tensor, model: Model, seed: int
) -> tuple[Sample, ...]:
    draws = torch.multinomial(
        probabilities,
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=1 << PHYSICAL_WIDTH)
    words = torch.nonzero(counts, as_tuple=False).flatten()
    rows = []
    for raw_word in words:
        word = int(raw_word)
        decision = word & ((1 << DECISIONS) - 1)
        rows.append(
            Sample(
                word,
                decision,
                int(costs[word]),
                int(counts[word]),
                model.dominates(decision),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.cost, row.word)))


@dataclass(frozen=True)
class Workflow:
    model: Model
    optimizer: NelderMeadResult
    program: synth.Program
    trace: tuple[Evaluation, ...]
    optimized: tuple[Sample, ...]
    uniform: tuple[Sample, ...]
    classical: tuple[int, ...]


def run_workflow(*, iterations: int = ITERATIONS) -> Workflow:
    if type(iterations) is not int or not 1 <= iterations <= ITERATIONS:
        raise ValueError("iterations must be in [1, 70]")
    model = fresh_model()
    costs = energies(model)
    trace: list[Evaluation] = []

    def evaluate(point: torch.Tensor) -> torch.Tensor:
        parameters = tuple(float(angle) for angle in point)
        value = lower_cvar(distribution(build_program(parameters)), costs)
        trace.append(Evaluation(parameters, value))
        return torch.tensor(value, dtype=torch.float64)

    optimizer = NelderMeadOptimizer(maxiter=iterations, initial_step=0.07).minimize(
        evaluate, torch.tensor(INITIAL, dtype=torch.float64)
    )
    program = build_program(tuple(float(angle) for angle in optimizer.parameters))
    optimized = sample(distribution(program), costs, model, 2718)
    uniform = sample(
        distribution(build_program((0.0,) * (2 * LAYERS))), costs, model, 31415
    )
    return Workflow(
        model,
        optimizer,
        program,
        tuple(trace),
        optimized,
        uniform,
        exact_solutions(model),
    )
