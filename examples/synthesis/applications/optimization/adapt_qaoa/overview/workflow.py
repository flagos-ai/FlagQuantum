"""Run the source ordinary comparison and adaptive commutator stop rule."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.graph_optimization import Evaluation
from flagquantum import synth
from flagquantum.algorithms import CobylaOptimizer

from .model import Mixer, costs, gradients, pool
from .quantum import build_program

BUDGET = 60
THRESHOLD = 0.027
SHOTS = 1000
LAYER_LIMIT = 24


class _ObjectiveBudgetError(Exception):
    """Refuse another objective call after the source budget."""


@dataclass(frozen=True)
class Optimization:
    program: synth.Program
    parameters: tuple[float, ...]
    trace: tuple[Evaluation, ...]
    budget_stop: bool
    converged: bool


@dataclass(frozen=True)
class Layer:
    mixers: tuple[int, ...]
    optimization: Optimization
    gradient_norm: float
    next_mixer: int


@dataclass(frozen=True)
class Observation:
    word: int
    energy: float
    count: int


@dataclass(frozen=True)
class Workflow:
    ordinary: Optimization
    layers: tuple[Layer, ...]
    ordinary_samples: tuple[Observation, ...]
    adaptive_samples: tuple[Observation, ...]


def state(program: synth.Program) -> torch.Tensor:
    if program.registers["v"] != tuple(range(8)):
        raise ValueError("expected eight logical modes in source bit order")
    statevector = fq.run(program).statevector().flatten()
    assert isinstance(statevector, torch.Tensor)
    return statevector


def optimize(
    mixers: tuple[Mixer, ...], diagonal: torch.Tensor, initial_radius: float
) -> Optimization:
    layers = len(mixers)
    initial = torch.cat(
        (
            torch.linspace(0, 1, layers, dtype=torch.float64),
            torch.linspace(1, 0, layers, dtype=torch.float64),
        )
    )
    trace: list[Evaluation] = []

    def evaluate(point: torch.Tensor) -> torch.Tensor:
        if len(trace) >= BUDGET:
            raise _ObjectiveBudgetError
        parameters = tuple(float(angle) for angle in point)
        value = state(build_program(mixers, parameters)).abs().square() @ diagonal
        trace.append(Evaluation(parameters, float(value)))
        return value

    stopped = converged = False
    try:
        result = CobylaOptimizer(
            maxiter=100, rhobeg=initial_radius, rhoend=1e-8
        ).minimize(evaluate, initial)
        converged = result.converged
    except _ObjectiveBudgetError:
        stopped = True
    best = min(trace, key=lambda row: row.cost)
    return Optimization(
        build_program(mixers, best.parameters),
        best.parameters,
        tuple(trace),
        stopped,
        converged,
    )


def draw(
    program: synth.Program, diagonal: torch.Tensor, seed: int
) -> tuple[Observation, ...]:
    draws = torch.multinomial(
        state(program).abs().square(),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(draws, minlength=256).tolist()
    rows = (
        Observation(word, float(diagonal[word]), int(count))
        for word, count in enumerate(counts)
        if count
    )
    return tuple(sorted(rows, key=lambda row: (row.energy, row.word)))


def run_workflow() -> Workflow:
    diagonal = costs()
    candidates = pool()
    ordinary = optimize((candidates[0],) * 3, diagonal, 1.0)
    selected = [0]
    history = []
    for _ in range(LAYER_LIMIT):
        current = tuple(candidates[i] for i in selected)
        result = optimize(current, diagonal, 0.5)
        gradient_state = state(
            build_program(current, result.parameters, extra_gamma=0.01)
        )
        values = gradients(gradient_state, diagonal, candidates)
        norm = float(values.norm() / len(candidates))
        winner = int(values.argmax())
        history.append(Layer(tuple(selected), result, norm, winner))
        if norm < THRESHOLD:
            break
        selected.append(winner)
    else:
        raise ValueError(
            "source gradient threshold was not reached within the local layer budget"
        )
    return Workflow(
        ordinary,
        tuple(history),
        draw(ordinary.program, diagonal, 8031),
        draw(history[-1].optimization.program, diagonal, 8032),
    )
