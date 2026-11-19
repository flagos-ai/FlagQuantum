"""Bind freshly generated distances into the retained native twelve-layer qfunc."""

from typing import cast

import torch

import flagquantum as fq
from examples.synthesis.applications.logistics.vehicle_routing_problem.main import main
from flagquantum import synth

from .problem import LAYERS, Problem


def initial_parameters() -> tuple[float, ...]:
    return tuple(
        value
        for layer in range(LAYERS)
        for value in (
            torch.pi * (2 * layer + 1) / (2 * LAYERS),
            torch.pi * (1 - (2 * layer + 1) / (2 * LAYERS)),
        )
    )


def build(problem: Problem, parameters: tuple[float, ...]) -> synth.Program:
    if len(parameters) != 2 * LAYERS:
        raise ValueError("twelve layers require twenty-four parameters")
    return synth.synthesize(
        main,
        classical_inputs={
            "params": list(parameters),
            "departure": [problem.distances[0][city] for city in range(2, 6)],
            "arrival": [problem.distances[1][city] for city in range(2, 6)],
            "distances": [
                problem.distances[first][second]
                for first in range(2, 6)
                for second in range(2, 6)
            ],
        },
        max_qubits=16,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )


def probabilities(problem: Problem, parameters: tuple[float, ...]) -> torch.Tensor:
    program = build(problem, parameters)
    state = cast(torch.Tensor, fq.run(program).statevector()).reshape(-1)
    return state.abs().square()


def logical_costs(costs: torch.Tensor) -> torch.Tensor:
    """Native QArray element zero is the least significant logical word bit."""
    words = torch.arange(costs.numel(), dtype=torch.int64)
    reversed_words = torch.zeros_like(words)
    for bit in range(16):
        reversed_words |= ((words >> bit) & 1) << (15 - bit)
    return costs[reversed_words]
