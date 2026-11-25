"""Sample each original time point and report actual exit-word counts."""

from dataclasses import dataclass

import torch

from .quantum import evolve

SHOTS = 8192


def time_points(qubits: int) -> tuple[float, ...]:
    if qubits not in (6, 20):
        raise ValueError("choose six or twenty qubits")
    return tuple(float(2 * (qubits - 2) + offset) for offset in range(-12, 13, 2))


def exit_word(qubits: int) -> int:
    if qubits not in (6, 20):
        raise ValueError("choose six or twenty qubits")
    return (1 << (qubits - 1)) - 3


@dataclass(frozen=True)
class Point:
    time: float
    shots: int
    exit_count: int
    exact_probability: float
    operations: int
    workspace: int

    @property
    def sampled_probability(self) -> float:
        return self.exit_count / self.shots


def run_range(qubits: int) -> tuple[Point, ...]:
    target = exit_word(qubits)
    generator = torch.Generator().manual_seed(9000 + qubits)
    results = []
    for time in time_points(qubits):
        state = evolve(qubits, time)
        probabilities = state.amplitudes.abs().square()
        draws = torch.multinomial(
            probabilities, SHOTS, replacement=True, generator=generator
        )
        results.append(
            Point(
                time,
                SHOTS,
                int((draws == target).sum()),
                float(probabilities[target]),
                state.operations,
                state.work_qubits,
            )
        )
    return tuple(results)
