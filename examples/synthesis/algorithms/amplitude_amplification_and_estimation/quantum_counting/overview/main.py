"""Native QPE counting and bounded count-only iterative amplitude estimation."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.main import (
    build_program,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.quantum_counting_qpe.main import (
    build_program as build_qpe,
)
from flagquantum import synth

from .iterative import Estimate, estimate

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class PhaseEstimate:
    program: synth.Program
    counts: tuple[tuple[Fraction, int], ...]
    phase: Fraction
    solutions: float
    resolution_error: float


class IndicatorSampler:
    """Sample actual native indicator distributions, caching at most17 programs.

    Each local circuit runs once per k. Subsequent calls draw fresh binomial
    counts from its computed marginal. This is not a hardware timing model.
    """

    def __init__(self, seed: int = 123) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.programs: dict[int, synth.Program] = {}
        self.probabilities: dict[int, float] = {}

    def __call__(self, k: int, shots: int) -> int:
        if type(k) is not int or not 0 <= k <= 16:
            raise ValueError("Grover iteration count must be an integer in0..16.")
        if type(shots) is not int or not 1 <= shots <= 2048:
            raise ValueError("Local sample batch must be an integer in1..2048.")
        if k not in self.programs:
            program = build_program(k)
            values = dict(
                program.probabilities(fq.run(program), registers=("indicator",))
            )
            self.programs[k] = program
            self.probabilities[k] = values.get((1,), 0.0)
        count = torch.binomial(
            torch.tensor(float(shots), dtype=torch.float64),
            torch.tensor(self.probabilities[k], dtype=torch.float64),
            generator=self.generator,
        )
        return int(count.item())


def sample_qpe(program: synth.Program | None = None) -> PhaseEstimate:
    """Retain five phase bits; sample2048 local outcomes with seed123."""
    if program is None:
        program = build_qpe()
    values = program.numeric_probabilities(
        fq.run(program), registers=("phase_var",), threshold=0
    )
    phases = tuple(key[0] for key in values)
    indices = torch.multinomial(
        torch.tensor(list(values.values()), dtype=torch.float64),
        2048,
        replacement=True,
        generator=torch.Generator().manual_seed(123),
    )
    frequencies = torch.bincount(indices, minlength=len(phases))
    counts = tuple((x, int(n)) for x, n in zip(phases, frequencies, strict=True))
    phase = phases[int(frequencies.argmax())]
    points = torch.linspace(0, 1, 100, dtype=torch.float64)
    error = (
        16
        * (
            torch.sin(math.pi * points).square()
            - torch.sin(math.pi * (points - 1 / 32)).square()
        )
        .abs()
        .max()
    )
    return PhaseEstimate(
        program, counts, phase, 16 * math.sin(math.pi * float(phase)) ** 2, float(error)
    )


def print_estimate(result: Estimate, epsilon: float) -> None:
    """Report interval width and all actual per-iteration success/failure counts."""
    print("Requested amplitude error:", epsilon, "tail probability:", 0.01)
    print("Estimated solution count:", 16 * result.estimate)
    print("Solution-count interval:", tuple(16 * p for p in result.confidence_interval))
    for index, row in enumerate(result.iterations):
        print(
            index,
            "Grover iterations:",
            row.k,
            "counts:",
            (row.shots - row.successes, row.successes),
        )


def run_iterative(epsilon: float = 1 / 32) -> tuple[Estimate, IndicatorSampler]:
    """Use only sampled indicator counts; known answers are validation-only."""
    sampler = IndicatorSampler()
    result = estimate(sampler, epsilon=epsilon, alpha=0.01)
    print_estimate(result, epsilon)
    return result, sampler


def plot_results(phase: PhaseEstimate, result: Estimate) -> "Figure":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].bar(
        [float(x) for x, _ in phase.counts], [n for _, n in phase.counts], width=0.025
    )
    axes[0].set_xlabel("Fractional phase")
    axes[0].set_ylabel("Count (2048 local samples)")
    axes[0].set_title("Five-bit QPE")
    rows = result.iterations
    lower = [16 * math.sin(r.theta_interval[0]) ** 2 for r in rows]
    upper = [16 * math.sin(r.theta_interval[1]) ** 2 for r in rows]
    axes[1].fill_between(
        range(1, len(rows) + 1),
        lower,
        upper,
        alpha=0.3,
        label="99% simultaneous interval",
    )
    axes[1].axhline(
        6, color="black", linestyle="--", label="Reference only: six solutions"
    )
    axes[1].set_xlabel("Adaptive iteration")
    axes[1].set_ylabel("Solution count")
    axes[1].set_title("Bounded iterative estimate")
    axes[1].legend()
    plt.show()
    return figure


def run_example() -> None:
    phase = sample_qpe()
    print("Phase-resolution count error:", phase.resolution_error)
    print("Most frequent phase:", phase.phase, "estimated solutions:", phase.solutions)
    assert abs(phase.solutions - 6) <= 1.5
    run_iterative()
    refined, _ = run_iterative(1 / (16 * 15))
    assert abs(16 * refined.estimate - 6) <= 1 / 15
    plot_results(phase, refined)


if __name__ == "__main__":
    run_example()
