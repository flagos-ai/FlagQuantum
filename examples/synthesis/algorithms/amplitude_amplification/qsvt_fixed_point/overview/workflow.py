"""Report actual fixed-point search samples and the fitted binomial expectation."""

import math
from dataclasses import dataclass

import numpy as np
import torch

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate

from .fitting import PhaseFit, polynomial
from .quantum import build_program, matching_words

SHOTS = 2048
SAMPLE_SEED = 6703


@dataclass(frozen=True)
class Outcome:
    """A full observed sample and the independent fitted probability."""

    counts: tuple[tuple[tuple[int, ...], int], ...]
    good: int
    expected: float
    sigma: float
    actual_probability: float
    predicted_probability: float
    resources: ResourceEstimate


def run_search(fitted: PhaseFit) -> Outcome:
    """Draw the source 2048 shots from the evaluated native state."""
    program = build_program(fitted.phases)
    result = fq.run(program)
    observed = program.probabilities(
        result, registers=("a", "b", "c", "aux"), threshold=0.0
    )
    words = tuple(sorted(observed))
    weights = torch.tensor([observed[word] for word in words], dtype=torch.float64)
    draws = torch.multinomial(
        weights,
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(SAMPLE_SEED),
    )
    counts = tuple(
        (word, count)
        for word, count in zip(
            words, torch.bincount(draws, minlength=len(words)).tolist(), strict=True
        )
        if count
    )
    marked = tuple((*word, 0) for word in matching_words())
    good = sum(count for word, count in counts if word in marked)
    actual = math.fsum(observed[word] for word in marked)
    overlap = math.sqrt(len(marked) / 64)
    predicted = float(polynomial(np.asarray([overlap]), fitted.coefficients)[0] ** 2)
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("Expected native synthesis resources")
    return Outcome(
        counts,
        good,
        SHOTS * predicted,
        math.sqrt(SHOTS * predicted * (1 - predicted)),
        actual,
        predicted,
        resources,
    )


def show_fit(fit: PhaseFit) -> None:
    """Display the actual target error, phase error and optimization counts."""
    print("Degree and phase count:", len(fit.coefficients) - 1, len(fit.phases))
    print("Alternating fit iterations:", fit.polynomial_iterations)
    print("Phase warmup evaluations:", fit.phase_evaluations)
    print(f"Measured uniform target error: {fit.target_error:.12g}")
    print(f"Phase and polynomial agreement error: {fit.phase_error:.12g}")


def show_outcome(outcome: Outcome) -> None:
    """Expose accepted outcomes and the source five-sigma check."""
    for word, count in outcome.counts:
        a, b, c, aux = word
        if a + b == 3 and c - a == 2 and aux == 0:
            print(f"Accepted {word}: {count}")
    print("Actual shots:", sum(count for _, count in outcome.counts))
    print("Measured good shots:", outcome.good)
    print("Expected good shots:", outcome.expected)
    print("Binomial standard deviation:", outcome.sigma)
    print("Actual probability:", outcome.actual_probability)
    print("Fitted probability:", outcome.predicted_probability)
    print("Native circuit resources:", outcome.resources)
