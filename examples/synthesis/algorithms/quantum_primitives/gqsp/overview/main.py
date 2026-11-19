"""Fit native GQSP phases, execute and sample the source cosine-cubed model."""

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.gqsp.main import build_program
from flagquantum import synth

from .phases import Factorization, factor_phases

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class Workflow:
    """Exact native state followed by an explicit finite classical sample."""

    fit: Factorization
    program: synth.Program
    shots: int
    counts: tuple[tuple[tuple[Fraction, Fraction], int], ...]
    points: torch.Tensor
    measured: torch.Tensor
    exact: torch.Tensor
    target: torch.Tensor


def read_workflow(program: synth.Program, fit: Factorization) -> Workflow:
    """Execute this program once, then retain 100000 local multinomial samples."""
    values = program.numeric_probabilities(
        fq.run(program), registers=("x", "ind"), threshold=0
    )
    keys = tuple((x, indicator) for (x, indicator) in values)
    probabilities = np.array(list(values.values()), dtype=np.float64)
    if abs(float(probabilities.sum()) - 1) > 1e-12:
        raise ValueError("Native joint probabilities must be normalized.")
    shots = 100000
    samples = np.random.RandomState(123).multinomial(shots, probabilities)
    bounds = 6 * np.sqrt(probabilities * (1 - probabilities) / shots) + 6 / shots
    if np.any(np.abs(samples / shots - probabilities) > bounds):
        raise ValueError(
            "Local sample exceeds the stated finite-shot acceptance bound."
        )
    counts = tuple((key, int(count)) for key, count in zip(keys, samples, strict=True))
    rows = sorted(
        (x, count / shots, values[(x, indicator)])
        for (x, indicator), count in counts
        if indicator == 0
    )
    points = torch.tensor([float(x) for x, _, _ in rows], dtype=torch.float64)
    measured = torch.tensor([p / 0.99**2 for _, p, _ in rows], dtype=torch.float64)
    exact = torch.tensor([p / 0.99**2 for _, _, p in rows], dtype=torch.float64)
    target = torch.cos(2 * torch.pi * points).pow(6) / 128
    torch.testing.assert_close(exact, target, atol=1e-12, rtol=0)
    print("Fresh native phase rows:", fit.phases)
    print("Finite local samples:", shots, "seed:", 123)
    print("Postselected samples:", sum(count for (_, ind), count in counts if ind == 0))
    print("Maximum exact probability error:", float((exact - target).abs().max()))
    print("Maximum sampled probability error:", float((measured - target).abs().max()))
    print("Resources:", program.report["resources"])
    return Workflow(fit, program, shots, counts, points, measured, exact, target)


def run_workflow() -> Workflow:
    """Retain the source scaling, Laurent shift, quantum width and 100000 shots."""
    coefficients = 0.99 / 8 * np.array([1.0, 0, 3, 0, 3, 0, 1])
    fit = factor_phases(coefficients)
    program = build_program(negative_power=3, phases=fit.phases)
    return read_workflow(program, fit)


def plot_results(workflow: Workflow) -> "Figure":
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(9, 5), layout="constrained")
    axes.plot(
        workflow.points,
        workflow.measured,
        ".",
        label=f"GQSP with {workflow.shots} local samples",
    )
    axes.plot(
        workflow.points, workflow.target, "-", label=r"$\frac{1}{2^N}\cos^6(2\pi x)$"
    )
    axes.set_xlabel("x", fontsize=16)
    axes.set_ylabel("f(x)", fontsize=16)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=14)
    plt.show()
    return figure


def run_example() -> None:
    plot_results(run_workflow())


if __name__ == "__main__":
    run_example()
