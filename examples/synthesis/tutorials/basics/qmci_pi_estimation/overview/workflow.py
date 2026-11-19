"""Full QMCI workflow with independent states and actual local observation counts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    Estimate,
    estimate,
)
from examples.synthesis.qpe_reference import dense_qpe
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from .operations import build_program, build_qpe

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class StateCheck:
    bits: int
    k: int
    maximum_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate


@dataclass(frozen=True)
class PhaseEstimate:
    observations: tuple[tuple[Fraction, int], ...]
    phase: Fraction
    source_pi: float
    squared_probability_pi: float
    state_check: StateCheck


def grid_pi(axis_length: int) -> float:
    """Classical finite-grid reference; the argument is an axis length, not bits."""
    if type(axis_length) is not int or not 1 <= axis_length <= 80:
        raise ValueError("use an integer grid axis length in 1..80")
    count = sum(
        x * x + y * y < axis_length * axis_length
        for x in range(axis_length)
        for y in range(axis_length)
    )
    return 4 * count / axis_length**2


def reference_indicator(bits: int, k: int) -> fq.Circuit:
    """Independent marker permutation and reflection, in indicator/x/y order."""
    size = 1 << bits
    points = size * size
    dimension = 2 * points
    marker = torch.zeros((dimension, dimension), dtype=torch.complex128)
    psi = torch.zeros(dimension, dtype=torch.complex128)
    for x in range(size):
        for y in range(size):
            good = int(x * x + y * y < points)
            word = x * size + y
            psi[good * points + word] = 1 / size
            for flag in range(2):
                marker[(flag ^ good) * points + word, flag * points + word] = 1
    reflection = 2 * torch.outer(psi, psi.conj()) - torch.eye(
        dimension, dtype=torch.complex128
    )
    circuit = fq.Circuit(2 * bits + 1, dtype=torch.complex128)
    for wire in range(1, 2 * bits + 1):
        circuit.h(wire)
    circuit.any(*range(2 * bits + 1), matrix=marker)
    for _ in range(k):
        circuit.z(0).any(*range(2 * bits + 1), matrix=reflection)
    return circuit


def check_state(
    program: synth.Program,
    result: fq.ExecutionResult,
    reference: fq.Circuit,
    bits: int,
    k: int,
) -> StateCheck:
    """Check retained coordinates too; drop does not mean a clean environment."""
    expected = fq.run(reference).statevector().flatten()
    actual = result.statevector().reshape(expected.numel(), -1)
    error = float((actual[:, 0] - expected).abs().max())
    dirty = float(actual[:, 1:].abs().square().sum())
    torch.testing.assert_close(actual[:, 0], expected, atol=2e-10, rtol=0)
    if dirty > 1e-24:
        raise AssertionError("QMCI clean work did not return to zero")
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return StateCheck(bits, k, error, dirty, resources)


class IndicatorSampler:
    """Return fresh counts only, using actual native marginals cached per k."""

    def __init__(self, bits: int, seed: int) -> None:
        if type(bits) is not int or not 1 <= bits <= 4:
            raise ValueError("use an integer coordinate precision in 1..4")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("use an integer seed in 0..2**32-1")
        self.bits = bits
        self.generator = torch.Generator().manual_seed(seed)
        self.probabilities: dict[int, float] = {}
        self.checks: dict[int, StateCheck] = {}

    def __call__(self, k: int, shots: int) -> int:
        if type(k) is not int or not 0 <= k <= 16:
            raise ValueError("use an integer Grover count in 0..16")
        if type(shots) is not int or not 1 <= shots <= 2048:
            raise ValueError("use an integer local shot batch in 1..2048")
        if k not in self.probabilities:
            program = build_program(self.bits, k)
            result = fq.run(program)
            self.checks[k] = check_state(
                program, result, reference_indicator(self.bits, k), self.bits, k
            )
            self.probabilities[k] = program.probabilities(
                result, registers=("indicator",)
            ).get((1,), 0.0)
        return int(
            torch.binomial(
                torch.tensor(float(shots), dtype=torch.float64),
                torch.tensor(self.probabilities[k], dtype=torch.float64),
                generator=self.generator,
            ).item()
        )


def run_iterative(bits: int, seed: int) -> tuple[Estimate, IndicatorSampler]:
    """Reuse the bounded example estimator with source epsilon .03 and alpha .01."""
    sampler = IndicatorSampler(bits, seed)
    result = estimate(sampler, epsilon=0.03, alpha=0.01)
    return result, sampler


def sample_qpe(seed: int = 1800) -> PhaseEstimate:
    """Retain source three-bit precision and 2048 local observations."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("use an integer seed in 0..2**32-1")
    program = build_qpe()
    result = fq.run(program)
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    preparation = torch.ones((1, 1), dtype=torch.complex128)
    for _ in range(6):
        preparation = torch.kron(preparation, h)
    oracle = torch.diag(
        torch.tensor(
            [-1 if (w >> 3) ** 2 + (w & 7) ** 2 < 64 else 1 for w in range(64)],
            dtype=torch.complex128,
        )
    )
    zero = torch.eye(64, dtype=torch.complex128)
    zero[0, 0] = -1
    grover = -preparation @ zero @ preparation.mH @ oracle
    check = check_state(program, result, dense_qpe(grover, preparation, 3), 3, -1)
    values = program.numeric_probabilities(result, registers=("phase_reg",))
    probabilities = torch.zeros(8, dtype=torch.float64)
    for (value,), probability in values.items():
        probabilities[int(8 * value)] = probability
    samples = torch.multinomial(
        probabilities,
        2048,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = torch.bincount(samples, minlength=8)
    observations = tuple(
        (Fraction(i, 8), int(n)) for i, n in enumerate(counts.tolist())
    )
    phase = Fraction(int(counts.argmax()), 8)
    amplitude = math.sin(math.pi * float(phase))
    return PhaseEstimate(observations, phase, 4 * amplitude, 4 * amplitude**2, check)


def plot_lattice(bits: int = 4) -> Figure:
    import matplotlib.pyplot as plt

    size = 1 << bits
    figure, axis = plt.subplots(figsize=(4, 4), layout="constrained")
    for inside, color, label in ((True, "blue", "Inside"), (False, "red", "Outside")):
        points = [
            (x, y)
            for x in range(size)
            for y in range(size)
            if (x * x + y * y < size * size) == inside
        ]
        axis.scatter(
            [x for x, _ in points], [y for _, y in points], s=10, c=color, label=label
        )
    angles = torch.linspace(0, math.pi / 2, 100, dtype=torch.float64)
    axis.plot(size * angles.cos(), size * angles.sin(), color="black")
    axis.set(
        xlim=(-0.5, size),
        ylim=(-0.5, size),
        xlabel="x",
        ylabel="y",
        title=f"Lattice points ({bits} bits per axis)",
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.4)
    axis.legend()
    return figure


def plot_estimates(results: tuple[Estimate, ...]) -> Figure:
    import matplotlib.pyplot as plt

    if len(results) != 4:
        raise ValueError("provide estimates for all four source coordinate widths")
    figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    estimates = [4 * row.estimate for row in results]
    lower = [
        value - 4 * row.confidence_interval[0]
        for value, row in zip(estimates, results, strict=True)
    ]
    upper = [
        4 * row.confidence_interval[1] - value
        for value, row in zip(estimates, results, strict=True)
    ]
    axis.errorbar(
        [4**bits for bits in range(1, 5)],
        estimates,
        yerr=[lower, upper],
        fmt="o",
        color="red",
        capsize=5,
        label="Bounded iterative QMCI (local)",
    )
    sizes = (2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80)
    axis.plot(
        [n * n for n in sizes],
        [grid_pi(n) for n in sizes],
        "--",
        color="black",
        label="Classical finite grid",
    )
    axis.axhline(math.pi, color="green", linestyle="-.", label="True pi")
    axis.set(
        xscale="log",
        ylim=(2.8, 4.2),
        xlabel="Lattice points (not actual shots)",
        ylabel="Estimated pi",
    )
    axis.legend()
    return figure
