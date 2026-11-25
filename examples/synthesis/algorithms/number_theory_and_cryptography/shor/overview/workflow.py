"""Complete Shor Notebook workflow, reusing native period finding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from matplotlib.figure import Figure

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate

from ..comparisons.circuit import reference_circuit
from ..main import build_program
from .classical import factors_from_order, recover_order


@dataclass(frozen=True)
class Estimate:
    """Actual sampled phases, recovered period and independent-state evidence."""

    seed: int
    observations: tuple[tuple[Fraction, int], ...]
    fractions: tuple[Fraction, ...]
    order: int
    factors: tuple[int, int]
    maximum_state_error: float
    analytic_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate

    @property
    def shots(self) -> int:
        return sum(count for _, count in self.observations)


def run_estimate(*, seed: int = 1700) -> Estimate:
    """Preserve original 21/11, ten phase bits and 2048 actual observations."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("use an integer seed in 0..2**32-1")
    program = build_program()
    result = fq.run(program)
    actual = result.statevector().reshape(32768, -1)
    expected = fq.run(reference_circuit()).statevector().flatten()
    analytic = torch.zeros((1024, 32), dtype=torch.complex128)
    for power in range(1024):
        analytic[power, pow(11, power, 21)] = 1 / 32
    analytic = torch.fft.fft(analytic, dim=0, norm="ortho").flatten()
    torch.testing.assert_close(expected, analytic, atol=2e-11, rtol=0)
    torch.testing.assert_close(actual[:, 0], expected, atol=2e-11, rtol=0)
    dirty = float(actual[:, 1:].abs().square().sum())
    if dirty > 1e-24:
        raise AssertionError("Shor clean workspace did not return to zero")
    numeric = program.numeric_probabilities(result, registers=("phase_var",))
    probabilities = torch.zeros(1024, dtype=torch.float64)
    for (phase,), probability in numeric.items():
        probabilities[int(phase * 1024)] = probability
    samples = torch.multinomial(
        probabilities,
        2048,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    reference_probabilities = expected.reshape(1024, 32).abs().square().sum(dim=-1)
    reference_samples = torch.multinomial(
        reference_probabilities,
        2048,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    if not torch.equal(samples, reference_samples):
        raise AssertionError("same-seed independent Shor observations disagree")
    counts = torch.bincount(samples, minlength=1024)
    observations = tuple(
        sorted(
            (
                (Fraction(index, 1024), int(count))
                for index, count in enumerate(counts.tolist())
                if count
            ),
            key=lambda row: (-row[1], row[0]),
        )
    )
    fractions, order = recover_order(observations, 21)
    if any((value * 6).denominator != 1 for value in fractions):
        raise AssertionError("source fraction assertion failed")
    if order != 6:
        raise AssertionError("observations did not recover the source period")
    factors = factors_from_order(11, order, 21)
    if factors is None or math.prod(factors) != 21:
        raise AssertionError("sampled order did not factor the source modulus")
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return Estimate(
        seed,
        observations,
        fractions,
        order,
        factors,
        float((actual[:, 0] - expected).abs().max()),
        float((expected - analytic).abs().max()),
        dirty,
        resources,
    )


def plot_estimate(result: Estimate) -> Figure:
    """Plot actual source-size observation frequencies and extracted fractions."""
    figure = Figure(figsize=(12, 4), layout="constrained")
    axis = figure.subplots()
    axis.bar(
        [float(phase) for phase, _ in result.observations],
        [count / result.shots for _, count in result.observations],
        width=4 / 1024,
    )
    axis.set_xlabel("s/r")
    axis.set_ylabel("Observed probability")
    top = axis.secondary_xaxis("top")
    top.set_xticks([float(value) for value in result.fractions])
    top.set_xticklabels([str(value) for value in result.fractions], color="red")
    top.set_xlabel("Extracted fractions", color="red")
    return figure
