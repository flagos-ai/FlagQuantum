"""Verify both full demonstrations and every coherent retained price branch."""

import math
from dataclasses import dataclass
from fractions import Fraction

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.brownian_chebyshev_polynomials.comparisons.circuit import (
    prices,
    reference_circuit,
)
from flagquantum.compiler import ResourceEstimate, estimate_resources

from . import model, quantum


@dataclass(frozen=True)
class Result:
    gaussian_four: model.Gaussian
    gaussian_one: model.Gaussian
    sine: tuple[Fraction, ...]
    recurrence: tuple[tuple[Fraction, ...], ...]
    distribution: dict[tuple[Fraction, ...], float]
    state_error: float
    workspace: float
    nonzero_words: int
    sine_resources: ResourceEstimate
    recurrence_resources: ResourceEstimate
    price_resources: ResourceEstimate


def verify_demo(
    circuit: fq.Circuit,
    rows: tuple[tuple[Fraction, ...], ...],
    widths: tuple[int, ...],
    precision: tuple[int, ...],
) -> None:
    expected = torch.zeros(1 << circuit.n_qubits, dtype=torch.complex128)
    for word, row in enumerate(rows):
        number = word
        for value, width, fraction in zip(row, widths, precision, strict=True):
            scaled = value * (1 << fraction)
            if scaled.denominator != 1:
                raise ValueError("The original sampled arithmetic must remain exact")
            number = (number << width) | (int(scaled) % (1 << width))
        expected[number] = 1 / math.sqrt(len(rows))
    actual: torch.Tensor = circuit.state().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def run() -> Result:
    fine, binary = model.gaussian(4), model.gaussian(1)
    sine = model.sine_values()
    chebyshev = model.chebyshev_values()
    s = quantum.sine_demo()
    verify_demo(s, tuple((value,) for value in sine), quantum.SINE_WIDTHS, (6,))
    c = quantum.chebyshev_demo()
    verify_demo(
        c, tuple(row[1:] for row in chebyshev), quantum.RECURRENCE_WIDTHS, (0, 1, 3, 5)
    )
    program = quantum.build_price(binary.probabilities)
    result = fq.run(program)
    actual = result.statevector().flatten()
    expected: torch.Tensor = reference_circuit().state().flatten()
    error = float((actual - expected).abs().max())
    workspace = float(actual.reshape(-1, 4)[:, 1:].abs().square().sum())
    nonzero = int((actual.abs() > 1e-12).sum())
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("Expected the measured synthesis resources")
    if error > 1e-12 or workspace > 1e-24 or nonzero != 32 or resources.n_qubits > 139:
        raise ValueError(
            "The complete price model failed its constraints or full-state check"
        )
    distribution = dict(
        program.numeric_probabilities(
            result, registers=("t", "price"), max_amplitudes=1 << 23
        )
    )
    independent: dict[tuple[Fraction, ...], float] = {}
    for row, signed in enumerate(prices()):
        key = (Fraction(row & 1), Fraction(signed, 256))
        independent[key] = independent.get(key, 0.0) + 1 / 32
    if distribution.keys() != independent.keys() or any(
        abs(value - independent[key]) > 1e-12 for key, value in distribution.items()
    ):
        raise ValueError(
            "The named price marginal differs from the coherent branch reference"
        )
    return Result(
        fine,
        binary,
        sine,
        chebyshev,
        distribution,
        error,
        workspace,
        nonzero,
        estimate_resources(s),
        estimate_resources(c),
        resources,
    )
