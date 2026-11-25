"""Reuse native reversible loading and estimate the payoff from sampled counts."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    Estimate,
    estimate,
)
from examples.synthesis.amplification_reference import amplified_circuit
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.main import (
    main as rainbow,
)
from examples.synthesis.qpe_reference import probability_preparation
from flagquantum import synth

from .model import (
    ALPHA,
    ASSERTION_ALPHA,
    EPSILON,
    SOURCE_PRICE,
    STRIKE,
    Model,
    fresh_model,
)


@dataclass(frozen=True)
class Pricing:
    model: Model
    estimate: Estimate
    price: float
    interval: tuple[float, float]
    source_tolerance: float
    state_errors: tuple[tuple[int, float], ...]


def build_program(model: Model, power: int, max_width: int) -> synth.Program:
    """Bind the freshly generated tables within the original width budgets."""
    if type(power) is not int or not 0 <= power <= 16:
        raise ValueError("Use a bounded Grover power in 0..16")
    if max_width not in (14, 25):
        raise ValueError("Use the original preparation or estimation width budget")
    return synth.synthesize(
        rainbow,
        classical_inputs=dict(
            k=power, probabilities=model.probabilities, angles=model.angles
        ),
        max_qubits=max_width,
        max_workspace=0,
        max_operations=8192,
        memory_limit_bytes=1024**2,
    )


def reference_circuit(model: Model, power: int) -> fq.Circuit:
    """Build the 32-dimensional joint loader directly from the raw payoff."""
    rotation = torch.zeros((32, 32), dtype=torch.complex128)
    for word, index in enumerate(model.selectors):
        payoff = model.payoff_grid[index] / model.scale
        c, s = math.sqrt(1 - payoff), math.sqrt(payoff)
        rotation[word, word] = rotation[word + 16, word + 16] = c
        rotation[word, word + 16], rotation[word + 16, word] = -s, s
    loader = probability_preparation(model.probabilities)
    preparation = rotation @ torch.kron(
        torch.eye(2, dtype=torch.complex128), torch.kron(loader, loader)
    )
    return amplified_circuit(preparation, power)


def run_pricing() -> Pricing:
    """Run the local iterative estimator with the source epsilon and alpha."""
    model = fresh_model()
    build_program(model, 0, 14)
    generator = torch.Generator().manual_seed(9137)
    verified: dict[int, tuple[float, float]] = {}

    def sample(power: int, shots: int) -> int:
        if power not in verified:
            program = build_program(model, power, 25)
            result = fq.run(program)
            actual = result.statevector().flatten()
            expected = fq.run(reference_circuit(model, power)).statevector().flatten()
            error = float((actual - expected).abs().max())
            probability = program.probabilities(result, registers=("indicator",)).get(
                (1,), 0.0
            )
            if (
                error > 1e-11
                or abs(probability - float(expected[16:].abs().square().sum())) > 1e-11
            ):
                raise AssertionError(
                    "Native amplification state or named indicator differs"
                )
            verified[power] = probability, error
        probability, _ = verified[power]
        count = torch.binomial(
            torch.tensor(float(shots), dtype=torch.float64),
            torch.tensor(probability, dtype=torch.float64),
            generator=generator,
        )
        return int(count)

    estimated = estimate(sample, epsilon=EPSILON, alpha=ALPHA)
    price = estimated.estimate * model.scale - STRIKE
    lower, upper = estimated.confidence_interval
    interval = lower * model.scale - STRIKE, upper * model.scale - STRIKE
    tolerance = (
        0.5 * (interval[1] - interval[0]) * math.sqrt(math.log(ALPHA / ASSERTION_ALPHA))
    )
    if abs(price - SOURCE_PRICE) > tolerance:
        raise AssertionError("The observed price exceeds the source confidence rule")
    return Pricing(
        model,
        estimated,
        price,
        interval,
        tolerance,
        tuple((power, row[1]) for power, row in verified.items()),
    )
