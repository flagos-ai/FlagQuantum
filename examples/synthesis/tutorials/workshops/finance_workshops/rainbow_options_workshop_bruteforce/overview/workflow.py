"""Run the workshop's 20,000-shot count-only payoff estimate."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.overview.model import (
    ALPHA,
    ASSERTION_ALPHA,
    EPSILON,
    STRIKE,
    Model,
    fresh_model,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.overview.workflow import (
    reference_circuit,
)
from flagquantum import synth

from ..main import main as payoff_preparation

SHOTS = 20_000
SEED = 9137


@dataclass(frozen=True)
class Pricing:
    model: Model
    successes: int
    probability: float
    amplitude_interval: tuple[float, float]
    price: float
    price_interval: tuple[float, float]
    source_tolerance: float
    state_error: float


def maximum_asset_price(model: Model) -> float:
    """Evaluate the original zero-payoff guard in the unrounded asset space."""
    days = 250
    stdev = math.sqrt(0.000335)
    cross = 0.000257 / stdev
    row_sums = stdev, cross + math.sqrt(0.000418 - cross**2)
    return max(
        spot * math.exp(days * mean + model.grid[-1] * math.sqrt(days) * slope)
        for spot, mean, slope in zip(
            (193.97, 189.12), (0.00050963, 0.00062552), row_sums, strict=True
        )
    )


def build_program(model: Model, *, max_width: int) -> synth.Program:
    """Retain the workshop allocation and compile the first estimation power."""
    if max_width not in (14, 25):
        raise ValueError("Use the workshop preparation or estimation width budget")
    return synth.synthesize(
        payoff_preparation,
        classical_inputs={
            "k": 0,
            "probabilities": model.probabilities,
            "angles": model.angles,
        },
        max_qubits=max_width,
        max_workspace=0,
        max_operations=8192,
        memory_limit_bytes=1024**2,
    )


def run_pricing() -> Pricing:
    """Use the first actual batch and check the precision before stopping."""
    model = fresh_model()
    if maximum_asset_price(model) < STRIKE:
        raise ValueError("The strike exceeds the original asset grid")
    program = build_program(model, max_width=25)
    result = fq.run(program)
    measured = result.statevector().flatten()
    expected = fq.run(reference_circuit(model, 0)).statevector().flatten()
    error = float((measured - expected).abs().max())
    if error > 1e-11:
        raise AssertionError(
            "The complete native preparation differs from the reference"
        )

    # The original indicator is the only named output of the preparation.
    weights = program.probabilities(result, registers=("indicator",))
    samples = torch.multinomial(
        torch.tensor(
            [weights.get((0,), 0.0), weights.get((1,), 0.0)], dtype=torch.float64
        ),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    successes = int(samples.sum())
    probability = successes / SHOTS
    radius = math.sqrt(math.log(2 / ALPHA) / (2 * SHOTS))
    lower, upper = max(0.0, probability - radius), min(1.0, probability + radius)
    if upper - lower > 2 * EPSILON:
        raise ValueError(
            "The workshop's first batch did not reach the requested precision"
        )

    price = probability * model.scale - STRIKE
    price_interval = lower * model.scale - STRIKE, upper * model.scale - STRIKE
    tolerance = (
        0.5
        * (price_interval[1] - price_interval[0])
        * math.sqrt(math.log(ALPHA / ASSERTION_ALPHA))
    )
    return Pricing(
        model,
        successes,
        probability,
        (lower, upper),
        price,
        price_interval,
        tolerance,
        error,
    )
