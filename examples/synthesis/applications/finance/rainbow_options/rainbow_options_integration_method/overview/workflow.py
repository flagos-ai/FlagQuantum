"""Estimate the two-asset payoff using the original integration branch."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    Estimate,
    estimate,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.overview.model import (
    ALPHA,
    ASSERTION_ALPHA,
    EPSILON,
    FRACTION_BITS,
    SELECTOR_BITS,
    SOURCE_PRICE,
    STRIKE,
    Model,
    fresh_model,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.main import (
    main as integration,
)
from flagquantum import synth

MAX_WIDTH = 25
MAX_POWER = 4


@dataclass(frozen=True)
class Integrator:
    asset_model: Model
    weights: tuple[float, ...]
    angles: tuple[float, ...]
    lower_angle: float
    price_offset: float
    price_scale: float


@dataclass(frozen=True)
class Pricing:
    model: Integrator
    estimate: Estimate
    price: float
    interval: tuple[float, float]
    source_tolerance: float
    state_errors: tuple[tuple[int, float], ...]


def build_model() -> Integrator:
    """Derive the exponential reference distribution and source affine rescaling."""
    asset = fresh_model()
    size = 1 << SELECTOR_BITS
    step = asset.rate / (1 << FRACTION_BITS)
    base_price = asset.scale * math.exp(-(size - 1) * step)
    offset = base_price * math.exp(-step)
    scale = offset * math.expm1(size * step)
    weights = tuple(math.exp(word * step) for word in range(size))
    total = math.fsum(weights)
    angles = tuple(
        2 * math.atan(math.exp(step * (1 << bit) / 2)) for bit in range(SELECTOR_BITS)
    )
    lower_angle = 2 * math.asin(math.sqrt((STRIKE - offset) / scale))
    return Integrator(
        asset,
        tuple(value / total for value in weights),
        angles,
        lower_angle,
        offset,
        scale,
    )


def build_program(model: Integrator, power: int) -> synth.Program:
    """Bind the original five-bit exponential reference and amplification power."""
    if type(power) is not int or not 0 <= power <= MAX_POWER:
        raise ValueError("Use a bounded Grover power in 0..4")
    return synth.synthesize(
        integration,
        classical_inputs={
            "k": power,
            "probabilities": model.asset_model.probabilities,
            "angles": model.angles,
            "lower_angle": model.lower_angle,
        },
        max_qubits=MAX_WIDTH,
        max_operations=100_000,
        memory_limit_bytes=4 * 1024**3,
    )


def prepared_reference(model: Integrator) -> torch.Tensor:
    """Generate the complete logical amplitudes without native gate operations."""
    size = 1 << SELECTOR_BITS
    target = torch.zeros((2, 4, 4, size), dtype=torch.complex128)
    asset = model.asset_model
    low = (STRIKE - model.price_offset) / model.price_scale
    cutoff = asset.threshold * (1 << FRACTION_BITS)
    for word, score in enumerate(asset.selectors):
        x, y = word % 4, word // 4
        weight = math.sqrt(asset.probabilities[x] * asset.probabilities[y])
        if score <= cutoff:
            target[0, x, y, 0] = weight * math.sqrt(1 - low)
            target[1, x, y, 0] = weight * math.sqrt(low)
            continue
        for reference, probability in enumerate(model.weights):
            target[int(reference <= score), x, y, reference] = weight * math.sqrt(
                probability
            )
    return target.flatten()


def reference_circuit(model: Integrator, power: int) -> fq.Circuit:
    """Prepare the independent state and apply its phase-exact reflection."""
    target = prepared_reference(model)
    identity = torch.eye(target.numel(), dtype=target.dtype)
    delta = identity[:, 0] - target
    preparation = identity - 2 * torch.outer(delta, delta.conj()) / torch.vdot(
        delta, delta
    )
    reflection = 2 * torch.outer(target, target.conj()) - identity
    circuit = fq.Circuit(10, dtype=torch.complex128).any(*range(10), matrix=preparation)
    for _ in range(power):
        circuit.z(0).any(*range(10), matrix=reflection)
    return circuit


def run_pricing() -> Pricing:
    model = build_model()
    rng = torch.Generator().manual_seed(9137)
    verified: dict[int, tuple[float, float]] = {}

    def sample(power: int, shots: int) -> int:
        if power not in verified:
            program = build_program(model, power)
            result = fq.run(program)
            actual = result.statevector().reshape(1024, -1)
            expected = fq.run(reference_circuit(model, power)).statevector().flatten()
            error = float((actual[:, 0] - expected).abs().max())
            if error > 1e-10 or float(actual[:, 1:].abs().square().sum()) > 1e-20:
                raise AssertionError(
                    "The original logical environment or clean work differs"
                )
            probability = program.probabilities(
                result, registers=("indicator",), max_amplitudes=1 << MAX_WIDTH
            ).get((1,), 0.0)
            if abs(probability - float(expected[512:].abs().square().sum())) > 1e-10:
                raise AssertionError("The indicator marginal differs")
            verified[power] = probability, error
        probability, _ = verified[power]
        return int(
            torch.binomial(
                torch.tensor(float(shots), dtype=torch.float64),
                torch.tensor(probability, dtype=torch.float64),
                generator=rng,
            )
        )

    measured = estimate(sample, epsilon=EPSILON, alpha=ALPHA, max_k=MAX_POWER)
    lower, upper = measured.confidence_interval
    offset = model.price_offset - STRIKE
    interval = lower * model.price_scale + offset, upper * model.price_scale + offset
    price = measured.estimate * model.price_scale + offset
    tolerance = (
        0.5 * (interval[1] - interval[0]) * math.sqrt(math.log(ALPHA / ASSERTION_ALPHA))
    )
    if abs(price - SOURCE_PRICE) > tolerance:
        raise AssertionError("The observed estimate exceeds the source confidence rule")
    return Pricing(
        model,
        measured,
        price,
        interval,
        tolerance,
        tuple((power, row[1]) for power, row in verified.items()),
    )
