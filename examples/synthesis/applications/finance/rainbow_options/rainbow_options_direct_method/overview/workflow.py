"""Sample the direct payoff family and verify all retained model amplitudes."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    Estimate,
    estimate,
)
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.main import (
    main as direct,
)
from flagquantum import synth

from .model import (
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

MAX_POWER = 4


@dataclass(frozen=True)
class Pricing:
    model: Model
    estimate: Estimate
    price: float
    interval: tuple[float, float]
    source_tolerance: float
    state_errors: tuple[tuple[int, float], ...]


def build_program(model: Model, power: int) -> synth.Program:
    if type(power) is not int or not 0 <= power <= MAX_POWER:
        raise ValueError("Use a Grover power in 0..4")
    return synth.synthesize(
        direct,
        classical_inputs=dict(
            k=power,
            probabilities=model.probabilities,
            angles=model.angles,
            lower_angle=model.lower_angle,
        ),
        max_qubits=25,
        max_operations=100_000,
        memory_limit_bytes=512 * 1024**2,
    )


def prepared_reference(model: Model) -> torch.Tensor:
    """Independently evaluate the branch amplitudes from exponential factors."""
    size = 1 << SELECTOR_BITS
    state = torch.zeros((2, 4, 4, size), dtype=torch.complex128)
    step = model.rate / (1 << FRACTION_BITS)
    stay_zero = tuple(math.exp(-step * (1 << bit)) for bit in range(SELECTOR_BITS))
    lower_probability = STRIKE / model.scale
    for word, index in enumerate(model.selectors):
        x, y = word % 4, word // 4
        weight = math.sqrt(model.probabilities[x] * model.probabilities[y])
        if index <= model.threshold * (1 << FRACTION_BITS):
            state[0, x, y, 0] = weight * math.sqrt(1 - lower_probability)
            state[1, x, y, 0] = weight * math.sqrt(lower_probability)
            continue
        for auxiliary in range(size):
            if auxiliary & index:
                continue
            probability = math.prod(
                1 - zero if auxiliary & (1 << bit) else zero
                for bit, zero in enumerate(stay_zero)
                if not index & (1 << bit)
            )
            state[int(auxiliary == 0), x, y, auxiliary] = weight * math.sqrt(
                probability
            )
    return state.flatten()


def reference_circuit(model: Model, power: int) -> fq.Circuit:
    """Apply an independent Householder preparation and exact reflections."""
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
    model = fresh_model()
    generator = torch.Generator().manual_seed(9137)
    verified: dict[int, tuple[float, float]] = {}

    def sample(power: int, shots: int) -> int:
        if power not in verified:
            program = build_program(model, power)
            result = fq.run(program)
            measured = result.statevector().reshape(1024, -1)
            expected = fq.run(reference_circuit(model, power)).statevector().flatten()
            error = float((measured[:, 0] - expected).abs().max())
            probability = program.probabilities(result, registers=("indicator",)).get(
                (1,), 0.0
            )
            if error > 1e-10 or float(measured[:, 1:].abs().square().sum()) > 1e-20:
                raise AssertionError(
                    "The native complete logical state or clean work bits differ"
                )
            if abs(probability - float(expected[512:].abs().square().sum())) > 1e-10:
                raise AssertionError(
                    "The named indicator differs from the independent Circuit"
                )
            verified[power] = probability, error
        probability, _ = verified[power]
        return int(
            torch.binomial(
                torch.tensor(float(shots), dtype=torch.float64),
                torch.tensor(probability, dtype=torch.float64),
                generator=generator,
            )
        )

    estimated = estimate(sample, epsilon=EPSILON, alpha=ALPHA, max_k=MAX_POWER)
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
