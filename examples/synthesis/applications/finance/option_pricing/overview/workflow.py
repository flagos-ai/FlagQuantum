"""Count-only adaptive estimation with fresh native circuits and independent checks."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    Estimate,
    estimate,
)
from flagquantum import synth

from .model import OptionModel, build_program, fresh_model
from .reference import dense_reference, reference_circuit


@dataclass(frozen=True)
class CircuitCheck:
    k: int
    state_error: float
    named_error: float
    probability: float


@dataclass(frozen=True)
class PricingResult:
    model: OptionModel
    estimate: Estimate
    price: float
    interval: tuple[float, float]
    source_tolerance: float
    checks: tuple[CircuitCheck, ...]
    programs: tuple[synth.Program, ...]


class IndicatorSampler:
    """Compile/execute each requested structural k once, then draw fresh counts.

    This is a bounded cached family, not runtime parameter binding or a hardware
    timing model. Only sampled counts reach the estimator; exact payoff is validation-only.
    """

    def __init__(
        self, model: OptionModel, *, seed: int = 6100, max_width: int = 20
    ) -> None:
        if (
            type(seed) is not int
            or not 0 <= seed < 2**63
            or type(max_width) is not int
            or max_width not in (20, 28)
        ):
            raise ValueError(
                "use a nonnegative63bit seed and source width budget20or28"
            )
        self.model = model
        self.max_width = max_width
        self.generator = torch.Generator().manual_seed(seed)
        self.programs: dict[int, synth.Program] = {}
        self.checks: dict[int, CircuitCheck] = {}

    def ensure(self, k: int) -> synth.Program:
        if type(k) is not int or not 0 <= k <= 16:
            raise ValueError("Grover power must be a bounded integer in0..16")
        if k not in self.programs:
            program = build_program(self.model, k, self.max_width)
            result = fq.run(program)
            actual = result.statevector().flatten()
            circuit = reference_circuit(self.model, k).state()
            assert isinstance(circuit, torch.Tensor)
            expected = circuit.flatten()
            direct = dense_reference(self.model, k)
            error = float(
                max((actual - expected).abs().max(), (expected - direct).abs().max())
            )
            marginal = program.probabilities(result, registers=("indicator",))
            probability = marginal.get((1,), 0.0)
            named_error = abs(probability - float(expected[32:].abs().square().sum()))
            theory = (
                math.sin(
                    (2 * k + 1)
                    * math.asin(
                        math.sqrt(self.model.expected_payoff / self.model.scale)
                    )
                )
                ** 2
            )
            if (
                actual.numel() != 64
                or error > 2e-11
                or named_error > 2e-11
                or abs(probability - theory) > 2e-11
            ):
                raise AssertionError(
                    "full absolute state,bit order,Grover angle or named indicator differs"
                )
            self.programs[k] = program
            self.checks[k] = CircuitCheck(k, error, named_error, probability)
        return self.programs[k]

    def __call__(self, k: int, shots: int) -> int:
        if type(shots) is not int or not 1 <= shots <= 2048:
            raise ValueError("sample batch must be an integer in1..2048")
        self.ensure(k)
        count = torch.binomial(
            torch.tensor(float(shots), dtype=torch.float64),
            torch.tensor(self.checks[k].probability, dtype=torch.float64),
            generator=self.generator,
        )
        return int(count)


def run_pricing(
    *, max_width: int = 20, seed: int = 6100, source_tolerance_factor: int = 5
) -> tuple[PricingResult, IndicatorSampler]:
    if type(source_tolerance_factor) is not int or source_tolerance_factor not in (
        5,
        10,
    ):
        raise ValueError("retain source final tolerance factor5or10")
    model = fresh_model()
    sampler = IndicatorSampler(model, seed=seed, max_width=max_width)
    sampler.ensure(0)
    estimated = estimate(sampler, epsilon=0.05, alpha=0.01)
    price = estimated.estimate * model.scale
    interval = (
        estimated.confidence_interval[0] * model.scale,
        estimated.confidence_interval[1] * model.scale,
    )
    tolerance = source_tolerance_factor * (interval[1] - interval[0])
    if abs(price - model.expected_payoff) > tolerance:
        raise AssertionError(
            "observed pricing estimate exceeds original final tolerance"
        )
    return (
        PricingResult(
            model,
            estimated,
            price,
            interval,
            tolerance,
            tuple(sampler.checks.values()),
            tuple(sampler.programs.values()),
        ),
        sampler,
    )
