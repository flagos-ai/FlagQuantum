"""Bounded sample-only CDF estimates inside the unchanged source search."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.quantum_counting.overview.iterative import (
    Estimate,
    estimate,
)
from flagquantum import synth

from .model import LossModel, SearchResult, fresh_model, search
from .quantum import main
from .reference import dense_reference, reference_circuit


@dataclass(frozen=True)
class CircuitCheck:
    threshold: int
    k: int
    state_error: float
    named_error: float
    probability: float
    workspace_qubits: int
    workspace_error: float


class ThresholdSampler:
    """Cache bounded structural programs; supply fresh counts only to estimation."""

    def __init__(self, model: LossModel, *, seed: int = 6200) -> None:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative63bit integer")
        self.model = model
        self.generator = torch.Generator().manual_seed(seed)
        self.programs: dict[tuple[int, int], synth.Program] = {}
        self.checks: dict[tuple[int, int], CircuitCheck] = {}
        self.estimates: list[tuple[int, Estimate]] = []

    def ensure(self, threshold: int, k: int) -> synth.Program:
        if type(threshold) is not int or not 0 <= threshold <= 128:
            raise ValueError("threshold must be an integer in0..128")
        if type(k) is not int or not 0 <= k <= 16:
            raise ValueError("Grover power must be an integer in0..16")
        key = threshold, k
        if key not in self.programs:
            if len(self.programs) >= 102:
                raise ValueError("source search exceeded its six-query program budget")
            program = synth.synthesize(
                main,
                classical_inputs={
                    "probabilities": list(self.model.probabilities),
                    "threshold": threshold,
                    "k": k,
                },
                max_qubits=28,
                max_operations=100_000,
                max_workspace=20,
                memory_limit_bytes=64 * 1024**2,
            )
            result = fq.run(program)
            physical = result.statevector().reshape(256, -1)
            actual = physical[:, 0]
            workspace_qubits = physical.shape[1].bit_length() - 1
            workspace_error = (
                float(physical[:, 1:].abs().max()) if physical.shape[1] > 1 else 0.0
            )
            circuit = reference_circuit(self.model, threshold, k).state()
            assert isinstance(circuit, torch.Tensor)
            expected = circuit.flatten()
            direct = dense_reference(self.model, threshold, k)
            physical_expected = torch.zeros_like(physical)
            physical_expected[:, 0] = expected
            error = float(
                max(
                    (physical - physical_expected).abs().max(),
                    (expected - direct).abs().max(),
                )
            )
            probability = program.probabilities(result, registers=("indicator",)).get(
                (1,), 0.0
            )
            named_error = abs(probability - float(expected[128:].abs().square().sum()))
            theory = (
                math.sin(
                    (2 * k + 1) * math.asin(math.sqrt(self.model.alpha(threshold)))
                )
                ** 2
            )
            if (
                actual.numel() != 256
                or max(error, named_error, workspace_error, abs(probability - theory))
                > 3e-11
            ):
                raise AssertionError(
                    "absolute state, bit order or indicator differs from independent reference"
                )
            self.programs[key] = program
            self.checks[key] = CircuitCheck(
                threshold,
                k,
                error,
                named_error,
                probability,
                workspace_qubits,
                workspace_error,
            )
        return self.programs[key]

    def sample(self, threshold: int, k: int, shots: int) -> int:
        if type(shots) is not int or not 1 <= shots <= 2048:
            raise ValueError("sample batch must be an integer in1..2048")
        self.ensure(threshold, k)
        return int(
            torch.binomial(
                torch.tensor(float(shots), dtype=torch.float64),
                torch.tensor(
                    self.checks[threshold, k].probability, dtype=torch.float64
                ),
                generator=self.generator,
            )
        )

    def alpha_at(self, threshold: int) -> float:
        if len(self.estimates) >= 6:
            raise ValueError("source bisection admits at most six CDF queries")
        result = estimate(
            lambda k, shots: self.sample(threshold, k, shots), epsilon=0.05, alpha=0.01
        )
        self.estimates.append((threshold, result))
        return result.estimate


def run_risk(
    seed: int = 6200,
) -> tuple[LossModel, SearchResult, SearchResult, ThresholdSampler]:
    model = fresh_model()
    classical = search(model, model.alpha)
    sampler = ThresholdSampler(model, seed=seed)
    measured = search(model, sampler.alpha_at)
    return model, classical, measured, sampler
