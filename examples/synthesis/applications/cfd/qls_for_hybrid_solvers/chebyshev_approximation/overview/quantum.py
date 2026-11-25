"""Prepare the full inverse LCU weights using checked native amplitude loading."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.synth import (
    H,
    Output,
    QArray,
    QBit,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from .model import DTYPE, PREPARATION_ERROR

WIDTH = 7
SIZE = 1 << WIDTH


@qfunc
def load_distribution(
    amplitudes: list[float], uniform_bits: int, block: Output[QArray[QBit]]
) -> None:
    allocate(WIDTH, block)
    inplace_prepare_amplitudes(amplitudes, 0.0, block[uniform_bits:WIDTH])
    for bit in range(uniform_bits):
        H(block[bit])


@dataclass(frozen=True)
class Preparation:
    bits_grouped: int
    resources: ResourceEstimate
    coefficients: torch.Tensor
    amplitude_error: float
    probability_error: float


def approximation(
    probabilities: torch.Tensor, tolerance: float
) -> tuple[int, torch.Tensor]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("The preparation tolerance must be finite and nonnegative")
    for bits in range(WIDTH - 1, 0, -1):
        size = 1 << bits
        reduced = probabilities.reshape(-1, size).sum(dim=1)
        expanded = reduced.repeat_interleave(size) / size
        if float((expanded.sqrt() - probabilities.sqrt()).norm()) <= tolerance:
            return bits, reduced
    return 0, probabilities


def compile_distribution(
    probabilities: torch.Tensor, tolerance: float
) -> tuple[synth.Program, int]:
    if probabilities.shape != (SIZE,) or bool((probabilities < 0).any()):
        raise ValueError("Provide the 128 nonnegative LCU weights")
    if not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("The LCU probabilities must sum to one")
    bits, reduced = approximation(probabilities, tolerance)
    program = synth.synthesize(
        load_distribution,
        classical_inputs={
            "amplitudes": list(map(float, reduced.sqrt())),
            "uniform_bits": bits,
        },
        max_qubits=WIDTH,
        max_workspace=0,
        max_operations=512,
        memory_limit_bytes=4 * 1024**2,
    )
    return program, bits


def prepare(coefficients: torch.Tensor, tolerance: float) -> Preparation:
    normalization = float(coefficients.abs().sum())
    probabilities = coefficients.abs() / normalization
    program, grouped = compile_distribution(probabilities, tolerance)
    state = fq.run(program).statevector().flatten()
    error = float((state - probabilities.sqrt()).norm())
    actual = state.abs().square()
    if error > tolerance + 1e-12:
        raise ValueError("Executed amplitude error exceeds the preparation tolerance")
    if not bool(torch.isfinite(actual).all()):
        raise ValueError("Expected a finite LCU distribution")
    return Preparation(
        grouped,
        estimate_resources(program.circuit),
        coefficients.sign() * actual.to(DTYPE) * normalization,
        error,
        float((actual - probabilities).norm()),
    )


def compare(coefficients: torch.Tensor) -> tuple[Preparation, Preparation]:
    exact = prepare(coefficients, 0.0)
    reduced = prepare(coefficients, PREPARATION_ERROR)
    if (
        reduced.resources.operation_counts["ry"] * 2
        >= exact.resources.operation_counts["ry"]
    ):
        raise ValueError(
            "The bounded approximate program must use less than half the RY gates"
        )
    return exact, reduced
