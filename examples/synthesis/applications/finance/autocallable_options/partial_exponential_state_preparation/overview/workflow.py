"""Fresh native programs and full physical-state validation of source hints."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from .model import IntervalCase
from .quantum import amplify, prepare_full
from .reference import direct_state, reference_circuit


@dataclass(frozen=True)
class PreparationCheck:
    state_error: float
    named_error: float
    workspace_error: float
    workspace_qubits: int
    private_attenuation_one_probability: float | None
    interval_leakage: float
    source_amplitude_error: float


@dataclass(frozen=True)
class PreparationResult:
    program: synth.Program
    probabilities: tuple[float, ...]
    check: PreparationCheck
    resources: ResourceEstimate


def prepare_distribution(amplitudes: tuple[float, ...]) -> PreparationResult:
    program = synth.synthesize(
        prepare_full,
        classical_inputs={"amplitudes": list(amplitudes)},
        max_qubits=8,
        max_workspace=2,
        memory_limit_bytes=1024**2,
    )
    result = fq.run(program)
    physical = result.statevector().reshape(32, -1)
    expected = torch.zeros_like(physical)
    expected[:, 0] = torch.tensor(amplitudes, dtype=torch.complex128)
    error = float((physical - expected).abs().max())
    marginal = program.probabilities(result, registers=("x",))
    probabilities = tuple(marginal.get((x,), 0.0) for x in range(32))
    named_error = max(
        abs(p - a * a) for p, a in zip(probabilities, amplitudes, strict=True)
    )
    if max(error, named_error) > 1e-11:
        raise AssertionError("base exponential state or named x probabilities differ")
    workspace_error = (
        float(physical[:, 1:].abs().max()) if physical.shape[1] > 1 else 0.0
    )
    resource = program.report["resources"]
    assert isinstance(resource, ResourceEstimate)
    return PreparationResult(
        program,
        probabilities,
        PreparationCheck(
            error,
            named_error,
            workspace_error,
            physical.shape[1].bit_length() - 1,
            None,
            0.0,
            0.0,
        ),
        resource,
    )


def prepare_interval(case: IntervalCase) -> PreparationResult:
    lower, upper = case.lower + case.shift, case.upper + case.shift
    if not 0 <= lower <= upper < 32:
        raise ValueError("source interval plot requires a non-wrapping shifted domain")
    program = synth.synthesize(
        amplify,
        classical_inputs={
            "amplitudes": list(case.amplitudes),
            "beta": case.beta,
            "iterations": case.iterations,
            "lower": case.lower,
            "upper": case.upper,
            "shift": case.shift,
        },
        max_qubits=8,
        max_workspace=2,
        memory_limit_bytes=1024**2,
    )
    result = fq.run(program)
    physical = result.statevector().reshape(64, -1)
    direct = direct_state(case)
    reference = reference_circuit(case).state()
    assert isinstance(reference, torch.Tensor)
    expected = torch.zeros_like(physical)
    expected[:, 0] = direct
    error = float(
        max(
            (physical - expected).abs().max(),
            (direct - reference.flatten()).abs().max(),
        )
    )
    workspace_error = (
        float(physical[:, 1:].abs().max()) if physical.shape[1] > 1 else 0.0
    )
    marginal = program.probabilities(result, registers=("x",))
    probabilities = tuple(marginal.get((x,), 0.0) for x in range(32))
    expected_probabilities = direct.abs().square().reshape(32, 2).sum(dim=1)
    named_error = max(
        abs(p - float(expected_probabilities[x])) for x, p in enumerate(probabilities)
    )
    if max(error, workspace_error, named_error) > 1e-11:
        raise AssertionError(
            "full physical state,bit order,clean work or named x differs"
        )
    norm = math.fsum(math.exp(case.rate * x) for x in range(lower, upper + 1))
    amplitude_error = max(
        abs(math.sqrt(probabilities[x]) - math.sqrt(math.exp(case.rate * x) / norm))
        for x in range(lower, upper + 1)
    )
    by_x = physical.reshape(32, 2, -1).abs().square().sum(dim=(1, 2))
    leakage = float(torch.cat((by_x[:lower], by_x[upper + 1 :])).sum())
    private_flag = float(physical.reshape(32, 2, -1)[:, 1, :].abs().square().sum())
    if private_flag > 1e-22 or leakage > 1e-22:
        raise AssertionError(
            "exact overlap must yield clean attenuation and no interval leakage"
        )
    resource = program.report["resources"]
    assert isinstance(resource, ResourceEstimate)
    return PreparationResult(
        program,
        probabilities,
        PreparationCheck(
            error,
            named_error,
            workspace_error,
            physical.shape[1].bit_length() - 1,
            private_flag,
            leakage,
            amplitude_error,
        ),
        resource,
    )
