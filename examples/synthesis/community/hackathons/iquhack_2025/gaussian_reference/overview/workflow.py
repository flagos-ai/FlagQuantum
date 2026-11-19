"""Reproduce both source overlap formulas and verify their actual cleanup."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler import ResourceEstimate
from flagquantum.synth import Program, synthesize

from . import quantum


@dataclass(frozen=True)
class Amplification:
    width: int
    overlap: float
    actual_overlap: float
    repetitions: int
    workspace_probability: float
    predicted_workspace_probability: float
    state_error: float
    program: Program
    state: torch.Tensor


def grid(width: int) -> torch.Tensor:
    scale = 2 ** (width - 2)
    sign = 1 << (width - 1)
    signed = [
        (word if word < sign else word - 2 * sign) / scale for word in range(2 * sign)
    ]
    return torch.tensor(signed, dtype=torch.float64)


def target(width: int) -> torch.Tensor:
    gaussian = torch.exp(-grid(width).square())
    return gaussian / gaussian.sum()


def exact_grid() -> tuple[Program, float]:
    probabilities = target(6)
    program = synthesize(
        quantum.direct,
        classical_inputs={"probabilities": probabilities.tolist()},
        max_qubits=6,
        max_workspace=0,
    )
    result = fq.run(program).statevector().flatten()
    error = float((result - probabilities.sqrt()).abs().max())
    assert error < 3e-12
    return program, error


def overlaps(width: int) -> tuple[float, float]:
    """The archived endpoint and the lookup-table endpoint differ."""
    archived_grid = torch.linspace(-2, 1, 2**width, dtype=torch.float64)
    source_overlap = float(torch.exp(-archived_grid.square()).mean().sqrt())
    lookup_overlap = float(torch.exp(-grid(width).square()).mean().sqrt())
    return source_overlap, lookup_overlap


def run_amplification(width: int, *, preserve_source_overlap: bool) -> Amplification:
    if width not in (4, 8):
        raise ValueError("the source amplification stages use four and eight bits")
    archived, actual = overlaps(width)
    overlap = archived if preserve_source_overlap else actual
    repetitions = math.ceil(math.pi / (4 * math.asin(overlap)) - 0.5)
    theta = math.pi / (4 * repetitions + 2)
    beta = 2 * math.acos(math.sin(theta) / overlap)
    amplitudes = torch.exp(-grid(width).square() / 2)
    angles = (2 * torch.asin(amplitudes)).tolist()
    program = synthesize(
        quantum.amplify,
        classical_inputs={
            "width": width,
            "repetitions": repetitions,
            "angles": angles,
            "beta": beta,
        },
        max_qubits=18 if width == 8 else 127,
        max_operations=100_000,
        memory_limit_bytes=64 * 1024**2,
    )
    state = fq.run(program).statevector().reshape(2**width, 8, -1)
    assert float(state[:, :, 1:].abs().square().sum()) < 1e-22
    assert float(state[:, 1::2, :].abs().square().sum()) < 1e-22
    workspace = float(state[:, 1:, :].abs().square().sum())
    # Independent two-dimensional Grover rotation with the actual loaded ratio.
    rotated = (2 * repetitions + 1) * math.asin(actual * math.cos(beta / 2))
    expected_success = math.sin(rotated)
    expected_dirty = math.cos(rotated) ** 2
    assert abs(workspace - expected_dirty) < 2e-13
    error = float(
        (state[:, 0, 0] - expected_success * target(width).sqrt()).abs().max()
    )
    assert error < 3e-12
    if preserve_source_overlap:
        assert workspace > 1e-4
    else:
        assert workspace < 1e-22
    return Amplification(
        width,
        overlap,
        actual,
        repetitions,
        workspace,
        expected_dirty,
        error,
        program,
        state[:, 0, 0],
    )


def resource_row(program: Program) -> tuple[int, int, int]:
    cost = program.report["resources"]
    assert isinstance(cost, ResourceEstimate)
    return cost.n_qubits, cost.n_operations, cost.depth
