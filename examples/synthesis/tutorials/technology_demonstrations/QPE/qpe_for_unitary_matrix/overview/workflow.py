"""Verify every QPE precision and measure the actual CX/U3 circuit costs."""

import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.exact_basis import (
    exact_rules,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis

from .physics import HISTORICAL_QISKIT_CX, HISTORICAL_QISKIT_DEPTHS, PRECISIONS, UNITARY


@dataclass(frozen=True)
class StudyPoint:
    precision: int
    resources: ResourceEstimate
    maximum_state_error: float
    converted_state_error: float
    named_probability_error: float
    retained_environment_probability: float
    elapsed_seconds: float


def reference_circuit(precision: int) -> fq.Circuit:
    """Build controlled dense matrix powers and inverse Fourier independently."""
    if type(precision) is not int or not 1 <= precision <= 8:
        raise ValueError("use an integer precision in 1..8")
    data = tuple(range(precision, precision + 3))
    circuit = fq.Circuit(precision + 3, dtype=torch.complex128)
    matrix = torch.tensor(UNITARY, dtype=torch.complex128)
    for bit in range(precision):
        controlled = torch.eye(16, dtype=torch.complex128)
        controlled[8:, 8:] = torch.linalg.matrix_power(matrix, 1 << bit)
        wire = precision - 1 - bit
        circuit.h(wire).any(wire, *data, matrix=controlled)
    dimension = 1 << precision
    axis = torch.arange(dimension, dtype=torch.float64)
    fourier = torch.exp(
        -2j * math.pi * torch.outer(axis, axis) / dimension
    ) / math.sqrt(dimension)
    return circuit.any(*range(precision), matrix=fourier)


def verify_program(program: synth.Program, precision: int) -> StudyPoint:
    """Include all retained data amplitudes; never treat drop as reset."""
    started = time.perf_counter()
    if (
        tuple(program.registers) != ("phase",)
        or len(program.registers["phase"]) != precision
    ):
        raise ValueError("use the matching source fractional phase register")
    result = fq.run(program)
    actual = result.statevector()
    expected = fq.run(reference_circuit(precision)).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    error = float((actual.flatten() - expected.flatten()).abs().max())
    executable = convert_basis(
        program.circuit,
        gates=("cx", "u3"),
        max_added_operations=500000,
        rules=exact_rules(),
    ).program
    resources = estimate_resources(executable, gates=("cx", "u3"))
    converted = fq.run(executable).statevector()
    assert isinstance(converted, torch.Tensor)
    converted_error = float((converted.flatten() - expected.flatten()).abs().max())
    # All three private data bits remain part of the actual quantum state.
    probability = expected.reshape(1 << precision, 8).abs().square().sum(dim=1)
    named = program.numeric_probabilities(result, registers=("phase",))
    named_error = max(
        abs(
            float(probability[word]) - named.get((Fraction(word, 1 << precision),), 0.0)
        )
        for word in range(1 << precision)
    )
    retained = float(actual.reshape(1 << precision, 8)[:, 1:].abs().square().sum())
    if error > 2e-11 or converted_error > 2e-10 or named_error > 2e-11:
        raise AssertionError(
            "QPE absolute state or named marginal differs from independent Circuit"
        )
    if resources.n_qubits != precision + 3 or set(resources.operation_counts) != {
        "cx",
        "u3",
    }:
        raise AssertionError(
            "target cost must retain original width and only CX/U3 operations"
        )
    return StudyPoint(
        precision,
        resources,
        error,
        converted_error,
        named_error,
        retained,
        time.perf_counter() - started,
    )


def plot_study(points: tuple[StudyPoint, ...], directory: Path) -> Path:
    """Distinguish actual native costs from the source's historical Qiskit data."""
    import matplotlib.pyplot as plt

    if tuple(point.precision for point in points) != PRECISIONS:
        raise ValueError("include all original precisions 1..8")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.semilogy(
        PRECISIONS,
        [p.resources.depth for p in points],
        "o-",
        label="Native CX/U3 depth (measured)",
    )
    axis.semilogy(
        PRECISIONS,
        [p.resources.operation_counts["cx"] for p in points],
        "*-",
        label="Native CX count (measured)",
    )
    axis.semilogy(
        PRECISIONS,
        HISTORICAL_QISKIT_DEPTHS,
        "s--",
        label="Qiskit depth (source historical data)",
    )
    axis.semilogy(
        PRECISIONS,
        HISTORICAL_QISKIT_CX,
        "v--",
        label="Qiskit CX count (source historical data)",
    )
    axis.set_xlabel("QPE precision (bits)")
    axis.set_ylabel("Depth and CX count")
    axis.set_xticks(PRECISIONS)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "precision_resources.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
