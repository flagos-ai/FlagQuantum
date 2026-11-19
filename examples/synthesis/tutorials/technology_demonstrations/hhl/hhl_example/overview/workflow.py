"""Measure every original precision in exact CX/U3 and retain physical readout."""

from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.exact_basis import (
    exact_rules,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis

from .physics import (
    CLASSICAL,
    HISTORICAL_CX,
    HISTORICAL_DEPTHS,
    HISTORICAL_FIDELITIES,
    HISTORICAL_WIDTHS,
    PRECISIONS,
)
from .reference import reference_circuit


@dataclass(frozen=True)
class StudyPoint:
    precision: int
    resources: ResourceEstimate
    maximum_state_error: float
    converted_state_error: float
    dirty_workspace_probability: float
    fidelity: float
    success_probability: float
    retained_phase_leakage: float
    quantum_solution: tuple[complex, ...]


def verify_program(program: synth.Program, precision: int) -> StudyPoint:
    result = fq.run(program)
    actual = result.statevector()
    expected = fq.run(reference_circuit(precision)).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    width = precision + 4
    visible = tuple(w for wires in program.registers.values() for w in wires)
    work = tuple(w for w in range(width) if w not in visible)
    ordered = (
        actual.reshape((2,) * width).permute((*visible, *work)).reshape(1 << width, -1)
    )
    error = float((ordered[:, 0] - expected.flatten()).abs().max())
    dirty = float(ordered[:, 1:].abs().square().sum())
    converted = convert_basis(
        program.circuit,
        gates=("cx", "u3"),
        max_added_operations=500000,
        rules=exact_rules(),
    ).program
    converted_state = fq.run(converted).statevector()
    assert isinstance(converted_state, torch.Tensor)
    converted_ordered = (
        converted_state.reshape((2,) * width)
        .permute((*visible, *work))
        .reshape(1 << width, -1)
    )
    converted_error = float((converted_ordered[:, 0] - expected.flatten()).abs().max())
    tensor = ordered[:, 0].reshape(8, 1 << precision, 2)
    selected = tensor[:, 0, 1]
    qsol = torch.where(selected.abs() > 1e-12, selected, torch.zeros_like(selected)) * (
        1 << precision
    )
    fidelity = float(
        torch.dot(CLASSICAL.to(torch.complex128) / CLASSICAL.norm(), qsol / qsol.norm())
        .abs()
        .square()
    )
    resources = estimate_resources(converted, gates=("cx", "u3"))
    assert error < 2e-11 and converted_error < 2e-10 and dirty < 1e-22
    assert resources.n_qubits == width and set(resources.operation_counts) == {
        "cx",
        "u3",
    }
    return StudyPoint(
        precision,
        resources,
        error,
        converted_error,
        dirty,
        fidelity,
        float(selected.abs().square().sum()),
        float(tensor[:, 1:, 1].abs().square().sum()),
        tuple(complex(v) for v in qsol),
    )


def plot_study(points: tuple[StudyPoint, ...], directory: Path) -> Path:
    import matplotlib.pyplot as plt

    if tuple(point.precision for point in points) != PRECISIONS:
        raise ValueError("retain every source precision from two through eight")
    directory.mkdir(parents=True, exist_ok=True)
    fidelities = [point.fidelity for point in points]
    depths = [point.resources.depth for point in points]
    cx = [point.resources.operation_counts["cx"] for point in points]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.semilogy(fidelities, depths, "-o", label="Native CX/U3 depth (measured)")
    axis.semilogy(fidelities, cx, "-*", label="Native CX count (measured)")
    axis.semilogy(
        HISTORICAL_FIDELITIES,
        HISTORICAL_DEPTHS,
        "--s",
        label="Qiskit depth (source historical data)",
    )
    axis.semilogy(
        HISTORICAL_FIDELITIES,
        HISTORICAL_CX,
        "--v",
        label="Qiskit CX count (source historical data)",
    )
    for fidelity, depth, point in zip(fidelities, depths, points, strict=True):
        axis.annotate(
            str(point.resources.n_qubits),
            (fidelity, depth),
            xytext=(-8, 7),
            ha="right",
            textcoords="offset points",
            fontsize=8,
        )
    for fidelity, depth, width in zip(
        HISTORICAL_FIDELITIES, HISTORICAL_DEPTHS, HISTORICAL_WIDTHS, strict=True
    ):
        axis.annotate(
            str(width),
            (fidelity, depth),
            xytext=(8, -12),
            ha="left",
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlim(0.28, 1.05)
    axis.set_xlabel("Squared overlap with normalized classical solution")
    axis.set_ylabel("Depth and CX count")
    axis.legend(fontsize=9)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = directory / "fidelity_resources.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
