"""Read the actual success branch and report the source normalization discrepancy."""

from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources

from .physics import AMPLITUDES, MAX_EIG, PREFACTOR, SOURCE, finite_difference_problem
from .reference import reference_circuit


@dataclass(frozen=True)
class Verification:
    maximum_state_error: float
    dirty_workspace_probability: float
    named_probability_error: float
    resources: ResourceEstimate


@dataclass(frozen=True)
class Solution:
    quantum: torch.Tensor
    classical: torch.Tensor
    global_phase: float
    relative_distance: float
    transpose_relative_distance: float
    relative_residual: float
    success: float
    retained_phase_leakage: float


def declared_state(program: synth.Program, state: torch.Tensor) -> torch.Tensor:
    width = state.numel().bit_length() - 1
    visible = tuple(w for wires in program.registers.values() for w in wires)
    work = tuple(w for w in range(width) if w not in visible)
    return state.reshape((2,) * width).permute((*visible, *work)).reshape(8192, -1)


def verify_program(program: synth.Program) -> Verification:
    result = fq.run(program)
    actual = result.statevector()
    assert isinstance(actual, torch.Tensor)
    ordered = declared_state(program, actual)
    expected = fq.run(reference_circuit(AMPLITUDES)).statevector()
    assert isinstance(expected, torch.Tensor)
    state_error = float((ordered[:, 0] - expected.flatten()).abs().max())
    dirty = float(ordered[:, 1:].abs().square().sum())
    measured = dict(
        program.numeric_probabilities(result, registers=("xy", "phase", "indicator"))
    )
    reference = expected.reshape(64, 64, 2).abs().square()
    probability_error = max(
        abs(
            float(probability)
            - float(reference[int(xy), int(64 * phase), int(indicator)])
        )
        for (xy, phase, indicator), probability in measured.items()
    )
    assert state_error < 2e-11 and dirty < 1e-22 and probability_error < 2e-11
    return Verification(
        state_error, dirty, probability_error, estimate_resources(program.circuit)
    )


def extract_solution(program: synth.Program) -> Solution:
    actual = fq.run(program).statevector()
    assert isinstance(actual, torch.Tensor)
    state = declared_state(program, actual)[:, 0].reshape(64, 64, 2)
    selected = state[:, 0, 1]
    filtered = torch.where(selected.abs() > 1e-12, selected, torch.zeros_like(selected))
    phase = torch.angle(filtered[0])
    matrix = ((filtered * torch.exp(-1j * phase)).real / (PREFACTOR / MAX_EIG)).reshape(
        8, 8
    )
    if matrix[4, 4] < 0:
        matrix = -matrix
    operator, rhs, classical = finite_difference_problem()
    return Solution(
        matrix,
        classical,
        float(phase),
        float((matrix - classical).norm() / classical.norm()),
        float((matrix.T - classical).norm() / classical.norm()),
        float((operator @ matrix.flatten() - rhs).norm() / rhs.norm()),
        float(selected.abs().square().sum()),
        float(state[:, 1:, 1].abs().square().sum()),
    )


def plot_source(directory: Path) -> Path:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    axis_grid = (torch.arange(8, dtype=torch.float64) + 0.5) / 8
    mesh_x, mesh_y = torch.meshgrid(axis_grid, axis_grid, indexing="xy")
    figure, axis = plt.subplots(figsize=(5, 4))
    contour = axis.contourf(
        mesh_x.numpy(), mesh_y.numpy(), SOURCE.reshape(8, 8).T.numpy()
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Normalized source")
    figure.colorbar(contour, ax=axis)
    figure.tight_layout()
    path = directory / "source.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_solution(solution: Solution, directory: Path) -> Path:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    axis_grid = (torch.arange(8, dtype=torch.float64) + 0.5) / 8
    mesh_x, mesh_y = torch.meshgrid(axis_grid, axis_grid, indexing="xy")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, matrix, title in zip(
        axes,
        (solution.quantum, solution.classical),
        ("Quantum (source scale)", "Classical"),
        strict=True,
    ):
        contour = axis.contourf(mesh_x.numpy(), mesh_y.numpy(), matrix.T.numpy())
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(title)
        figure.colorbar(contour, ax=axis)
    figure.tight_layout()
    path = directory / "solution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
