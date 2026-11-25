"""Independent HHL validation and the original workshop postprocessing."""

from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.hhl_reference import householder, matrix_hhl_circuit
from flagquantum import synth

from .physics import AMPLITUDES, ANGLES, MATRIX, QPE_SIZE, RECIPROCAL_SCALE, UNITARY, B


@dataclass(frozen=True)
class StateCheck:
    maximum_state_error: float
    dirty_workspace_probability: float


@dataclass(frozen=True)
class Solution:
    rounded_quantum: torch.Tensor
    classical: torch.Tensor
    fidelity: float
    postselection_probability: float
    indicator_probability: float
    retained_phase_leakage_probability: float
    magnitude_diagnostic: torch.Tensor
    magnitude_relative_distance_percent: float
    complex_relative_distance: float


def validate_problem(matrix: torch.Tensor, b: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape != (4, 4) or b.shape != (4,):
        raise ValueError(
            "the source problem requires a 4x4 matrix and four-entry vector"
        )
    if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("matrix and vector must be finite")
    if not torch.allclose(matrix, matrix.mT, rtol=1e-6, atol=1e-6):
        raise ValueError("the matrix must be symmetric")
    eigenvalues = torch.linalg.eigvalsh(matrix)
    if bool((eigenvalues <= 0).any()) or bool((eigenvalues >= 1).any()):
        raise ValueError(
            "HHL matrix eigenvalues must lie strictly between zero and one"
        )
    if not bool(
        torch.isclose(
            torch.linalg.vector_norm(b),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError("the source vector must be normalized")


def declared_amplitudes(program: synth.Program) -> torch.Tensor:
    state = fq.run(program).statevector()
    assert isinstance(state, torch.Tensor)
    n = state.numel().bit_length() - 1
    logical = tuple(w for wires in program.registers.values() for w in wires)
    hidden = tuple(w for w in range(n) if w not in logical)
    return (
        state.reshape((2,) * n)
        .permute((*logical, *hidden))
        .reshape(1 << len(logical), -1)
    )


def verify_program(
    program: synth.Program, *, preparation_only: bool = False
) -> StateCheck:
    actual = declared_amplitudes(program)
    reference = (
        fq.Circuit(2, dtype=torch.complex128).any(0, 1, matrix=householder(AMPLITUDES))
        if preparation_only
        else matrix_hhl_circuit(7, (0, 1), (2, 3, 4, 5), 6, UNITARY, ANGLES, AMPLITUDES)
    )
    expected = fq.run(reference).statevector()
    assert isinstance(expected, torch.Tensor)
    error = float((actual[:, 0] - expected.flatten()).abs().max())
    dirty = float(actual[:, 1:].abs().square().sum())
    if error > 2e-11 or dirty > 1e-22:
        raise AssertionError(
            "absolute native state or compiler work differs from independent Circuit"
        )
    return StateCheck(error, dirty)


def extract_solution(program: synth.Program) -> Solution:
    if tuple(program.registers) != ("res", "phase_", "indicator") or tuple(
        len(w) for w in program.registers.values()
    ) != (2, QPE_SIZE, 1):
        raise ValueError("solution readout needs source res,phase_,indicator ports")
    actual = declared_amplitudes(program)
    if float(actual[:, 1:].abs().square().sum()) > 1e-22:
        raise AssertionError("compiler work must be zero")
    tensor = actual[:, 0].reshape(4, 1 << QPE_SIZE, 2)
    selected = tensor[:, 0, 1]
    masked = torch.where(selected.abs() > 1e-10, selected, torch.zeros_like(selected))
    qsol = masked / RECIPROCAL_SCALE
    rounded = torch.complex(
        torch.round(qsol.real, decimals=5), torch.round(qsol.imag, decimals=5)
    )
    if float(torch.linalg.vector_norm(rounded)) == 0:
        raise ValueError("the postselected solution has zero norm")
    classical = torch.linalg.solve(MATRIX, B)
    normalized = classical.to(torch.complex128) / torch.linalg.vector_norm(classical)
    fidelity = float(
        torch.vdot(normalized, rounded / torch.linalg.vector_norm(rounded))
        .abs()
        .square()
    )
    # The source removes a different phase from every entry. This is a
    # magnitude-only diagnostic, not a physical global-phase correction.
    magnitude = torch.real(rounded / torch.exp(1j * torch.angle(rounded)))
    relative = float(
        torch.linalg.vector_norm(classical - magnitude)
        / torch.linalg.vector_norm(classical)
    )
    complex_distance = float(
        torch.linalg.vector_norm(classical - rounded)
        / torch.linalg.vector_norm(classical)
    )
    return Solution(
        rounded,
        classical,
        fidelity,
        float(selected.abs().square().sum()),
        float(tensor[:, :, 1].abs().square().sum()),
        float(tensor[:, 1:, 1].abs().square().sum()),
        magnitude,
        round(100 * relative, 1),
        complex_distance,
    )


def plot_solution(solution: Solution, directory: Path) -> Path:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(solution.classical.tolist(), "bo", label="Classical")
    axis.plot(
        solution.magnitude_diagnostic.tolist(),
        "ro",
        label="HHL (source magnitude diagnostic)",
    )
    axis.legend()
    axis.set_xlabel("Component i")
    axis.set_ylabel("Solution component x_i")
    figure.tight_layout()
    path = directory / "hhl_solution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
