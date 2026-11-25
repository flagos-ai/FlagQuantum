"""Measure exact CX/U3 resources and preserve the source HHL readout conventions."""

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

from .ansatz import Route
from .physics import MATRIX, QPE_SIZE, B
from .reference import reference_circuit


@dataclass(frozen=True)
class StateCheck:
    maximum_state_error: float
    converted_state_error: float
    dirty_workspace_probability: float
    resources: ResourceEstimate


@dataclass(frozen=True)
class Solution:
    quantum: torch.Tensor
    classical: torch.Tensor
    corrected: torch.Tensor
    global_phase: float
    relative_distance: float
    fidelity: float
    complex_fidelity: float
    success: float
    retained_phase_leakage: float


def declared_amplitudes(program: synth.Program, state: torch.Tensor) -> torch.Tensor:
    width = state.numel().bit_length() - 1
    logical = tuple(w for wires in program.registers.values() for w in wires)
    hidden = tuple(w for w in range(width) if w not in logical)
    return (
        state.reshape((2,) * width)
        .permute((*logical, *hidden))
        .reshape(1 << len(logical), -1)
    )


def verify_program(program: synth.Program, route: Route) -> StateCheck:
    actual = fq.run(program).statevector()
    assert isinstance(actual, torch.Tensor)
    ordered = declared_amplitudes(program, actual)
    reference = fq.run(reference_circuit(route)).statevector()
    assert isinstance(reference, torch.Tensor)
    error = float((ordered[:, 0] - reference.flatten()).abs().max())
    dirty = float(ordered[:, 1:].abs().square().sum())
    executable = convert_basis(
        program.circuit,
        gates=("cx", "u3"),
        max_added_operations=100000,
        rules=exact_rules(),
    ).program
    converted = fq.run(executable).statevector()
    assert isinstance(converted, torch.Tensor)
    converted_error = float(
        (declared_amplitudes(program, converted)[:, 0] - reference.flatten())
        .abs()
        .max()
    )
    resources = estimate_resources(executable, gates=("cx", "u3"))
    if error > 2e-11 or converted_error > 2e-11 or dirty > 1e-22:
        raise AssertionError(
            "native or converted state differs in absolute phase from independent Circuit"
        )
    if set(resources.operation_counts) - {"cx", "u3"}:
        raise AssertionError("reported target resources must contain only CX and U3")
    return StateCheck(error, converted_error, dirty, resources)


def extract_solution(program: synth.Program, *, check: bool = True) -> Solution:
    if tuple(program.registers) != ("res", "estimator_var", "indicator"):
        raise ValueError("use the original res, estimator_var and indicator ports")
    state = fq.run(program).statevector()
    assert isinstance(state, torch.Tensor)
    amplitudes = declared_amplitudes(program, state)
    if float(amplitudes[:, 1:].abs().square().sum()) > 1e-22:
        raise AssertionError("hidden compiler work must be zero")
    tensor = amplitudes[:, 0].reshape(4, 1 << QPE_SIZE, 2)
    selected = tensor[:, 0, 1]
    q = (
        torch.where(selected.abs() > 1e-12, selected, torch.zeros_like(selected))
        * 2**QPE_SIZE
    )
    quantum = torch.complex(
        torch.round(q.real, decimals=5), torch.round(q.imag, decimals=5)
    )
    if float(quantum.norm()) == 0:
        raise ValueError("postselected solution has zero norm")
    classical = torch.linalg.solve(MATRIX, B)
    phase = torch.angle(
        torch.vdot(MATRIX.to(torch.complex128) @ quantum, B.to(torch.complex128))
    )
    corrected = (torch.exp(1j * phase) * quantum).real
    distance = float((classical - corrected).norm() / classical.norm())
    fidelity = float(
        torch.dot(classical / classical.norm(), corrected / corrected.norm()).square()
    )
    complex_fidelity = float(
        torch.vdot(
            classical.to(torch.complex128) / classical.norm(), quantum / quantum.norm()
        )
        .abs()
        .square()
    )
    if check and distance > 0.1:
        raise AssertionError(
            "source HHL solution exceeds ten-percent relative distance"
        )
    return Solution(
        quantum,
        classical,
        corrected,
        float(phase),
        distance,
        fidelity,
        complex_fidelity,
        float(selected.abs().square().sum()),
        float(tensor[:, 1:, 1].abs().square().sum()),
    )


def plot_solution(solution: Solution, route: Route, directory: Path) -> Path:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(solution.classical.tolist(), "bo", label="Classical")
    axis.plot(solution.corrected.tolist(), "ro", label="HHL (source real projection)")
    axis.set_xlabel("Component i")
    axis.set_ylabel("Solution component x_i")
    axis.set_title(route + " evolution")
    axis.legend()
    figure.tight_layout()
    path = directory / ("hhl_" + route + ".png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
