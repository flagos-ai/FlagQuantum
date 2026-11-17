"""Validate every conditional thermal branch and the complete mock QAE layer."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    response,
)
from flagquantum.compiler import ResourceEstimate, estimate_resources

from . import model, quantum

BINDINGS = ((0.67, 0.31), (-0.41, 0.23))


@dataclass(frozen=True)
class Result:
    matrix: torch.Tensor
    reference: torch.Tensor
    computed: torch.Tensor
    reciprocal: model.Reciprocal
    demo_reciprocal: model.Reciprocal
    matrix_error: float
    conditional_error: float
    inverse_error: float
    relative_error: float
    inverse_resources: ResourceEstimate
    mock_resources: ResourceEstimate
    conditional_resources: ResourceEstimate
    costs: torch.Tensor
    mock_probabilities: torch.Tensor
    mock_errors: tuple[float, ...]
    configurations: torch.Tensor


def verify_conditional(heat: torch.Tensor) -> tuple[float, ResourceEstimate]:
    circuit = quantum.conditional_thermal_encoding()
    dimension = 1 << circuit.n_qubits
    initial = torch.zeros(model.CONFIGURATIONS, dimension, dtype=model.COMPLEX)
    for word in range(model.CONFIGURATIONS):
        initial[word, word * 128 : word * 128 + 4] = heat
    execution = fq.Circuit.from_ir(
        circuit.to_ir(), dtype=model.COMPLEX, inputs=initial, bsz=model.CONFIGURATIONS
    )
    states: torch.Tensor = execution.state().reshape(
        model.CONFIGURATIONS, model.CONFIGURATIONS, 128
    )
    values = torch.arange(model.CONFIGURATIONS)
    diagonal = states[values, values]
    reference = torch.stack(
        [
            model.build_matrix(word).to(model.COMPLEX) @ heat / model.alpha()
            for word in range(model.CONFIGURATIONS)
        ]
    )
    maximum = float((diagonal[:, :4] - reference).abs().max())
    leakage = float((states.abs().square().sum() - diagonal.abs().square().sum()).abs())
    if maximum > 1e-10 or leakage > 1e-10:
        raise ValueError(
            "The coherent design register or its conditioned thermal projection changed"
        )
    return maximum, estimate_resources(circuit)


def qae_reference(cost: float) -> torch.Tensor:
    """One explicit controlled Grover step and inverse one-bit Fourier transform."""
    angle = math.asin(cost)
    b = math.cos(angle)
    initial = torch.tensor((b, cost), dtype=model.REAL)
    c, s = math.cos(2 * angle), math.sin(2 * angle)
    rotate = torch.tensor(((c, -s), (s, c)), dtype=model.REAL)
    rotated = rotate @ initial
    vector = torch.cat((initial + rotated, initial - rotated)) / 2
    normalized: torch.Tensor = vector / vector.norm()
    return normalized.to(model.COMPLEX)


def check_qae(costs: torch.Tensor) -> None:
    for value in costs:
        cost = float(value)
        actual = quantum.qae_state(cost)
        expected = qae_reference(cost)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


def preparation_reference(amplitudes: torch.Tensor) -> torch.Tensor:
    magnitudes = amplitudes.abs().reshape(2, 2)
    first, second = (float(row.norm()) for row in magnitudes)
    start = torch.tensor(((first, -second), (second, first)), dtype=model.COMPLEX)
    controlled = torch.zeros(4, 4, dtype=model.COMPLEX)
    for branch, row in enumerate(magnitudes):
        norm = float(row.norm())
        c, s = (float(row[0]) / norm, float(row[1]) / norm) if norm else (1.0, 0.0)
        part = slice(2 * branch, 2 * (branch + 1))
        controlled[part, part] = torch.tensor(((c, -s), (s, c)), dtype=model.COMPLEX)
    result: torch.Tensor = (1j * amplitudes.angle()).exp()[:, None] * (
        controlled @ torch.kron(start, torch.eye(2, dtype=model.COMPLEX))
    )
    return result


def mock_state(costs: torch.Tensor, gamma: float, beta: float) -> torch.Tensor:
    states = torch.empty(model.CONFIGURATIONS, 4, dtype=model.COMPLEX)
    diagonal = torch.ones(4, dtype=model.COMPLEX)
    diagonal[2:] = torch.exp(
        torch.tensor(1j * quantum.QAE_PHASE_WEIGHT * gamma, dtype=model.COMPLEX)
    )
    for index, value in enumerate(costs):
        vector = quantum.qae_state(float(value))
        states[index] = preparation_reference(vector).mH @ (diagonal * vector) / 8
    c, s = math.cos(beta), 1j * math.sin(beta)
    rotation = torch.tensor(((c, s), (s, c)), dtype=model.COMPLEX)
    mixer = rotation
    for _ in range(1, len(model.EDGES)):
        mixer = torch.kron(mixer, rotation)
    mixed: torch.Tensor = mixer @ states
    return mixed.flatten()


def run() -> Result:
    matrix = model.build_matrix()
    rhs = torch.tensor(model.HEAT, dtype=model.REAL)
    signal = quantum.unitary(quantum.thermal_encoding())
    matrix_error = float((signal[:4, :4] * model.alpha() - matrix).abs().max())
    if matrix_error > 1e-10:
        raise ValueError("The encoded block and independent thermal matrix disagree")
    conditional_error, conditional_resources = verify_conditional(
        (rhs / rhs.norm()).to(model.COMPLEX)
    )
    reciprocal = model.inverse_target(1 / (model.alpha() * model.R_ENV), 11)
    demo = model.inverse_target(1 / 3, 11, target_scale=0.3)
    grid = torch.linspace(1 / 3, 1, 1025, dtype=model.REAL)
    if float((response(grid, demo.fit.phases) - 0.3 / grid).abs().max()) >= 0.1:
        raise ValueError(
            "The original demonstration reciprocal interval was not reached"
        )
    circuit = quantum.inverse_circuit(reciprocal)
    projected: torch.Tensor = -circuit.state().flatten()[:4]
    independent = model.spectral_solution(matrix, reciprocal, rhs / rhs.norm())
    inverse_error = float((projected - independent).abs().max())
    computed = projected.real * rhs.norm() / (model.alpha() * reciprocal.scale)
    configurations = model.physical_solutions()
    reference = configurations[-1]
    relative_error = float((computed - reference).norm() / reference.norm())
    if inverse_error > 1e-10 or relative_error >= 0.04:
        raise ValueError("The original-sized degree-11 thermal solution failed")

    costs = model.load_demo_costs()
    check_qae(costs)
    template = quantum.mock_qaoa(costs)
    rows = []
    probabilities = []
    for gamma, beta in BINDINGS:
        state: torch.Tensor = (
            template.bind_parameters({"gamma": gamma, "beta": beta}).state().flatten()
        )
        expected = mock_state(costs, gamma, beta)
        error = float((state - expected).abs().max())
        if error > 1e-10:
            raise ValueError("The complete mocked thermal QAOA state disagrees")
        rows.append(error)
        probabilities.append(
            state.reshape(model.CONFIGURATIONS, 4).abs().square().sum(dim=1)
        )
    if template.parameter_names != ("beta", "gamma"):
        raise ValueError("The model must remain symbolic across both executions")
    return Result(
        matrix,
        reference,
        computed,
        reciprocal,
        demo,
        matrix_error,
        conditional_error,
        inverse_error,
        relative_error,
        estimate_resources(circuit),
        estimate_resources(template),
        conditional_resources,
        costs,
        torch.stack(probabilities),
        tuple(rows),
        configurations,
    )
