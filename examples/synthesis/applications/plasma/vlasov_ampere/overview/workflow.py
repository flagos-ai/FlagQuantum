"""Execute the full source inverse and compare its polynomial and classical fields."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.model import (
    evaluate,
)
from examples.synthesis.applications.plasma.vlasov_ampere.vlasov_ampere_qiskit.overview import (
    physics,
)
from examples.synthesis.applications.plasma.vlasov_ampere.vlasov_ampere_qiskit.overview import (
    quantum as encoding,
)
from flagquantum.compiler import ResourceEstimate, estimate_resources

from . import execution, model, quantum, reference


@dataclass(frozen=True)
class Result:
    encoded_matrix: torch.Tensor
    model: model.Model
    state: torch.Tensor
    quantum_field: torch.Tensor
    classical_field: torch.Tensor
    full_state_error: float
    polynomial_error: float
    linear_solution_error: float
    projection_probability: float
    resources: ResourceEstimate


def reconstruct_block() -> torch.Tensor:
    block = encoding.block_encoding()
    matrix = torch.empty((128, 128), dtype=torch.complex128)
    for column in range(128):
        initial = torch.zeros(1 << 17, dtype=torch.complex128)
        initial[column] = 1
        circuit = fq.Circuit(17, dtype=torch.complex128, inputs=initial).compose(block)
        state = circuit.state().reshape(1024, 128)
        matrix[:, column] = state[0] * encoding.NORMALIZATION
        if float(state.reshape(4, 256, 128)[1:].abs().square().sum()) >= 1e-24:
            raise ValueError("Original block reconstruction left nonzero work")
    torch.testing.assert_close(matrix, physics.matrix(), atol=1e-12, rtol=0)
    return matrix


def run(prepared: model.Model | None = None) -> Result:
    encoded = reconstruct_block() if prepared is None else prepared.matrix
    value = model.build(encoded) if prepared is None else prepared
    if len(value.phases) != 252:
        raise ValueError("The original inverse requires all 252 fresh phases")
    circuit = quantum.circuit(value)
    state = execution.execute(circuit)
    independent = reference.full_state(value)
    full_error = float((state - independent).abs().max())
    projected = state.reshape(2, 1024, 128)[0, 0]
    solution = -value.inverse_normalization * value.source_norm * projected
    left, singular, vh = torch.linalg.svd(value.matrix)
    polynomial = evaluate(value.coefficients, singular / value.normalization).to(
        torch.complex128
    )
    predicted = vh.mH @ (polynomial * (left.mH @ value.source.to(torch.complex128)))
    predicted *= value.inverse_normalization * value.source_norm
    polynomial_error = float((solution - predicted).abs().max())
    rhs = value.source.to(torch.complex128) * value.source_norm
    exact = torch.linalg.solve(value.matrix, rhs)
    # The original classical helper includes i in its source, while the quantum
    # preparation explicitly omits it. Its field plot compares absolute values.
    classical = torch.linalg.solve(value.matrix, 1j * rhs)
    if full_error >= 1e-10 or polynomial_error >= 1e-10:
        raise ValueError(
            "The original-degree full state or polynomial projection failed"
        )
    return Result(
        encoded,
        value,
        state,
        solution[64::8],
        classical[64::8],
        full_error,
        polynomial_error,
        float((solution - exact).norm()),
        float(projected.abs().square().sum()),
        estimate_resources(circuit.to_ir()),
    )
