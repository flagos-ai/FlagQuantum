"""Run the three source Chebyshev-LCU solves with fresh spectral normalization."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.model import (
    SCALE,
    evaluate,
    fit_alternating,
    interpolation_grid,
    odd_table,
    relative_reciprocal,
    source_matrix,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.quantum import (
    approximation,
    load_distribution,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.model import (
    source_matrix as small_matrix,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.quantum import (
    preparation,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources

from .encoding import CTYPE, DTYPE, Encoding, banded_encoding, pauli_encoding, unitary
from .reference import full_state


@dataclass(frozen=True)
class Result:
    degree: int
    selector_width: int
    encoding: Encoding
    relative_cutoff: float
    solution: torch.Tensor
    expected: torch.Tensor
    solution_error: float
    projected_reference_error: float
    full_state_error: float
    preparation_error: float
    projection_probability: float
    resources: ResourceEstimate
    build_seconds: float
    execution_seconds: float


def input_vector(name: str, width: int) -> torch.Tensor:
    data = json.loads(Path(__file__).with_name(name + ".json").read_text())
    vector = torch.tensor(data, dtype=DTYPE)
    if vector.shape != (1 << width,) or not bool(vector.isfinite().all()):
        raise ValueError("The original right-hand side must match the data register")
    return vector


def coefficients(
    matrix: torch.Tensor, alpha: float, degree: int
) -> tuple[torch.Tensor, float, float]:
    a = float(torch.linalg.svdvals(matrix / alpha)[-1])
    grid = interpolation_grid(degree + 1)
    values = relative_reciprocal(grid, a, degree) / a
    coefficients = 2 / grid.numel() * (odd_table(grid, degree).T @ values)
    peak = float(evaluate(coefficients, interpolation_grid(32768)).abs().max())
    scale = SCALE / peak
    return scale * coefficients, scale, a


def inverse_preparation(
    probabilities: torch.Tensor, tolerance: float
) -> tuple[fq.Circuit, torch.Tensor, int]:
    expected = probabilities.sqrt()
    bits = 0
    if tolerance:
        bits, group = approximation(probabilities, tolerance)
        program = synth.synthesize(
            load_distribution,
            classical_inputs={
                "amplitudes": list(map(float, group.sqrt())),
                "uniform_bits": bits,
            },
            max_qubits=7,
            max_workspace=0,
        )
        circuit = fq.Circuit.from_ir(program.circuit, dtype=CTYPE)
    else:
        circuit = preparation(expected)
    state = fq.run(circuit).statevector().flatten()
    if float((state - expected).norm()) > tolerance + 1e-12:
        raise ValueError("Inverse coefficient preparation exceeds its tolerance")
    return circuit, state, bits


def walk_circuit(
    encoding: Encoding,
    b: torch.Tensor,
    rhs: torch.Tensor,
    prep: fq.Circuit,
    inverse_width: int,
) -> fq.Circuit:
    width = encoding.data_width + encoding.block_width
    size = 1 << width
    block_zero = 1 << encoding.data_width
    reflection_signs = -torch.ones(size, dtype=DTYPE)
    reflection_signs[:block_zero] = 1.0
    reflection = torch.diag(reflection_signs).to(CTYPE)
    symmetrized = b @ reflection @ b.mH @ reflection
    selector = tuple(range(inverse_width))
    encoded = tuple(range(inverse_width, inverse_width + width))
    data = encoded[-encoding.data_width :]
    circuit = fq.Circuit(inverse_width + width, dtype=CTYPE)
    circuit.compose(prep, qubits=selector)
    circuit.compose(preparation(rhs / rhs.norm()), qubits=data)
    circuit.z(selector[-1])
    controlled = torch.eye(2 * size, dtype=CTYPE)
    for wire in reversed(selector):
        # One exact powered block for each original binary control.
        controlled[size:, size:] = symmetrized
        circuit.any(wire, *encoded, matrix=controlled.clone())
        symmetrized = symmetrized @ symmetrized
    circuit.any(*encoded, matrix=reflection @ b.mH @ reflection)
    circuit.compose(prep.adjoint(), qubits=selector)
    return circuit


def solve(large: bool, tolerance: float = 0.0) -> Result:
    if (
        not math.isfinite(tolerance)
        or tolerance not in (0.0, 0.03)
        or (not large and tolerance != 0)
    ):
        raise ValueError("Use exact preparation or the original 0.03 banded tolerance")
    build = time.perf_counter()
    matrix = source_matrix().to(CTYPE) if large else small_matrix()
    enc = banded_encoding(matrix) if large else pauli_encoding(matrix)
    rhs = input_vector("nozzle_008_b" if large else "b_nozzle_small", enc.data_width)
    degree = 255 if large else 31
    inverse_width = 7 if large else 4
    coefs, scale, cutoff = coefficients(matrix, enc.alpha, degree)
    if tolerance:
        coefs = fit_alternating(coefs)
    signs = 1 - 2 * (torch.arange(coefs.numel()) % 2)
    if not bool((coefs.sign() == signs).all()):
        raise ValueError("The odd terms must alternate signs")
    normalization = float(coefs.abs().sum())
    probabilities = coefs.abs() / normalization
    prep, prepped, uniform_bits = inverse_preparation(probabilities, tolerance)
    matrix_unitary = unitary(enc)
    circuit = walk_circuit(enc, matrix_unitary, rhs, prep, inverse_width)
    build_seconds = time.perf_counter() - build
    start = time.perf_counter()
    state = fq.run(circuit).statevector().flatten()
    runtime = time.perf_counter() - start
    reference = full_state(matrix_unitary, rhs, probabilities, uniform_bits)
    torch.testing.assert_close(state, reference, atol=1e-10, rtol=0)
    dimension = rhs.numel()
    projected = state[:dimension]
    actual = projected * (normalization * rhs.norm() / (enc.alpha * scale))
    u, sigma, vh = torch.linalg.svd(enc.target / enc.alpha)
    matched = signs * prepped.abs().square() * normalization
    expected_filtered = (
        vh.mH
        @ (evaluate(matched, sigma) * (u.mH @ rhs.to(CTYPE)))
        / (enc.alpha * scale)
    )
    torch.testing.assert_close(actual, expected_filtered, atol=1e-9, rtol=0)
    expected = torch.linalg.solve(matrix, rhs.to(CTYPE))
    return Result(
        degree,
        inverse_width,
        enc,
        cutoff,
        actual.real,
        expected.real,
        float((actual - expected).norm()),
        float((actual - expected_filtered).abs().max()),
        float((state - reference).abs().max()),
        float((prepped - probabilities.sqrt()).norm()),
        float(projected.abs().square().sum()),
        estimate_resources(circuit),
        build_seconds,
        runtime,
    )
