"""Solve both source nozzle systems with fresh QSVT phase sequences."""

import math
import time
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.chebyshev_approximation.overview.model import (
    source_matrix,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
    CTYPE,
    DTYPE,
    Encoding,
    banded_encoding,
    pauli_encoding,
    unitary,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.workflow import (
    coefficients,
    input_vector,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.model import (
    source_matrix as small_matrix,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.quantum import (
    preparation,
)
from flagquantum.compiler import ResourceEstimate, estimate_resources

from .phases import PhaseFit, fit, response


@dataclass(frozen=True)
class Result:
    request_degree: int
    block_width: int
    data_width: int
    alpha: float
    minimum_sigma: float
    phases: PhaseFit
    solution: torch.Tensor
    expected: torch.Tensor
    solution_l2: float
    spectral_error: float
    full_state_error: float
    projection_probability: float
    resources: ResourceEstimate
    build_seconds: float
    execute_seconds: float


def projector_signs(block: torch.Tensor, data_length: int) -> torch.Tensor:
    signs = -torch.ones(block.size(0), dtype=DTYPE)
    signs[:data_length] = 1
    return signs


def shifted_phases(fitted: PhaseFit) -> torch.Tensor:
    """Convert Wx QSP phases to the alternating block/adjoint convention."""
    shifted = fitted.phases - math.pi / 2
    shifted[0] += math.pi / 4
    shifted[-1] += math.pi / 4
    return shifted


def branch_matrix(
    block: torch.Tensor, signs: torch.Tensor, phases: torch.Tensor
) -> torch.Tensor:
    phases_by_row = torch.exp(1j * phases[:, None] * signs)
    state = torch.diag(phases_by_row[0])
    adjoint = block.mH
    for position, angle in enumerate(phases_by_row[1:], start=1):
        signal = adjoint if position % 2 else block
        state = angle[:, None] * (signal @ state)
    return state


def build_circuit(
    enc: Encoding, block: torch.Tensor, phases: torch.Tensor, rhs: torch.Tensor
) -> fq.Circuit:
    """Apply the two projector sequences as exact packed simulator operations."""
    encoded_qubits = enc.data_width + enc.block_width
    data = tuple(range(encoded_qubits + 1 - enc.data_width, encoded_qubits + 1))
    signs = projector_signs(block, rhs.numel())
    direct = branch_matrix(block, signs, phases)
    opposite = branch_matrix(block, signs, -phases)
    size = block.size(0)
    branches = torch.zeros((2 * size, 2 * size), dtype=CTYPE)
    branches[:size, :size] = direct
    branches[size:, size:] = opposite
    circuit = fq.Circuit(encoded_qubits + 1, dtype=CTYPE)
    circuit.compose(preparation(rhs / rhs.norm()), qubits=data)
    circuit.h(0)
    circuit.any(*range(encoded_qubits + 1), matrix=branches)
    return circuit.h(0)


def independent_state(
    block: torch.Tensor, phases: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Evaluate both streams on the input vector without packing an operator."""
    size = block.size(0)
    signs = projector_signs(block, rhs.numel())
    inputs = torch.zeros(size, dtype=CTYPE)
    inputs[: rhs.numel()] = rhs / rhs.norm()
    plus = inputs * (1j * float(phases[0]) * signs).exp()
    minus = inputs * (-1j * float(phases[0]) * signs).exp()
    for index, angle in enumerate(phases[1:], 1):
        signal = block.mH if index % 2 else block
        rotation = (1j * float(angle) * signs).exp()
        plus = rotation * (signal @ plus)
        minus = rotation.conj() * (signal @ minus)
    return torch.cat((plus + minus, plus - minus)) / 2


def solve(large: bool) -> Result:
    start = time.perf_counter()
    original = source_matrix().to(CTYPE) if large else small_matrix()
    enc = banded_encoding(original) if large else pauli_encoding(original)
    degree = 255 if large else 101
    coefs, factor, cutoff = coefficients(original, enc.alpha, degree)
    fitted = fit(coefs)
    phases = shifted_phases(fitted)
    block = unitary(enc)
    rhs = input_vector("nozzle_008_b" if large else "b_nozzle_small", enc.data_width)
    circuit = build_circuit(enc, block, phases, rhs)
    build_duration = time.perf_counter() - start

    start = time.perf_counter()
    actual = fq.run(circuit).statevector().flatten()
    execute_duration = time.perf_counter() - start
    expected_state = independent_state(block, phases, rhs)
    torch.testing.assert_close(actual, expected_state, atol=1e-10, rtol=0)
    zero = actual[: rhs.numel()]
    sign = (-1) ** ((degree - 1) // 2)
    solution = zero * (sign * rhs.norm() / (enc.alpha * factor))
    u, sigma, vh = torch.linalg.svd(enc.target / enc.alpha)
    p = response(sigma, fitted.phases)
    projected = vh.mH @ (p * (u.mH @ rhs.to(CTYPE))) / (enc.alpha * factor)
    torch.testing.assert_close(solution, projected, atol=1e-10, rtol=0)
    exact = torch.linalg.solve(original, rhs.to(CTYPE))
    return Result(
        degree,
        enc.block_width + 1,
        enc.data_width,
        enc.alpha,
        cutoff * enc.alpha,
        fitted,
        solution.real,
        exact.real,
        float((solution - exact).norm()),
        float((solution - projected).abs().max()),
        float((actual - expected_state).abs().max()),
        float(zero.abs().square().sum()),
        estimate_resources(circuit),
        build_duration,
        execute_duration,
    )
