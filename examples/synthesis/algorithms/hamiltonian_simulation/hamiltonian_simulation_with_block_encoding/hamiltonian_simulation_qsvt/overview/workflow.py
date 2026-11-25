"""Run the original degree-33 Jacobi–Anger Hamiltonian evolution."""

import math
import time
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
    CTYPE,
    DTYPE,
    Encoding,
    unitary,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    PhaseFit,
    fit,
    response,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.workflow import (
    branch_matrix,
    projector_signs,
    shifted_phases,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.model import (
    PAULIS,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.quantum import (
    preparation,
)
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.compiler import ResourceEstimate, estimate_resources

from .even_phases import fit_even

TIME = 22
EPS = 1e-7
DEGREE = 33
SCALE = 0.9999
WORDS = ("II", "ZI", "XX", "ZZ")
WEIGHTS = (0.4, 0.1, 0.05, 0.2)
ALPHA = math.fsum(WEIGHTS)
DEFAULT_SEED = 132
HADAMARD = torch.tensor(((1, 1), (1, -1)), dtype=CTYPE) / math.sqrt(2)


@dataclass(frozen=True)
class Result:
    model: Encoding
    cosine: PhaseFit
    sine: PhaseFit
    state: torch.Tensor
    expected: torch.Tensor
    evolution_error: float
    full_state_error: float
    block_error: float
    cosine_error: float
    sine_error: float
    overlap: float
    success_probability: float
    resources: ResourceEstimate
    phase_seconds: float


def encoding() -> Encoding:
    matrix = torch.zeros((4, 4), dtype=CTYPE)
    for word, weight in zip(WORDS, WEIGHTS, strict=True):
        matrix += weight * torch.kron(PAULIS[word[0]], PAULIS[word[1]])
    weights = preparation(torch.tensor(WEIGHTS, dtype=DTYPE).div(ALPHA).sqrt())
    circuit = fq.Circuit(4, dtype=CTYPE).compose(weights, qubits=(0, 1))
    operator = SelectPauliLCU(2, 2, WORDS)
    circuit.append(operator.make_on(selection=(0, 1), target=(2, 3)))
    circuit.compose(weights.adjoint(), qubits=(0, 1))
    return Encoding(circuit, ALPHA, matrix, 2, 2, (4, 4))


def input_state(seed: int = DEFAULT_SEED) -> torch.Tensor:
    values = torch.rand(4, dtype=DTYPE, generator=torch.Generator().manual_seed(seed))
    normalized: torch.Tensor = values / values.norm()
    return normalized


def coefficients(duration: float) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = 4096
    theta = math.pi * (torch.arange(nodes, dtype=DTYPE) + 0.5) / nodes
    orders = torch.arange(DEGREE + 1, dtype=DTYPE)
    bessel = torch.cos(orders[:, None] * theta - duration * theta.sin()).mean(dim=1)
    cosine = torch.cat(
        (bessel[:1], 2 * bessel[2::2] * (-1) ** torch.arange(1, DEGREE // 2 + 1))
    )
    sine = 2 * bessel[1::2] * (-1) ** torch.arange((DEGREE + 1) // 2)
    return SCALE * cosine, SCALE * sine


def selection_weights(degree: int) -> tuple[complex, complex]:
    if degree % 2:
        sign = complex((-1) ** ((degree - 1) // 2))
        return sign, sign
    phase = 1j * (-1) ** (degree // 2)
    return -phase, phase


def selected_response(block: torch.Tensor, fitted: PhaseFit) -> torch.Tensor:
    shifted = shifted_phases(fitted)
    signs = projector_signs(block, 4)
    left, right = selection_weights(shifted.numel() - 1)
    plus = left * branch_matrix(block, signs, shifted)
    minus = right * branch_matrix(block, signs, -shifted)
    dimension = block.size(0)
    combined = torch.zeros((2 * dimension, 2 * dimension), dtype=CTYPE)
    combined[:dimension, :dimension] = plus
    combined[dimension:, dimension:] = minus
    toggle = torch.kron(HADAMARD, torch.eye(dimension, dtype=CTYPE))
    selected: torch.Tensor = toggle @ combined @ toggle
    return selected


def evolution_circuit(
    block: torch.Tensor, cosine: PhaseFit, sine: PhaseFit, amps: torch.Tensor
) -> fq.Circuit:
    cos_gate = selected_response(block, cosine)
    sin_gate = selected_response(block, sine)
    size = cos_gate.size(0)
    selection = torch.zeros((2 * size, 2 * size), dtype=CTYPE)
    selection[:size, :size] = cos_gate
    selection[size:, size:] = -1j * sin_gate
    circuit = fq.Circuit(6, dtype=CTYPE).compose(preparation(amps), qubits=(4, 5))
    circuit.h(0).any(*range(6), matrix=selection).h(0)
    return circuit


def sequential(
    block: torch.Tensor,
    signs: torch.Tensor,
    angles: torch.Tensor,
    start: torch.Tensor,
) -> torch.Tensor:
    state = (1j * float(angles[0]) * signs).exp() * start
    for index, angle in enumerate(angles[1:], start=1):
        operation = block.mH if index % 2 else block
        state = (1j * float(angle) * signs).exp() * (operation @ state)
    return state


def full_reference(
    block: torch.Tensor, cosine: PhaseFit, sine: PhaseFit, amps: torch.Tensor
) -> torch.Tensor:
    start = torch.zeros(block.size(0), dtype=CTYPE)
    start[:4] = amps
    signs = projector_signs(block, 4)
    streams = []
    for fitting in (cosine, sine):
        shifted = shifted_phases(fitting)
        positive_weight, negative_weight = selection_weights(shifted.numel() - 1)
        positive = positive_weight * sequential(block, signs, shifted, start)
        negative = negative_weight * sequential(block, signs, -shifted, start)
        streams.append(torch.cat((positive + negative, positive - negative)) / 2)
    return torch.cat((streams[0] - 1j * streams[1], streams[0] + 1j * streams[1])) / 2


def run(seed: int = DEFAULT_SEED) -> Result:
    model = encoding()
    block = unitary(model)
    amps = input_state(seed)
    block_program = fq.Circuit(4, dtype=CTYPE).compose(preparation(amps), qubits=(2, 3))
    block_program.compose(model.circuit, qubits=(0, 1, 2, 3))
    block_signal = ALPHA * fq.run(block_program).statevector().flatten()[:4]
    block_expected = model.target @ amps.to(CTYPE)
    torch.testing.assert_close(block_signal, block_expected, atol=1e-9, rtol=0)

    start = time.perf_counter()
    evolution_time = TIME * ALPHA
    even, odd = coefficients(evolution_time)
    cosine, sine = fit_even(even), fit(odd)
    phase_seconds = time.perf_counter() - start
    grid = torch.linspace(-1, 1, 4097, dtype=DTYPE)
    cosine_error = float(
        (response(grid, cosine.phases) - SCALE * (evolution_time * grid).cos())
        .abs()
        .max()
    )
    sine_error = float(
        (response(grid, sine.phases) - SCALE * (evolution_time * grid).sin())
        .abs()
        .max()
    )

    circuit = evolution_circuit(block, cosine, sine, amps)
    full = fq.run(circuit).statevector().flatten()
    independently = full_reference(block, cosine, sine, amps)
    torch.testing.assert_close(full, independently, atol=1e-10, rtol=0)
    result = full[:4] * (2 / SCALE)
    expected = torch.linalg.matrix_exp(-1j * TIME * model.target) @ amps.to(CTYPE)
    discrepancy = float((result - expected).norm())
    if discrepancy >= EPS:
        raise ValueError("Evolution error must meet the original 1e-7 threshold")
    return Result(
        model,
        cosine,
        sine,
        result,
        expected,
        discrepancy,
        float((full - independently).abs().max()),
        float((block_signal - block_expected).abs().max()),
        cosine_error,
        sine_error,
        float((expected.vdot(result).abs() / (expected.norm() * result.norm())) ** 2),
        float(full[:4].abs().square().sum()),
        estimate_resources(circuit),
        phase_seconds,
    )
