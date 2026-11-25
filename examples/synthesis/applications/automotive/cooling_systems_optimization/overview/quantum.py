"""Compose the thermal block encoding and the original mock-QAE template."""

import math

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.quantum import (
    preparation,
)

from .model import (
    COMPLEX,
    CONFIGURATIONS,
    EDGES,
    HEAT,
    REAL,
    Reciprocal,
    alpha,
    weights,
)

THERMAL_DATA = 2
THERMAL_BLOCK = 5
THERMAL_WIDTH = 7
FULL_QAE_BITS = 5
MOCK_QAE_BITS = 1
QAE_PHASE_WEIGHT = math.sin(math.pi / (1 << FULL_QAE_BITS))
DESIGN_BITS = len(EDGES)


def pair_block(i: int, j: int, *, enabled: bool) -> torch.Tensor:
    """Encode the thermal pair on flag, pair-ancilla and two data bits."""
    condition = i ^ j
    transform = torch.zeros(16, 16, dtype=COMPLEX)
    if not enabled:
        for word in range(16):
            transform[word ^ 8, word] = 1
        return transform
    for word in range(16):
        point = word & 3
        linked = point in (i, j)
        rotate = (word & 4) != 0
        target = word ^ (0 if linked else 8)
        if rotate:
            target ^= condition
        transform[target, word] = -1 if rotate else 1
    h = torch.tensor(((1, 1), (1, -1)), dtype=COMPLEX) / math.sqrt(2)
    lift = torch.kron(
        torch.eye(2, dtype=COMPLEX), torch.kron(h, torch.eye(4, dtype=COMPLEX))
    )
    return lift @ transform @ lift


def thermal_encoding(word: int = 63) -> fq.Circuit:
    """Use the source's six pair terms, ambient identity and unused selector."""
    prep = preparation((torch.tensor(weights(), dtype=REAL) / (alpha() / 2)).sqrt())
    selection = torch.eye(128, dtype=COMPLEX)
    for index, (i, j) in enumerate(EDGES):
        portion = slice(16 * index, 16 * (index + 1))
        selected = bool(word & (1 << (DESIGN_BITS - 1 - index)))
        selection[portion, portion] = pair_block(i, j, enabled=selected)
    circuit = fq.Circuit(THERMAL_WIDTH, dtype=COMPLEX).compose(prep, qubits=(0, 1, 2))
    circuit.any(*range(THERMAL_WIDTH), matrix=selection)
    return circuit.compose(prep.adjoint(), qubits=(0, 1, 2))


def conditional_thermal_encoding() -> fq.Circuit:
    """Define the original conditional thermal oracle for every six-bit design."""
    prep = preparation((torch.tensor(weights(), dtype=REAL) / (alpha() / 2)).sqrt())
    selectors = (6, 7, 8)
    thermal_wires = tuple(range(6, 13))
    circuit = fq.Circuit(13, dtype=COMPLEX).compose(prep, qubits=selectors)
    for index, (i, j) in enumerate(EDGES):
        operator = torch.eye(256, dtype=COMPLEX)
        for active in (False, True):
            offset = 128 * int(active) + 16 * index
            span = slice(offset, offset + 16)
            operator[span, span] = pair_block(i, j, enabled=active)
        circuit.any(index, *thermal_wires, matrix=operator)
    return circuit.compose(prep.adjoint(), qubits=selectors)


def inverse_circuit(reciprocal: Reciprocal) -> fq.Circuit:
    """Prepare the signed heat vector and apply all eleven encoded inverse calls."""
    block = thermal_encoding()
    inverse = block.adjoint()
    angles = reciprocal.fit.phases - math.pi / 2
    angles[0] += math.pi / 4
    angles[-1] += math.pi / 4
    if angles.numel() != 12:
        raise ValueError("Expected the source degree-11 reciprocal")
    heat = torch.tensor(HEAT, dtype=REAL)
    circuit = (
        fq.Circuit(8, dtype=COMPLEX)
        .compose(preparation(heat / heat.norm()), qubits=(6, 7))
        .h(0)
    )
    words = torch.arange(64)
    signs = 2 * (((words & 31) == 0) == (words >> 5 == 0)).to(REAL) - 1
    for index, angle in enumerate(angles):
        circuit.any(*range(6), matrix=torch.diag((1j * angle * signs).exp()))
        if index != angles.numel() - 1:
            circuit.compose(
                inverse if index % 2 == 0 else block, qubits=tuple(range(1, 8))
            )
    return circuit.h(0)


def qae_state(cost: float) -> torch.Tensor:
    """Form the source's two signed eigenphase branches for one phase bit."""
    phase = math.asin(cost)
    k = torch.arange(2, dtype=REAL)
    counter = torch.arange(2, dtype=REAL)[:, None]
    positive = (1j * k * (2 * phase - math.pi * counter)).exp().mean(dim=1)
    negative = (1j * k * (-2 * phase - math.pi * counter)).exp().mean(dim=1)
    cp = (math.sin(phase) - 1j * math.cos(phase)) / math.sqrt(2)
    cm = (math.sin(phase) + 1j * math.cos(phase)) / math.sqrt(2)
    accepted = (cp * positive + cm * negative) / math.sqrt(2)
    rejected = 1j * (cp * positive - cm * negative) / math.sqrt(2)
    amplitudes = torch.stack((rejected, accepted), dim=1).flatten()
    rounded = torch.complex(
        amplitudes.real.round(decimals=10), amplitudes.imag.round(decimals=10)
    )
    normalized: torch.Tensor = rounded / rounded.norm()
    return normalized


def complex_preparation(amplitudes: torch.Tensor) -> fq.Circuit:
    """Prepare passed complex amplitudes with real magnitude rotations and phases."""
    real = preparation(amplitudes.abs())
    circuit = fq.Circuit(2, dtype=COMPLEX).compose(real)
    return circuit.any(0, 1, matrix=torch.diag((1j * amplitudes.angle()).exp()))


def unitary(circuit: fq.Circuit) -> torch.Tensor:
    """Read a small local unitary by executing each basis column."""
    dimension = 1 << circuit.n_qubits
    if dimension > 128:
        raise ValueError("The verification batch is limited to seven wires")
    batched = fq.Circuit.from_ir(
        circuit.to_ir(),
        dtype=COMPLEX,
        bsz=dimension,
        inputs=torch.eye(dimension, dtype=COMPLEX),
    )
    state: torch.Tensor = batched.state().reshape(dimension, dimension)
    return state.T.contiguous()


def mock_qaoa(costs: torch.Tensor) -> fq.Circuit:
    """Build and reuse the same symbolic one-layer eight-qubit mock model."""
    selection = torch.zeros((256, 256), dtype=COMPLEX)
    for index in range(CONFIGURATIONS):
        prep = complex_preparation(qae_state(float(costs[index])))
        span = slice(4 * index, 4 * (index + 1))
        selection[span, span] = unitary(prep)
    wires = tuple(range(8))
    circuit = fq.Circuit(8, dtype=COMPLEX)
    for wire in range(DESIGN_BITS):
        circuit.h(wire)
    circuit.any(*wires, matrix=selection)
    circuit.phase(6, theta=QAE_PHASE_WEIGHT * fq.Parameter("gamma"))
    circuit.any(*wires, matrix=selection.mH)
    beta = fq.Parameter("beta")
    for wire in range(DESIGN_BITS):
        circuit.rx(wire, theta=-2 * beta)
    return circuit
