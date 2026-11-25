"""Execute the original graph encodings and both 20-point walk experiments."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.algorithms.core import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.primitives.linear_combination import (
    LinearCombinationEncoding,
)
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis

from . import model, quantum, reference

TREE_SIZES = (2, 4, 8, 16)
REPETITIONS = tuple(range(1, 21))
LOCAL_SHOTS = 1_000


@dataclass(frozen=True)
class TreeResult:
    size: int
    lcu_terms: int
    alpha: float
    oracle_resources: ResourceEstimate
    lcu_resources: ResourceEstimate
    oracle_error: float
    lcu_error: float
    projection: torch.Tensor


@dataclass(frozen=True)
class WalkResult:
    marked: bool
    exact: tuple[float, ...]
    counts: tuple[int, ...]
    state_error: float
    block_error: float
    projection: torch.Tensor


@dataclass(frozen=True)
class Result:
    trees: tuple[TreeResult, ...]
    cyclic_projection: torch.Tensor
    cyclic_error: float
    chebyshev_projection: torch.Tensor
    chebyshev_error: float
    walks: tuple[WalkResult, WalkResult]


def state_and_block(circuit: fq.Circuit, data_size: int) -> torch.Tensor:
    """Execute every encoded data basis column with all flags initially zero."""
    full_size = 1 << circuit.n_qubits
    inputs = torch.zeros(data_size, full_size, dtype=model.COMPLEX)
    indices = torch.arange(data_size)
    inputs[indices, indices] = 1
    batched = fq.Circuit.from_ir(
        circuit.to_ir(), dtype=model.COMPLEX, bsz=data_size, inputs=inputs
    )
    state: torch.Tensor = batched.state().reshape(data_size, full_size)
    return state.T.contiguous()


def error(
    actual: torch.Tensor, expected: torch.Tensor, *, tolerance: float = 1e-10
) -> float:
    maximum = float((actual - expected).abs().max())
    if maximum > tolerance:
        raise ValueError(
            f"Independent state verification failed: {maximum} > {tolerance}"
        )
    return maximum


def resources(circuit: fq.Circuit) -> ResourceEstimate:
    lowered = convert_basis(circuit, gates=("cx", "u"), max_added_operations=100_000)
    return estimate_resources(lowered.program)


def tree_case(size: int) -> TreeResult:
    width = size.bit_length() - 1
    matrix = model.tree(size)
    oracle = quantum.tree_encoding(width)
    output = state_and_block(oracle, size)
    basis = torch.eye(size, dtype=model.COMPLEX)
    expected = torch.stack(
        [reference.tree_action(size, quantum.TREE_THETA, column) for column in basis],
        dim=1,
    )
    oracle_error = max(error(output, expected), error(output[:size], matrix / 8))

    terms = model.pauli_terms(matrix)
    encoding = LinearCombinationEncoding(
        Hamiltonian(
            HamiltonianTerm(weight, word, range(width)) for weight, word in terms
        )
    )
    lcu = fq.Circuit(encoding.num_ancilla + width, dtype=model.COMPLEX)
    encoding.append_apply(
        lcu, ancilla=0, qubits=tuple(range(encoding.num_ancilla, lcu.n_qubits))
    )
    generator = torch.Generator().manual_seed(641 + size)
    data = torch.stack(
        (
            torch.full((size,), 1 / math.sqrt(size), dtype=model.REAL),
            torch.rand(size, dtype=model.REAL, generator=generator) - 0.5,
        )
    ).to(model.COMPLEX)
    data /= data.norm(dim=1, keepdim=True)
    initial = torch.zeros(2, 1 << lcu.n_qubits, dtype=model.COMPLEX)
    initial[:, :size] = data
    bound_lcu = fq.Circuit.from_ir(
        lcu.to_ir(), dtype=model.COMPLEX, bsz=2, inputs=initial
    )
    lcu_output: torch.Tensor = bound_lcu.state().reshape(2, -1).T.contiguous()
    expected_lcu = matrix.to(model.COMPLEX) @ data.T / encoding.alpha
    lcu_error = error(lcu_output[:size], expected_lcu, tolerance=1e-6)
    if encoding.num_ancilla > encoding.index_width:
        work = encoding.num_ancilla - encoding.index_width
        amps = lcu_output.reshape(1 << encoding.index_width, 1 << work, size, 2)
        if float(amps[:, 1:].abs().square().sum()) > 1e-20:
            raise ValueError("LCU compiler work must remain clean")
    projection = output[:size].sum(dim=1) / math.sqrt(size)
    return TreeResult(
        size,
        len(terms),
        encoding.alpha,
        resources(oracle),
        resources(lcu),
        oracle_error,
        lcu_error,
        projection,
    )


def walk_case(marked: bool) -> WalkResult:
    verify = quantum.discriminant_walk(2, marked=marked)
    small = state_and_block(verify, 4)[:4]
    block = model.discriminant(4, marked=marked)
    block_error = error(-small, block, tolerance=1e-5)
    projection = -small.sum(dim=1) / 2

    width = 4
    size = 1 << width
    template = quantum.discriminant_walk(width, marked=marked)
    cumulative = fq.Circuit(2 * width + 1, dtype=model.COMPLEX)
    for wire in range(width + 1, 2 * width + 1):
        cumulative.h(wire)
    step_ir = template.to_ir()
    state: torch.Tensor = cumulative.state().flatten()
    expected = torch.zeros(size, size, dtype=model.COMPLEX)
    expected[0] = 1 / math.sqrt(size)
    exact, counts = [], []
    maximum = 0.0
    for repetition in REPETITIONS:
        state = (
            fq.Circuit.from_ir(step_ir, dtype=model.COMPLEX, inputs=state)
            .state()
            .flatten()
        )
        actual = state.reshape(2, size, size)
        expected = reference.walk_step(expected, width, marked=marked)
        maximum = max(
            error(actual[0], expected),
            error(actual[1], torch.zeros_like(expected)),
            maximum,
        )
        probabilities = actual.abs().square().flatten()
        exact.append(float(probabilities[:size].sum()))
        generator = torch.Generator().manual_seed(4400 + 100 * marked + repetition)
        outcomes = torch.multinomial(
            probabilities, LOCAL_SHOTS, replacement=True, generator=generator
        )
        counts.append(int(((outcomes // size) % size == 0).sum()))
    return WalkResult(
        marked, tuple(exact), tuple(counts), maximum, block_error, projection
    )


def run() -> Result:
    trees = tuple(tree_case(size) for size in TREE_SIZES)
    stochastic = quantum.cyclic_encoding()
    cyclic = state_and_block(stochastic, 8)
    exact = reference.cyclic_operator(quantum.CYCLIC_THETA)
    cyclic_error = max(error(cyclic, exact[:, :8]), error(cyclic[:8], model.cyclic()))
    polynomial = quantum.cheb_circuit().state().flatten()
    cheb_error = error(
        polynomial, reference.chebyshev_state(exact, quantum.CHEB_AMPLITUDES)
    )
    matrix = model.cyclic()
    term = (2 * matrix @ matrix - torch.eye(8, dtype=model.REAL)) @ torch.tensor(
        quantum.CHEB_AMPLITUDES, dtype=model.REAL
    )
    cheb_error = max(cheb_error, error(-polynomial[:8], term))
    return Result(
        trees,
        cyclic[:8].sum(dim=1) / math.sqrt(8),
        cyclic_error,
        -polynomial[:8],
        cheb_error,
        (walk_case(False), walk_case(True)),
    )
