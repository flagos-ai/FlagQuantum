"""Construct the source Pauli and banded encodings from matrix entries."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.model import (
    PAULIS,
    pauli_expansion,
)
from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.verify_block_encoding.overview.quantum import (
    preparation,
)
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)

CTYPE = torch.complex128
DTYPE = torch.float64


@dataclass(frozen=True)
class Encoding:
    circuit: fq.Circuit
    alpha: float
    target: torch.Tensor
    data_width: int
    block_width: int
    pauli_counts: tuple[int, int] | None


def _pauli_keep(
    terms: tuple[tuple[str, complex], ...],
) -> tuple[tuple[str, complex], ...]:
    weights = sorted((abs(value) for _, value in terms), reverse=True)
    threshold = 0.1 * weights[1]
    start = 0
    cutoff = weights[0]
    for index, value in enumerate(weights):
        if index + 1 < len(weights) and value / weights[index + 1] < 1.1:
            continue
        if weights[start] < threshold:
            break
        cutoff = value
        if value <= threshold:
            break
        start = index + 1
    return tuple((word, value) for word, value in terms if abs(value) >= cutoff)


def pauli_encoding(matrix: torch.Tensor) -> Encoding:
    original = pauli_expansion(matrix)
    terms = _pauli_keep(original)
    if (len(original), len(terms)) != (24, 20):
        raise ValueError("Expected the nozzle_small 24-to-20 Pauli trim")
    n = 3
    m = (len(terms) - 1).bit_length()
    size = 1 << m
    selected = tuple(word for word, _ in terms)
    values = tuple(value for _, value in terms)
    alpha = math.fsum(map(abs, values))
    probabilities = tuple(abs(value) / alpha for value in values)
    weights = preparation(
        torch.tensor(
            (*probabilities, *((0.0,) * (size - len(terms)))), dtype=DTYPE
        ).sqrt()
    )
    block = tuple(range(m))
    data = tuple(range(m, m + n))
    circuit = fq.Circuit(m + n, dtype=CTYPE).compose(weights, qubits=block)
    circuit.append(SelectPauliLCU(m, n, selected).make_on(selection=block, target=data))
    factors = [value / abs(value) for value in values] + [1.0] * (size - len(values))
    circuit.any(*block, matrix=torch.diag(torch.tensor(factors, dtype=CTYPE)))
    circuit.compose(weights.adjoint(), qubits=block)
    target = torch.zeros_like(matrix)
    for word, value in terms:
        first, second, third = (PAULIS[axis] for axis in word)
        target += value * torch.kron(torch.kron(first, second), third)
    return Encoding(circuit, alpha, target, n, m, (len(original), len(terms)))


def banded_encoding(matrix: torch.Tensor) -> Encoding:
    size = matrix.size(0)
    n = size.bit_length() - 1
    widths = {
        offset: float(matrix.diagonal(offset).abs().max())
        for offset in range(1 - size, size)
        if bool(matrix.diagonal(offset).any())
    }
    alpha = math.fsum(widths.values())
    selectors = (len(widths) - 1).bit_length()
    block_width = selectors + 1
    group_size = 2 * size
    prep_prob = [weight / alpha for weight in widths.values()] + [0.0] * (
        (1 << selectors) - len(widths)
    )
    prep = preparation(torch.tensor(prep_prob, dtype=DTYPE).sqrt())
    selection = torch.eye((1 << selectors) * group_size, dtype=CTYPE)
    for index, (offset, scale) in enumerate(widths.items()):
        values = torch.zeros(size, dtype=DTYPE)
        shift = torch.zeros((size, size), dtype=CTYPE)
        for col in range(size):
            row = col - offset
            if 0 <= row < size:
                values[col] = matrix[row, col].real / scale
            shift[row % size, col] = 1.0
        diagonal = torch.diag(values).to(CTYPE)
        rest = torch.diag((1 - values.square()).clamp_min(0).sqrt()).to(CTYPE)
        dilation = torch.cat(
            (torch.cat((diagonal, rest), 1), torch.cat((rest, -diagonal), 1)), 0
        )
        pair_shift = torch.kron(torch.eye(2, dtype=CTYPE), shift)
        group = slice(index * group_size, (index + 1) * group_size)
        selection[group, group] = pair_shift @ dilation
    wires = tuple(range(block_width + n))
    selector_wires = wires[:selectors]
    circuit = fq.Circuit(len(wires), dtype=CTYPE).compose(prep, qubits=selector_wires)
    circuit.any(*wires, matrix=selection)
    circuit.compose(prep.adjoint(), qubits=selector_wires)
    return Encoding(circuit, alpha, matrix, n, block_width, None)


def unitary(encoding: Encoding) -> torch.Tensor:
    """Read the block encoding's linear action using a batch of basis states."""
    dimensions = 1 << encoding.circuit.n_qubits
    initial = torch.eye(dimensions, dtype=CTYPE)
    batched = fq.Circuit.from_ir(
        encoding.circuit.to_ir(), dtype=CTYPE, bsz=dimensions, inputs=initial
    )
    result: torch.Tensor = (
        batched.state().reshape(dimensions, dimensions).T.contiguous()
    )
    data = 1 << encoding.data_width
    torch.testing.assert_close(
        result[:data, :data], encoding.target / encoding.alpha, atol=1e-12, rtol=0
    )
    torch.testing.assert_close(result.mH @ result, initial, atol=1e-12, rtol=0)
    return result
