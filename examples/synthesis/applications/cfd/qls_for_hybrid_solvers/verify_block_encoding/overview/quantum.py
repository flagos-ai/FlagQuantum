"""Native PREPARE with explicit Pauli and banded SELECT blocks."""

import cmath

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.synth import Output, QArray, QBit, prepare_amplitudes, qfunc

from .model import CTYPE, bands, pauli_expansion


@qfunc
def prepare(amplitudes: list[float], q: Output[QArray[QBit]]) -> None:
    prepare_amplitudes(amplitudes, 0, q)


def preparation(values: torch.Tensor) -> fq.Circuit:
    data = tuple(float(value) for value in values)
    n = len(data).bit_length() - 1
    program = synth.synthesize(
        prepare, classical_inputs={"amplitudes": data}, max_qubits=n, max_workspace=0
    )
    assert program.registers["q"] == tuple(range(n))
    return fq.Circuit.from_ir(program.circuit, dtype=CTYPE)


def pauli_encoding(matrix: torch.Tensor, rhs: torch.Tensor) -> tuple[fq.Circuit, float]:
    coefficients = pauli_expansion(matrix)
    size = 32
    weights = [weight for _, weight in coefficients]
    alpha = sum(abs(weight) for weight in weights)
    amps = torch.zeros(size, dtype=torch.float64)
    amps[: len(weights)] = torch.tensor(
        [(abs(w) / alpha) ** 0.5 for w in weights], dtype=torch.float64
    )
    prep = preparation(amps)
    circuit = fq.Circuit(8, dtype=CTYPE)
    selectors = tuple(range(5))
    data = (5, 6, 7)
    circuit.compose(prep, qubits=selectors).compose(preparation(rhs), qubits=data)
    circuit.append(
        SelectPauliLCU(5, 3, tuple(label for label, _ in coefficients)).make_on(
            selection=selectors, target=data
        )
    )
    phases = torch.tensor(
        [cmath.exp(1j * cmath.phase(w)) for w in weights] + [1] * (size - len(weights)),
        dtype=CTYPE,
    )
    circuit.any(*selectors, matrix=torch.diag(phases))
    circuit.compose(prep.adjoint(), qubits=selectors)
    return circuit, alpha


def banded_encoding(
    matrix: torch.Tensor, rhs: torch.Tensor
) -> tuple[fq.Circuit, float]:
    offsets = bands(matrix)
    if offsets != (-1, 0, 1) or bool(matrix.imag.any()):
        raise ValueError(
            "this banded encoding requires the real source tridiagonal profile"
        )
    widths = [float(matrix.diagonal(offset).abs().max()) for offset in offsets]
    alpha = sum(widths)
    weights = preparation(
        torch.tensor([(w / alpha) ** 0.5 for w in widths] + [0], dtype=torch.float64)
    )
    matrix_parts = []
    for offset, scale in zip(offsets, widths, strict=True):
        diagonal = torch.zeros(8, dtype=torch.float64)
        shift = torch.zeros((8, 8), dtype=CTYPE)
        for col in range(8):
            row = col - offset
            if 0 <= row < 8:
                diagonal[col] = matrix[row, col].real / scale
            shift[row % 8, col] = 1
        d = torch.diag(diagonal).to(CTYPE)
        complement = torch.diag((1 - diagonal.square()).clamp_min(0).sqrt()).to(CTYPE)
        dilated = torch.cat(
            (torch.cat((d, complement), 1), torch.cat((complement, -d), 1)), 0
        )
        both = torch.zeros((16, 16), dtype=CTYPE)
        both[:8, :8] = both[8:, 8:] = shift
        matrix_parts.append(both @ dilated)
    matrix_parts.append(torch.eye(16, dtype=CTYPE))
    select = torch.zeros((64, 64), dtype=CTYPE)
    for index, part in enumerate(matrix_parts):
        group = slice(16 * index, 16 * (index + 1))
        select[group, group] = part
    torch.testing.assert_close(
        select.mH @ select, torch.eye(64, dtype=CTYPE), atol=1e-12, rtol=0
    )
    circuit = (
        fq.Circuit(6, dtype=CTYPE)
        .compose(weights, qubits=(0, 1))
        .compose(preparation(rhs), qubits=(3, 4, 5))
    )
    circuit.any(*range(6), matrix=select)
    circuit.compose(weights.adjoint(), qubits=(0, 1))
    return circuit, alpha
