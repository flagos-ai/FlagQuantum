"""Native block compositions and independent Pauli/banded Circuit references."""

import cmath
import math

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    X,
    allocate,
    apply,
    control,
    inplace_prepare_amplitudes,
    phase,
    prepare_amplitudes,
    qfunc,
    select_rotation,
    within,
)

from .model import CTYPE, bands, pauli_expansion


@qfunc
def prepare(amplitudes: list[float], q: Output[QArray[QBit]]) -> None:
    prepare_amplitudes(amplitudes, 0, q)


@qfunc
def pauli_word(codes: list[float], term: int, data: QArray[QBit]) -> None:
    for bit in range(3):
        if codes[3 * term + bit] == 1:
            X(data[bit])
        if codes[3 * term + bit] == 2:
            phase(data[bit], -math.pi / 2)
            X(data[bit])
            phase(data[bit], math.pi / 2)
        if codes[3 * term + bit] == 3:
            phase(data[bit], math.pi)


@qfunc
def pauli_block(
    amplitudes: list[float],
    weights: list[float],
    codes: list[float],
    angles: list[float],
    count: int,
    selection: Output[QNum[5]],
    data: Output[QArray[QBit]],
) -> None:
    prepare_amplitudes(amplitudes, 0, data)
    allocate(selection)
    with within():
        inplace_prepare_amplitudes(weights, 0, selection)
    with apply():
        # qfunc captures each lambda during static loop expansion, never later.
        for term in range(count):
            if codes[3 * term] + codes[3 * term + 1] + codes[3 * term + 2] > 0:
                control(
                    selection == term,
                    lambda: pauli_word(codes, term, data),  # noqa: B023
                )
            control(selection == term, lambda: phase(1, angles[term]))  # noqa: B023


def native_pauli_program(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    max_qubits: int = 12,
    max_workspace: int = 4,
) -> tuple[synth.Program, float]:
    """Compose the original LCU on eight outputs plus four clean work bits."""
    terms = pauli_expansion(matrix)
    if not terms or len(terms) > 32:
        raise ValueError(
            "the native five-bit selector requires 1 through 32 Pauli terms"
        )
    alpha = sum(abs(weight) for _, weight in terms)
    weights = [(abs(weight) / alpha) ** 0.5 for _, weight in terms]
    weights += [0.0] * (32 - len(weights))
    codes = [float("IXYZ".index(name)) for label, _ in terms for name in label[::-1]]
    program = synth.synthesize(
        pauli_block,
        classical_inputs={
            "amplitudes": tuple(float(value) for value in rhs),
            "weights": weights,
            "codes": codes,
            "angles": [cmath.phase(weight) for _, weight in terms],
            "count": len(terms),
        },
        max_qubits=max_qubits,
        max_workspace=max_workspace,
        max_operations=10_000,
        memory_limit_bytes=1024**2,
    )
    return program, alpha


@qfunc
def banded_term(angles: list[float], term: int, flag: QBit, data: QNum[3]) -> None:
    # RY(2*acos(d)) Z dilates each real signed diagonal value d.
    phase(flag, math.pi)
    select_rotation(
        "Y",
        [
            angles[8 * term],
            angles[8 * term + 1],
            angles[8 * term + 2],
            angles[8 * term + 3],
            angles[8 * term + 4],
            angles[8 * term + 5],
            angles[8 * term + 6],
            angles[8 * term + 7],
        ],
        data,
        flag,
    )
    if term == 0:
        data += 1
    if term == 2:
        data -= 1


@qfunc
def banded_block(
    amplitudes: list[float],
    weights: list[float],
    angles: list[float],
    selection: Output[QNum[2]],
    flag: Output[QBit],
    data: Output[QNum[3]],
) -> None:
    allocate(selection)
    allocate(flag)
    prepare_amplitudes(amplitudes, 0, data)
    with within():
        inplace_prepare_amplitudes(weights, 0, selection)
    with apply():
        for term in range(3):
            control(
                selection == term,
                lambda: banded_term(angles, term, flag, data),  # noqa: B023
            )


def native_banded_program(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    max_qubits: int = 8,
    max_workspace: int = 2,
    max_operations: int = 512,
) -> tuple[synth.Program, float]:
    """Compose the real source tridiagonal dilation with logical registers."""
    if matrix.shape != (8, 8) or bands(matrix) != (-1, 0, 1) or bool(matrix.imag.any()):
        raise ValueError(
            "the native banded encoding requires the real eight-row tridiagonal profile"
        )
    offsets = (-1, 0, 1)
    widths = [float(matrix.diagonal(offset).abs().max()) for offset in offsets]
    alpha = sum(widths)
    angles = []
    for offset, scale in zip(offsets, widths, strict=True):
        for column in range(8):
            row = column - offset
            diagonal = float(matrix[row, column].real / scale) if 0 <= row < 8 else 0.0
            angles.append(2 * math.acos(diagonal))
    program = synth.synthesize(
        banded_block,
        classical_inputs={
            "amplitudes": tuple(float(value) for value in rhs),
            "weights": [(width / alpha) ** 0.5 for width in widths] + [0.0],
            "angles": angles,
        },
        max_qubits=max_qubits,
        max_workspace=max_workspace,
        max_operations=max_operations,
        memory_limit_bytes=1024**2,
    )
    return program, alpha


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
