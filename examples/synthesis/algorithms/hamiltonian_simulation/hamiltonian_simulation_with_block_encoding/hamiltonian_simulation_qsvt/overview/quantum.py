"""Compose the original two QSVT streams without packing their response matrix."""

import cmath
import math

import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    PhaseFit,
)
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QArray,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    apply,
    control,
    inplace_prepare_amplitudes,
    phase,
    prepare_amplitudes,
    qfunc,
    within,
)


@qfunc
def projector_rotation(angle: float, branch: QBit, selection: QNum[2]) -> None:
    phase(1, -angle)
    control(selection == 0, lambda: phase(1, 2 * angle))
    phase(branch, 2 * angle)
    control(selection == 0, lambda: phase(branch, -4 * angle))


@qfunc
def mixed_rotation(
    even: float, odd: float, parity: QBit, branch: QBit, selection: QNum[2]
) -> None:
    projector_rotation(even, branch, selection)
    control(parity, lambda: projector_rotation(odd - even, branch, selection))


@qfunc
def pauli_pair(data: QArray[QBit], code: int) -> None:
    if code == 1:
        X(data[0])
        X(data[1])
    if code == 3:
        phase(data[0], math.pi)
        phase(data[1], math.pi)


@qfunc
def signal(weights: list[float], selection: QNum[2], data: QArray[QBit]) -> None:
    with within():
        inplace_prepare_amplitudes(weights, 0, selection)
    with apply():
        # The source words are II, ZI, XX, ZZ; data[0] is the low bit.
        control(selection == 1, lambda: phase(data[1], math.pi))
        control(selection == 2, lambda: pauli_pair(data, 1))
        control(selection == 3, lambda: pauli_pair(data, 3))


@qfunc
def main(
    block_weights: list[float],
    even: list[float],
    odd: list[float],
    weights: list[float],
    amplitudes: list[float],
    parity: Output[QBit],
    branch: Output[QBit],
    selection: Output[QNum[2]],
    data: Output[QArray[QBit]],
) -> None:
    allocate(parity)
    allocate(branch)
    allocate(selection)
    prepare_amplitudes(amplitudes, 0, data)
    H(parity)
    H(branch)
    mixed_rotation(even[0], odd[0], parity, branch, selection)
    for pair in range(16):
        with adjoint():
            signal(block_weights, selection, data)
        mixed_rotation(even[2 * pair + 1], odd[2 * pair + 1], parity, branch, selection)
        signal(block_weights, selection, data)
        mixed_rotation(even[2 * pair + 2], odd[2 * pair + 2], parity, branch, selection)
    with adjoint():
        control(parity, lambda: signal(block_weights, selection, data))
    control(parity, lambda: projector_rotation(odd[33], branch, selection))
    phase(1, weights[0])
    phase(branch, weights[1] - weights[0])
    phase(parity, weights[2] - weights[0])
    phase(branch * parity, weights[3] - weights[2] - weights[1] + weights[0])
    H(branch)
    H(parity)


def build_program(
    cosine: PhaseFit,
    sine: PhaseFit,
    amplitudes: torch.Tensor,
    *,
    max_qubits: int = 10,
    max_workspace: int = 4,
    max_operations: int = 3500,
) -> synth.Program:
    """Compose both full phase streams and the source four-term Pauli block.

    No QSVT response matrix is packed into a gate. Native control may allocate
    clean work bits; its resource report is separate from the packed Circuit.
    """
    from .workflow import ALPHA, WEIGHTS, selection_weights, shifted_phases

    even = shifted_phases(cosine).tolist()
    odd = shifted_phases(sine).tolist()
    if len(even) != 33 or len(odd) != 34:
        raise ValueError("retain the source's 33/34 phase sequences")
    weights = (*selection_weights(32), *(-1j * w for w in selection_weights(33)))
    weight_angles = tuple(cmath.phase(weight) for weight in weights)

    return synth.synthesize(
        main,
        classical_inputs={
            "block_weights": [(value / ALPHA) ** 0.5 for value in WEIGHTS],
            "even": even,
            "odd": odd,
            "weights": weight_angles,
            "amplitudes": tuple(float(value) for value in amplitudes),
        },
        max_qubits=max_qubits,
        max_workspace=max_workspace,
        max_operations=max_operations,
        memory_limit_bytes=1024**2,
    )
