"""Native signed three-bit phase encoding and gradient estimation."""

import math
from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    adjoint,
    allocate,
    apply,
    drop,
    hadamard_transform,
    inplace_prepare_amplitudes,
    phase,
    qft,
    qfunc,
    within,
)


@qfunc
def phase_oracle(coefficients: list[float], x: QNum) -> None:
    phase(coefficients[0] * x * x + coefficients[1] * x + coefficients[2], 2 * math.pi)


@qfunc
def phase_only(
    coefficients: list[float], x: Output[Annotated[QNum, (3, True, 0)]]
) -> None:
    allocate(x)
    hadamard_transform(x)
    phase_oracle(coefficients, x)


@qfunc
def gradient_1d(
    coefficients: list[float], x: Output[Annotated[QNum, (3, True, 0)]]
) -> None:
    allocate(x)
    hadamard_transform(x)
    phase_oracle(coefficients, x)
    with adjoint():
        qft(x)


@qfunc
def auxiliary_phase(x: Output[Annotated[QNum, (3, True, 0)]]) -> None:
    allocate(x)
    hadamard_transform(x)
    ancilla: Annotated[QNum, (3, True, 3)] = QNum()
    allocate(ancilla)
    with within():
        inplace_prepare_amplitudes([0, 0, 0, 0, 0, 0, 0, 1], 0, ancilla)
        qft(ancilla)
    with apply():
        with within():
            qft(ancilla)
        with apply():
            phase((0.25 * x + 2) * ancilla, 16 * math.pi)
    drop(ancilla)


@qfunc
def gradient_2d(
    coefficients: list[float],
    x: Output[Annotated[QNum, (3, True, 0)]],
    y: Output[Annotated[QNum, (3, True, 0)]],
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    phase(
        coefficients[0] * x * x
        + coefficients[1] * y * y
        + coefficients[2] * x * y
        + coefficients[3] * x
        + coefficients[4] * y
        + coefficients[5],
        2 * math.pi,
    )
    with adjoint():
        qft(x)
        qft(y)


def build_program(
    coefficients: tuple[float, ...], *, inverse_fourier: bool = True
) -> synth.Program:
    return synth.synthesize(
        gradient_1d if inverse_fourier else phase_only,
        classical_inputs={"coefficients": list(coefficients)},
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
