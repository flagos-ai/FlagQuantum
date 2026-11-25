"""Native retained Gaussian QSVT, branch phases and exact amplification."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CRX,
    RX,
    RZ,
    H,
    Output,
    QArray,
    QBit,
    R,
    X,
    adjoint,
    allocate,
    control,
    phase,
    qfunc,
)

from .data import EVEN_PHASES, ODD_PHASES, SINE_ANGLES, SUCCESS_AMPLITUDE


@qfunc
def u_sin(angles: list[float], data: QArray[QBit], signal: QBit) -> None:
    RZ(-math.pi / 2, signal)
    for i in range(8):
        CRX(angles[i], data[i], signal)
    RZ(math.pi / 2, signal)
    X(signal)


@qfunc
def u_back(angles: list[float], data: QArray[QBit], signal: QBit) -> None:
    with adjoint():
        u_sin(angles, data, signal)


@qfunc
def selected_back(
    angles: list[float], data: QArray[QBit], signal: QBit, selector: QBit
) -> None:
    control(selector == 0, lambda: X(signal))
    RX(math.pi / 2, signal)
    for i in range(8):
        phase((2 * signal - 1) * data[i] * (1 - selector), -angles[i] / 2)
    RX(-math.pi / 2, signal)


@qfunc
def qsvt_lcu(
    even: list[float],
    odd: list[float],
    angles: list[float],
    data: QArray[QBit],
    signal: QBit,
    aux: QBit,
    selector: QBit,
) -> None:
    H(aux)
    phase(
        (0.5 - signal - aux + 2 * signal * aux)
        * (even[0] + (odd[0] - even[0]) * selector),
        1,
    )
    for i in range(14):
        u_sin(angles, data, signal)
        phase(
            (0.5 - signal - aux + 2 * signal * aux)
            * (even[2 * i + 1] + (odd[2 * i + 1] - even[2 * i + 1]) * selector),
            1,
        )
        if i < 13:
            u_back(angles, data, signal)
            phase(
                (0.5 - signal - aux + 2 * signal * aux)
                * (even[2 * i + 2] + (odd[2 * i + 2] - even[2 * i + 2]) * selector),
                1,
            )
    selected_back(angles, data, signal, selector)
    phase((0.5 - signal - aux + 2 * signal * aux) * (1 - selector), even[28])
    H(aux)


@qfunc
def prepare(
    even: list[float],
    odd: list[float],
    angles: list[float],
    data: QArray[QBit],
    signal: QBit,
    aux: QBit,
    selector: QBit,
) -> None:
    for i in range(8):
        H(data[i])
    H(selector)
    qsvt_lcu(even, odd, angles, data, signal, aux, selector)
    H(selector)
    H(selector)


@qfunc
def reflect_zero(
    data: QArray[QBit],
    signal: QBit,
    aux: QBit,
    selector: QBit,
    attenuation: QBit,
    marker: QBit,
) -> None:
    for i in range(8):
        X(data[i])
    control(data, lambda: X(marker))
    phase(
        marker * (1 - signal) * (1 - aux) * (1 - selector) * (1 - attenuation), math.pi
    )
    control(data, lambda: X(marker))
    for i in range(8):
        X(data[i])
    phase(1, math.pi)


@qfunc
def main(
    even: list[float],
    odd: list[float],
    angles: list[float],
    beta: float,
    data: Output[QArray[QBit]],
    signal: Output[QBit],
    aux: Output[QBit],
    selector: Output[QBit],
    attenuation: Output[QBit],
    marker: Output[QBit],
) -> None:
    allocate(8, data)
    allocate(signal)
    allocate(aux)
    allocate(selector)
    allocate(attenuation)
    allocate(marker)
    prepare(even, odd, angles, data, signal, aux, selector)
    R(beta, math.pi / 2, attenuation)
    for _ in range(2):
        phase((1 - signal) * (1 - aux) * (1 - selector) * (1 - attenuation), math.pi)
        R(-beta, math.pi / 2, attenuation)
        with adjoint():
            prepare(even, odd, angles, data, signal, aux, selector)
        reflect_zero(data, signal, aux, selector, attenuation, marker)
        prepare(even, odd, angles, data, signal, aux, selector)
        R(beta, math.pi / 2, attenuation)


def classical_inputs() -> dict[str, list[float] | float]:
    return {
        "even": list(EVEN_PHASES),
        "odd": list(ODD_PHASES),
        "angles": list(SINE_ANGLES),
        "beta": 2 * math.acos(math.sin(math.pi / 10) / SUCCESS_AMPLITUDE),
    }


def build_program(success_amplitude: float = SUCCESS_AMPLITUDE) -> synth.Program:
    """Use the previously approved overlap; never correct source parameters silently."""
    if type(success_amplitude) not in (int, float) or not math.isfinite(
        success_amplitude
    ):
        raise ValueError("success amplitude must be a finite real")
    if abs(success_amplitude - SUCCESS_AMPLITUDE) > 1e-12:
        raise ValueError("success amplitude disagrees with the retained phase sequence")
    return synth.synthesize(
        main,
        classical_inputs=classical_inputs(),
        max_qubits=18,
        max_workspace=5,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("data",))))
    print(program.report["resources"])
