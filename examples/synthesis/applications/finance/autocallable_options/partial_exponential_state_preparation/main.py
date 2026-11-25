"""Prepare an exponential input, amplify its upper interval and shift modulo 32."""

import math
from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    R,
    adjoint,
    allocate,
    control,
    inplace_prepare_amplitudes,
    phase,
    qfunc,
)

SUCCESS_AMPLITUDE = 0.99795457844373736


def exponential_amplitudes(rate: float, width: int) -> list[float]:
    """Construct normalized input amplitudes proportional to exp(-rate*word/2)."""
    if type(rate) not in (int, float) or not math.isfinite(rate) or abs(rate) > 16:
        raise ValueError("rate must be a finite real in [-16, 16]")
    if type(width) is not int or not 1 <= width <= 8:
        raise ValueError("width must be an integer in 1..8")
    maximum = max(0.0, -rate * ((1 << width) - 1))
    probabilities = [math.exp(-rate * word - maximum) for word in range(1 << width)]
    total = math.fsum(probabilities)
    return [math.sqrt(value / total) for value in probabilities]


@qfunc
def prepare(amplitudes: list[float], x: QNum) -> None:
    inplace_prepare_amplitudes(amplitudes, 0, x)


@qfunc
def reflect_zero(x: QNum, attenuation: QBit) -> None:
    control(x == 0, lambda: phase(1 - attenuation, math.pi))
    phase(1, math.pi)


@qfunc
def shift(x: QNum) -> None:
    x -= 18


@qfunc
def main(
    amplitudes: list[float],
    beta: float,
    x: Output[Annotated[QNum, 5]],
    attenuation: Output[QBit],
) -> None:
    allocate(x)
    allocate(attenuation)
    prepare(amplitudes, x)
    R(beta, math.pi / 2, attenuation)
    control(x >= 21, lambda: phase(1 - attenuation, math.pi))
    R(-beta, math.pi / 2, attenuation)
    with adjoint():
        prepare(amplitudes, x)
    reflect_zero(x, attenuation)
    prepare(amplitudes, x)
    R(beta, math.pi / 2, attenuation)
    shift(x)


def classical_inputs(
    rate: float = -0.5, success_amplitude: float = SUCCESS_AMPLITUDE
) -> dict[str, list[float] | float]:
    amplitudes = exponential_amplitudes(rate, 5)
    observed = math.sqrt(math.fsum(value * value for value in amplitudes[21:]))
    if type(success_amplitude) not in (int, float) or not math.isfinite(
        success_amplitude
    ):
        raise ValueError("success amplitude must be a finite real")
    if not 0.5 <= success_amplitude <= 1 or abs(success_amplitude - observed) > 1e-12:
        raise ValueError(
            "declared success amplitude must match the observed overlap in [0.5, 1]"
        )
    return {"amplitudes": amplitudes, "beta": 2 * math.acos(0.5 / success_amplitude)}


def build_program(
    rate: float = -0.5, success_amplitude: float = SUCCESS_AMPLITUDE
) -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs=classical_inputs(rate, success_amplitude),
        max_qubits=8,
        max_workspace=2,
        memory_limit_bytes=1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
    print(program.report["resources"])
