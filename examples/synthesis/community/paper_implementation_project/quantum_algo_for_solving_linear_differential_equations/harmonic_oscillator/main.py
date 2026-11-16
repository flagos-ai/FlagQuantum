"""Harmonic oscillator encoding, evolution and exact inverse composition."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    PHASE,
    Output,
    QArray,
    QBit,
    R,
    allocate,
    apply,
    control,
    inplace_prepare_amplitudes,
    qfunc,
    unitary,
    within,
)

from .physics import ENCODING

AMPLITUDES = (0.707106781186547, 0.707106781186547)


@qfunc
def encoding(ancilla: QArray[QBit], y: QBit) -> None:
    PHASE(math.pi, y)
    control(~y, lambda: unitary(ENCODING, ancilla))


@qfunc
def evolution(scale: float, ancilla: QArray[QBit], x: QBit) -> None:
    control(ancilla[0], lambda: R(-math.pi * scale, math.pi / 2, x))
    control(ancilla[1], lambda: R(-2 * math.pi * scale, math.pi / 2, x))
    control(ancilla[2], lambda: R(-4 * math.pi * scale, math.pi / 2, x))


@qfunc
def main(
    amplitudes: list[float],
    scale: float,
    x: Output[QBit],
    ancilla: Output[QArray[QBit]],
    y: Output[QBit],
) -> None:
    allocate(x)
    allocate(3, ancilla)
    allocate(y)
    inplace_prepare_amplitudes(amplitudes, 0.01, x)
    with within():
        encoding(ancilla, y)
    with apply():
        evolution(scale, ancilla, x)


def build_program(
    scale: float = 1.0, amplitudes: tuple[float, ...] = AMPLITUDES
) -> synth.Program:
    if len(amplitudes) != 2:
        raise ValueError("use two oscillator input amplitudes")
    return synth.synthesize(
        main,
        classical_inputs={"amplitudes": list(amplitudes), "scale": scale},
        max_workspace=2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x", "ancilla", "y"))))
    print(program.report["resources"])
