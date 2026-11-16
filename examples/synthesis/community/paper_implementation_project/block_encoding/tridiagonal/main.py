"""Tridiagonal block encoding with reversible preparation and a bounded RX table."""

import math

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    adjoint,
    allocate,
    control,
    hadamard_transform,
    inplace_prepare_amplitudes,
    phase,
    qft,
    qfunc,
)

VALUES = (
    0.8015,
    -0.8087,
    0.6652,
    -0.1181,
    -0.2983,
    0.6675,
    0.8014,
    0.6437,
    0.7776,
    0.3898,
    -0.7502,
    -0.2429,
    0.683,
    0.2305,
    0.1147,
    0.6802,
    0.7784,
    0.3091,
    -0.2962,
    -0.2641,
    0.7889,
    0.3176,
    0.1564,
    -0.4584,
    -0.3464,
    0.5023,
    -0.1425,
    -0.8012,
    0.811,
    0.0217,
    0.465,
    0.0,
)
ANGLES = tuple(2 * math.acos(x) for x in VALUES)
RAW = (0.5774, 0.5774, 0.0, 0.5774)
AMPLITUDES = tuple(x / math.hypot(*RAW) for x in RAW)


@qfunc
def prepare(amplitudes: list[float], selector: QNum) -> None:
    inplace_prepare_amplitudes(amplitudes, 0, selector)


@qfunc
def table_entry(index: int, angles: list[float], s: QNum, j: QNum, data: QBit) -> None:
    control((((s & 1) + 2 * j) ^ index) < 1, lambda: phase(data - 0.5, angles[index]))


@qfunc
def main(
    amplitudes: list[float],
    angles: list[float],
    s: Output[QNum],
    j: Output[QNum],
    data: Output[QBit],
) -> None:
    allocate(2, s)
    allocate(4, j)
    allocate(data)
    hadamard_transform(j)
    prepare(amplitudes, s)
    qft(j)
    control((s & 3) == 1, lambda: phase(j, -0.39269908169872414))
    with adjoint():
        qft(j)
    hadamard_transform(data)
    for index in range(32):
        table_entry(index, angles, s, j, data)
    hadamard_transform(data)
    qft(j)
    control((s & 2) == 2, lambda: phase(j, 0.39269908169872414))
    with adjoint():
        qft(j)
    with adjoint():
        prepare(amplitudes, s)


def build_program(
    angles: tuple[float, ...] = ANGLES, amplitudes: tuple[float, ...] = AMPLITUDES
) -> synth.Program:
    if len(angles) != 32 or len(amplitudes) != 4:
        raise ValueError("use 32 table angles and four normalized selector amplitudes")
    return synth.synthesize(
        main,
        classical_inputs={"amplitudes": list(amplitudes), "angles": list(angles)},
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    import flagquantum as fq

    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("s", "j", "data"))))
    print(program.report["resources"])
