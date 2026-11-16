"""Toeplitz block encoding with reversible shifts and a pre-update overflow flag."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    adjoint,
    allocate,
    apply,
    control,
    hadamard_transform,
    phase,
    qft,
    qfunc,
    within,
)

VALUES = (0.6406, 0.6246, -0.3873, 0.952)
ANGLES = tuple(2 * math.acos(value) for value in VALUES)


@qfunc
def table_entry(index: int, angle: float, selector: QNum, data: QBit) -> None:
    control((selector ^ index) < 1, lambda: phase(data - 0.5, angle))


@qfunc
def mark_overflow(selector: QNum, position: QNum, flag: QBit) -> None:
    hadamard_transform(flag)
    control(selector + position < 1, lambda: phase(flag, math.pi))
    control(selector + position > 16, lambda: phase(flag, math.pi))
    hadamard_transform(flag)


@qfunc
def main(
    angles: list[float],
    data: Output[QBit],
    s: Output[QNum],
    j: Output[QNum],
    dlt: Output[QBit],
) -> None:
    allocate(data)
    allocate(2, s)
    allocate(4, j)
    allocate(dlt)
    hadamard_transform(j)
    with within():
        hadamard_transform(s)
    with apply():
        hadamard_transform(data)
        for index in range(4):
            table_entry(index, angles[index], s, data)
        hadamard_transform(data)
        mark_overflow(s, j, dlt)
        qft(j)
        phase((s - 1) * j, math.tau / 16)
        with adjoint():
            qft(j)


def build_program(angles: tuple[float, ...] = ANGLES) -> synth.Program:
    if len(angles) != 4:
        raise ValueError("use all four Toeplitz rotation angles")
    return synth.synthesize(
        main, classical_inputs={"angles": list(angles)}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("data", "s", "j", "dlt"))))
    print(program.report["resources"])
