"""Two-dimensional Laplacian with explicit coordinates and reversible preparation."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Output,
    QArray,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    apply,
    control,
    hadamard_transform,
    inplace_prepare_state,
    phase,
    qft,
    qfunc,
    within,
)

PROBABILITIES = (0.2, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0)
ANGLES = (2 * math.acos(-1.0), 2 * math.acos(0.25), 2 * math.acos(0.25))


@qfunc
def prepare(probabilities: list[float], s: QArray[QBit]) -> None:
    inplace_prepare_state(probabilities, 0, s)


@qfunc
def main(
    probabilities: list[float],
    angles: list[float],
    initial_data: int,
    initial_flag: int,
    s: Output[QArray[QBit]],
    j1: Output[QNum],
    j0: Output[QNum],
    data: Output[QBit],
    dlt: Output[QBit],
) -> None:
    allocate(3, s)
    allocate(2, j1)
    allocate(2, j0)
    allocate(data)
    allocate(dlt)
    if initial_data == 1:
        X(data)
    if initial_flag == 1:
        X(dlt)
    hadamard_transform(j1)
    hadamard_transform(j0)
    with within():
        prepare(probabilities, s)
    with apply():
        qft(j1)
        phase((1 - s[0]) * s[2] * j1, math.tau / 4)
        with adjoint():
            qft(j1)
        qft(j1)
        control(j0 > 2, lambda: phase((1 - s[0]) * s[1] * j1, math.tau / 4))
        with adjoint():
            qft(j1)
        qft(j0)
        phase((1 - s[0]) * s[1] * j0, math.tau / 4)
        with adjoint():
            qft(j0)
        hadamard_transform(dlt)
        phase(s[0] * (1 - s[1]) * (1 - s[2]) * dlt, math.pi)
        control(j0 < 1, lambda: phase(s[1] * dlt, math.pi))
        control(j1 < 1, lambda: phase(s[2] * dlt, math.pi))
        hadamard_transform(dlt)
        phase(data, math.pi)
        hadamard_transform(data)
        phase(
            (
                angles[0] * (1 - s[1]) * (1 - s[2])
                + angles[1] * s[1] * (1 - s[2])
                + angles[2] * (1 - s[1]) * s[2]
            )
            * (data - 0.5),
            1,
        )
        hadamard_transform(data)
        CX(s[1], s[0])
        CX(s[2], s[0])
        qft(j1)
        control(j0 < 1, lambda: phase((1 - s[0]) * s[1] * j1, -math.tau / 4))
        with adjoint():
            qft(j1)
        qft(j0)
        phase((1 - s[0]) * s[1] * j0, -math.tau / 4)
        with adjoint():
            qft(j0)
        qft(j1)
        phase((1 - s[0]) * s[2] * j1, -math.tau / 4)
        with adjoint():
            qft(j1)


def build_program(
    angles: tuple[float, ...] = ANGLES,
    probabilities: tuple[float, ...] = PROBABILITIES,
    initial_data: int = 0,
    initial_flag: int = 0,
) -> synth.Program:
    if len(angles) != 3 or len(probabilities) != 8:
        raise ValueError("use three rotation angles and eight selector probabilities")
    if type(initial_data) is not int or initial_data not in (0, 1):
        raise ValueError("initial_data must be a classical bit")
    if type(initial_flag) is not int or initial_flag not in (0, 1):
        raise ValueError("initial_flag must be a classical bit")
    return synth.synthesize(
        main,
        classical_inputs={
            "probabilities": list(probabilities),
            "angles": list(angles),
            "initial_data": initial_data,
            "initial_flag": initial_flag,
        },
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(program.probabilities(result, registers=("s", "j1", "j0", "data", "dlt")))
    )
    print(program.report["resources"])
