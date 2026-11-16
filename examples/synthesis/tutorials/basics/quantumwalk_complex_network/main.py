"""Three-round nonuniform quantum walk with native controlled reflections."""

from math import pi

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    SWAP,
    Output,
    QArray,
    QBit,
    QNum,
    adjoint,
    allocate,
    apply,
    control,
    drop,
    hadamard_transform,
    inplace_prepare_state,
    phase,
    qfunc,
    within,
)


@qfunc
def unprepare(probabilities: list[float], y: QArray[QBit]) -> None:
    with adjoint():
        inplace_prepare_state(probabilities, 0, y)


@qfunc
def reflect(probabilities: list[float], y: QArray[QBit]) -> None:
    with within():
        unprepare(probabilities, y)
    with apply():
        control((y[0] == 0) & (y[1] == 0), lambda: phase(1, pi))
        phase(1, pi)


@qfunc
def prepare(x: QNum, y: QArray[QBit]) -> None:
    hadamard_transform(x)
    control(
        x == 0,
        lambda: inplace_prepare_state(
            [0.0, 0.333333333333333, 0.333333333333333, 0.333333333333333], 0, y
        ),
    )
    control(x == 1, lambda: inplace_prepare_state([0.5, 0.0, 0.5, 0.0], 0, y))
    control(x == 2, lambda: inplace_prepare_state([0.5, 0.5, 0.0, 0.0], 0, y))
    control(x == 3, lambda: inplace_prepare_state([1.0, 0.0, 0.0, 0.0], 0, y))


@qfunc
def coin(x: QNum, y: QArray[QBit]) -> None:
    control(
        x == 0,
        lambda: reflect(
            [0.0, 0.333333333333333, 0.333333333333333, 0.333333333333333], y
        ),
    )
    control(x == 1, lambda: reflect([0.5, 0.0, 0.5, 0.0], y))
    control(x == 2, lambda: reflect([0.5, 0.5, 0.0, 0.0], y))
    control(x == 3, lambda: reflect([1.0, 0.0, 0.0, 0.0], y))


@qfunc
def main(steps: int, angle: float, x: Output[QArray[QBit]]) -> None:
    y: QArray[QBit] = QArray()
    allocate(2, x)
    allocate(2, y)
    prepare(x, y)
    phase(x[0], angle)
    phase(y[1], -0.7 * angle)
    for _ in range(steps):
        coin(x, y)
        SWAP(x[0], y[0])
        SWAP(x[1], y[1])
    drop(y)


def build_program(steps: int = 3, angle: float = 0.0) -> synth.Program:
    if type(steps) is not int or not 0 <= steps <= 3:
        raise ValueError("Use a quantum walk length in 0..3")
    return synth.synthesize(
        main,
        classical_inputs={"steps": steps, "angle": angle},
        max_qubits=5,
        max_workspace=1,
        memory_limit_bytes=1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x",))))
    print(program.report["resources"])
