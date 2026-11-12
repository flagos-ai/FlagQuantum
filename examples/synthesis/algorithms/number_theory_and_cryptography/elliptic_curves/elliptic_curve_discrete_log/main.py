"""Native coherent scalar addition on an elliptic curve over a finite field."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    Output,
    QArray,
    QBit,
    X,
    adjoint,
    allocate,
    apply,
    control,
    hadamard_transform,
    qft,
    qfunc,
    within,
)

from .curve import encode, translation_steps


@qfunc
def translation(steps: list[int], point: QArray[QBit]) -> None:
    for i in range(0, len(steps), 6):
        with within():
            if steps[i] != 5:
                CX(point[steps[i]], point[5])
                CX(point[5], point[steps[i]])
                CX(point[steps[i]], point[5])
            for j in range(5):
                if steps[i + 1 + j] == 0:
                    X(point[j])
        with apply():
            control(point[:5], lambda: X(point[5]))


@qfunc
def scalar_add(
    g0: list[int],
    g1: list[int],
    g2: list[int],
    point: QArray[QBit],
    scalar: QArray[QBit],
) -> None:
    control(scalar[0], lambda: translation(g0, point))
    control(scalar[1], lambda: translation(g1, point))
    control(scalar[2], lambda: translation(g2, point))


@qfunc
def main(
    seed: list[int],
    g0: list[int],
    g1: list[int],
    g2: list[int],
    x1: Output[QArray[QBit]],
    x2: Output[QArray[QBit]],
    point: Output[QArray[QBit]],
) -> None:
    allocate(3, x1)
    allocate(3, x2)
    allocate(6, point)
    for bit in range(6):
        if seed[bit] == 1:
            X(point[bit])
    hadamard_transform(x1)
    hadamard_transform(x2)
    scalar_add(g0, g1, g2, point, x1)
    scalar_add(g0, g1, g2, point, x2)
    with adjoint():
        qft(x1)
        qft(x2)


def build_program(seed: tuple[int, int] = (4, 2)) -> synth.Program:
    if seed not in ((4, 2), (3, 2)):
        raise ValueError("seed must belong to a retained nonsingular source orbit")
    steps = [translation_steps(point) for point in ((0, 5), (2, 1), (0, 2))]
    return synth.synthesize(
        main,
        classical_inputs={
            "seed": [(encode(seed) >> i) & 1 for i in range(6)],
            "g0": steps[0],
            "g1": steps[1],
            "g2": steps[2],
        },
        max_qubits=15,
        max_workspace=3,
        memory_limit_bytes=64 * 1024**2,
    )


def run_example() -> None:
    program = build_program()
    probabilities = program.probabilities(
        fq.run(program), registers=("x1", "x2", "point")
    )
    print(
        {
            (a / 8, b / 8, word % 8, word // 8): probability
            for (a, b, word), probability in probabilities.items()
        }
    )
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
