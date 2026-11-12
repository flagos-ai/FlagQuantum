"""Native constrained Knapsack quantum program with typed numeric storage."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
    suzuki_trotter,
)

MIXER_A = fq.X(0) + fq.X(1) + fq.X(2)
MIXER_B = fq.X(0) + fq.X(1)


@qfunc
def apply_cost(gamma: float, a: QNum, b: QNum) -> None:
    control(2 * a + 3 * b <= 12, lambda: phase(-3 * a - 5 * b, gamma))


@qfunc
def ansatz(params: list[float], a: QNum, b: QNum) -> None:
    for i in range(3):
        apply_cost(params[i], a, b)
        # exp(-i*beta*X/2) is RX(beta); these X terms commute exactly.
        suzuki_trotter(MIXER_A, params[3 + i] / 2, 1, 1, a)
        suzuki_trotter(MIXER_B, params[3 + i] / 2, 1, 1, b)


@qfunc
def main(params: list[float], a: Output[QNum], b: Output[QNum]) -> None:
    allocate(3, a)
    allocate(2, b)
    hadamard_transform(a)
    hadamard_transform(b)
    ansatz(params, a, b)


def build_program(
    params: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
) -> synth.Program:
    if len(params) != 6:
        raise ValueError("Three-layer Knapsack requires six real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("a", "b"))))
