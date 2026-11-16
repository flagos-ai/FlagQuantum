"""Complete native three-layer portfolio with retained numeric encodings."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
    suzuki_trotter,
)

MIXER = fq.X(0) + fq.X(1)
SLACK_MIXER = fq.X(0) + fq.X(1) + fq.X(2)


@qfunc
def main(
    params: list[float],
    a0: Output[QNum],
    a1: Output[QNum],
    a2: Output[QNum],
    slack: Output[QNum],
) -> None:
    allocate(2, a0)
    allocate(2, a1)
    allocate(2, a2)
    hadamard_transform(a0)
    hadamard_transform(a1)
    hadamard_transform(a2)
    allocate(3, slack)
    hadamard_transform(slack)
    for i in range(3):
        phase(
            1.0
            * (
                a0 * (0.9 * a0 + 0.5 * a1 - 0.7 * a2)
                + a1 * (0.5 * a0 + 0.9 * a1 - 0.2 * a2)
                + a2 * (-0.7 * a0 - 0.2 * a1 + 0.9 * a2)
                - 10.0 * (3 * a0 + 4 * a1 - a2)
                + 30.5 * (a0 + a1 + a2 + slack - 6) ** 2
            ),
            params[2 * i],
        )
        suzuki_trotter(MIXER, params[2 * i + 1] / 2, 1, 1, a0)
        suzuki_trotter(MIXER, params[2 * i + 1] / 2, 1, 1, a1)
        suzuki_trotter(MIXER, params[2 * i + 1] / 2, 1, 1, a2)
        suzuki_trotter(SLACK_MIXER, params[2 * i + 1] / 2, 1, 1, slack)


DEFAULT_PARAMS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 6:
        raise ValueError("Three portfolio layers require six real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(program.report["resources"])
    print("State norm:", result.statevector().norm().item())
