"""Complete native integer optimization with explicit numeric and slack ports."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QBit,
    QNum,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
    suzuki_trotter,
)

MIXER = fq.X(0) + fq.X(1)


@qfunc
def main(
    params: list[float],
    x0: Output[QNum],
    x1: Output[QNum],
    x2: Output[QNum],
    s0: Output[QBit],
    s1: Output[QBit],
) -> None:
    allocate(2, x0)
    allocate(2, x1)
    allocate(2, x2)
    allocate(s0)
    allocate(s1)
    hadamard_transform(x0)
    hadamard_transform(x1)
    hadamard_transform(x2)
    hadamard_transform(s0)
    hadamard_transform(s1)
    for layer in range(3):
        phase(
            -(
                10 * x0 * x1
                + 10 * x0 * x2
                - x0
                + 10 * x1 * x2
                - 2 * x1
                - 3 * x2
                + 10 * (s0 + x0 + x1 + x2 - 1.0) ** 2
                + 10 * (s1 + x0 + x1 + x2 - 1.0) ** 2
            ),
            params[layer],
        )
        suzuki_trotter(MIXER, params[3 + layer] / 2, 1, 1, x0)
        suzuki_trotter(MIXER, params[3 + layer] / 2, 1, 1, x1)
        suzuki_trotter(MIXER, params[3 + layer] / 2, 1, 1, x2)
        RX(params[3 + layer], s0)
        RX(params[3 + layer], s1)


DEFAULT_PARAMS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 6:
        raise ValueError(
            "Three integer-optimization layers require six real parameters"
        )
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())
