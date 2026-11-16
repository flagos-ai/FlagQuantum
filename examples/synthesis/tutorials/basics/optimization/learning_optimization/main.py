"""Use explicit unsigned numeric ports and a quantum slack register."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
    suzuki_trotter,
)

MIXER2 = fq.X(0) + fq.X(1)
MIXER4 = fq.X(0) + fq.X(1) + fq.X(2) + fq.X(3)


@qfunc
def main(
    params: list[float], x0: Output[QNum], x1: Output[QNum], slack: Output[QArray[QBit]]
) -> None:
    allocate(2, x0)
    allocate(2, x1)
    allocate(4, slack)
    hadamard_transform(x0)
    hadamard_transform(x1)
    hadamard_transform(slack)
    phase(
        -(
            3 * x0
            + 2 * x1
            + 20
            * (
                slack[0]
                + 2 * slack[1]
                + 4 * slack[2]
                + 3 * slack[3]
                - 3.0 * x0
                - x1
                + 2.0
            )
            ** 2
        ),
        params[0],
    )
    # Commuting X terms reproduce each source RX(beta), including its phase.
    suzuki_trotter(MIXER2, params[1] / 2, 1, 1, x0)
    suzuki_trotter(MIXER2, params[1] / 2, 1, 1, x1)
    suzuki_trotter(MIXER4, params[1] / 2, 1, 1, slack)


def build_program(params: tuple[float, ...] = (0.01, 0.02)) -> synth.Program:
    if len(params) != 2:
        raise ValueError("Constraint optimization requires two real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x0", "x1", "slack"))))
