"""Full eight-layer, twelve-bit native induced two-color subgraph program."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def cost(gamma: float, a: QArray[QBit], b: QArray[QBit]) -> None:
    phase(6, gamma)
    for j in range(6):
        phase(-(a[j] * b[j] + (1 - a[j]) * (1 - b[j])), gamma)
    phase(
        -4
        * (
            a[0] * a[1]
            + a[0] * a[2]
            + a[0] * a[4]
            + a[1] * a[2]
            + a[1] * a[3]
            + a[1] * a[5]
            + a[2] * a[3]
            + a[2] * a[4]
            + a[2] * a[5]
            + a[3] * a[4]
            + a[4] * a[5]
        )
        ** 2,
        gamma,
    )
    phase(
        -4
        * (
            b[0] * b[1]
            + b[0] * b[2]
            + b[0] * b[4]
            + b[1] * b[2]
            + b[1] * b[3]
            + b[1] * b[5]
            + b[2] * b[3]
            + b[2] * b[4]
            + b[2] * b[5]
            + b[3] * b[4]
            + b[4] * b[5]
        )
        ** 2,
        gamma,
    )
    phase(
        -8
        * (
            a[0] * a[1]
            + a[0] * a[2]
            + a[0] * a[4]
            + a[1] * a[2]
            + a[1] * a[3]
            + a[1] * a[5]
            + a[2] * a[3]
            + a[2] * a[4]
            + a[2] * a[5]
            + a[3] * a[4]
            + a[4] * a[5]
        )
        * (
            b[0] * b[1]
            + b[0] * b[2]
            + b[0] * b[4]
            + b[1] * b[2]
            + b[1] * b[3]
            + b[1] * b[5]
            + b[2] * b[3]
            + b[2] * b[4]
            + b[2] * b[5]
            + b[3] * b[4]
            + b[4] * b[5]
        ),
        gamma,
    )


@qfunc
def main(params: list[float], a: Output[QArray[QBit]], b: Output[QArray[QBit]]) -> None:
    allocate(6, a)
    allocate(6, b)
    hadamard_transform(a)
    hadamard_transform(b)
    for layer in range(8):
        cost(params[layer], a, b)
        for j in range(6):
            RX(params[8 + layer], a[j])
            RX(params[8 + layer], b[j])


DEFAULT_PARAMS = tuple(0.003 * (i + 1) * (-1 if i % 2 else 1) for i in range(16))


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    """Bind all eight cost and mixer layers, retaining zero compiler workspace."""
    if len(params) != 16:
        raise ValueError(
            "Eight induced-subgraph layers require sixteen real parameters"
        )
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(program.report["resources"])
    print("State norm:", result.statevector().norm().item())
