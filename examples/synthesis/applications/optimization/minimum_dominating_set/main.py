"""Full native minimum dominating set with retained slack encodings."""

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
def cost(
    g: float,
    x: QArray[QBit],
    s0: QArray[QBit],
    s1: QArray[QBit],
    s2: QArray[QBit],
    s3: QArray[QBit],
    s4: QArray[QBit],
    s5: QArray[QBit],
) -> None:
    phase(-(x[0] + x[1] + x[2] + x[3] + x[4] + x[5]), g)
    phase(-10 * (s0[0] + 2 * s0[1] - x[0] - x[1] - x[3] - x[5] + 1) ** 2, g)
    phase(-10 * (s1[0] + 2 * s1[1] - x[0] - x[1] - x[2] - x[4] + 1) ** 2, g)
    phase(
        -10 * (s2[0] + 2 * s2[1] + s2[2] - x[1] - x[2] - x[3] - x[4] - x[5] + 1) ** 2, g
    )
    phase(-10 * (s3[0] + 2 * s3[1] - x[0] - x[2] - x[3] - x[4] + 1) ** 2, g)
    phase(
        -10 * (s4[0] + 2 * s4[1] + s4[2] - x[1] - x[2] - x[3] - x[4] - x[5] + 1) ** 2, g
    )
    phase(-10 * (s5[0] + 2 * s5[1] - x[0] - x[2] - x[4] - x[5] + 1) ** 2, g)


@qfunc
def main(
    params: list[float],
    x: Output[QArray[QBit]],
    s0: Output[QArray[QBit]],
    s1: Output[QArray[QBit]],
    s2: Output[QArray[QBit]],
    s3: Output[QArray[QBit]],
    s4: Output[QArray[QBit]],
    s5: Output[QArray[QBit]],
) -> None:
    allocate(6, x)
    allocate(2, s0)
    allocate(2, s1)
    allocate(3, s2)
    allocate(2, s3)
    allocate(3, s4)
    allocate(2, s5)
    hadamard_transform(x)
    hadamard_transform(s0)
    hadamard_transform(s1)
    hadamard_transform(s2)
    hadamard_transform(s3)
    hadamard_transform(s4)
    hadamard_transform(s5)
    for layer in range(6):
        cost(params[layer], x, s0, s1, s2, s3, s4, s5)
        for j in range(6):
            RX(params[6 + layer], x[j])
        for j in range(2):
            RX(params[6 + layer], s0[j])
        for j in range(2):
            RX(params[6 + layer], s1[j])
        for j in range(3):
            RX(params[6 + layer], s2[j])
        for j in range(2):
            RX(params[6 + layer], s3[j])
        for j in range(3):
            RX(params[6 + layer], s4[j])
        for j in range(2):
            RX(params[6 + layer], s5[j])


DEFAULT_PARAMS = tuple(0.003 * (i + 1) * (-1 if i % 2 else 1) for i in range(12))


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    """Bind every source cost/mixer layer without a source importer."""
    if len(params) != 12:
        raise ValueError("The retained program requires 12 real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())
