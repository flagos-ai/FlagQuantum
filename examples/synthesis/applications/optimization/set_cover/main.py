"""Full native set cover with retained slack encodings."""

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
    s6: QArray[QBit],
    s7: QArray[QBit],
    s8: QArray[QBit],
    s9: QArray[QBit],
) -> None:
    phase(-(x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]), g)
    phase(-10 * (s0[0] - x[0] - x[4] + 1) ** 2, g)
    phase(-10 * (s1[0] + s1[1] - x[0] - x[1] - x[7] + 1) ** 2, g)
    phase(-10 * (s2[0] + s2[1] - x[0] - x[1] - x[5] + 1) ** 2, g)
    phase(-10 * (s3[0] + s3[1] - x[0] - x[1] - x[6] + 1) ** 2, g)
    phase(-10 * (s4[0] - x[1] - x[7] + 1) ** 2, g)
    phase(-10 * (s5[0] - x[2] - x[4] + 1) ** 2, g)
    phase(-10 * (s6[0] + s6[1] - x[2] - x[5] - x[6] + 1) ** 2, g)
    phase(-10 * (s7[0] + s7[1] - x[3] - x[4] - x[7] + 1) ** 2, g)
    phase(-10 * (s8[0] - x[3] - x[5] + 1) ** 2, g)
    phase(-10 * (s9[0] - x[3] - x[6] + 1) ** 2, g)


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
    s6: Output[QArray[QBit]],
    s7: Output[QArray[QBit]],
    s8: Output[QArray[QBit]],
    s9: Output[QArray[QBit]],
) -> None:
    allocate(8, x)
    allocate(1, s0)
    allocate(2, s1)
    allocate(2, s2)
    allocate(2, s3)
    allocate(1, s4)
    allocate(1, s5)
    allocate(2, s6)
    allocate(2, s7)
    allocate(1, s8)
    allocate(1, s9)
    hadamard_transform(x)
    hadamard_transform(s0)
    hadamard_transform(s1)
    hadamard_transform(s2)
    hadamard_transform(s3)
    hadamard_transform(s4)
    hadamard_transform(s5)
    hadamard_transform(s6)
    hadamard_transform(s7)
    hadamard_transform(s8)
    hadamard_transform(s9)
    for layer in range(3):
        cost(params[layer], x, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9)
        for j in range(8):
            RX(params[3 + layer], x[j])
        for j in range(1):
            RX(params[3 + layer], s0[j])
        for j in range(2):
            RX(params[3 + layer], s1[j])
        for j in range(2):
            RX(params[3 + layer], s2[j])
        for j in range(2):
            RX(params[3 + layer], s3[j])
        for j in range(1):
            RX(params[3 + layer], s4[j])
        for j in range(1):
            RX(params[3 + layer], s5[j])
        for j in range(2):
            RX(params[3 + layer], s6[j])
        for j in range(2):
            RX(params[3 + layer], s7[j])
        for j in range(1):
            RX(params[3 + layer], s8[j])
        for j in range(1):
            RX(params[3 + layer], s9[j])


DEFAULT_PARAMS = tuple(0.003 * (i + 1) * (-1 if i % 2 else 1) for i in range(6))


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    """Bind every source cost/mixer layer without a source importer."""
    if len(params) != 6:
        raise ValueError("The retained program requires 6 real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())
