"""Four-layer native electric-grid cost program with explicit field ports."""

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
def cost_layer(
    g: float,
    a: QArray[QBit],
    b: QArray[QBit],
    c: QArray[QBit],
    sa: QArray[QBit],
    sb: QArray[QBit],
    sc: QArray[QBit],
) -> None:
    phase(
        -(
            0.5 * a[0]
            + a[1]
            + a[2]
            + 2.1 * a[3]
            + b[0]
            + 0.6 * b[1]
            + 1.4 * b[2]
            + b[3]
            + c[0]
            + 1.4 * c[1]
            + 0.4 * c[2]
            + 2.3 * c[3]
        ),
        g,
    )
    for j in range(4):
        phase(-10 * (a[j] + b[j] + c[j] - 1) ** 2, g)
    phase(
        -10
        * (
            (sa[0] + sa[1] + a[0] + a[1] + a[2] + a[3] - 2) ** 2
            + (sb[0] + sb[1] + b[0] + b[1] + b[2] + b[3] - 2) ** 2
            + (sc[0] + sc[1] + c[0] + c[1] + c[2] + c[3] - 2) ** 2
        ),
        g,
    )


@qfunc
def main(
    params: list[float],
    a: Output[QArray[QBit]],
    b: Output[QArray[QBit]],
    c: Output[QArray[QBit]],
    sa: Output[QArray[QBit]],
    sb: Output[QArray[QBit]],
    sc: Output[QArray[QBit]],
) -> None:
    allocate(4, a)
    allocate(4, b)
    allocate(4, c)
    allocate(2, sa)
    allocate(2, sb)
    allocate(2, sc)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(sa)
    hadamard_transform(sb)
    hadamard_transform(sc)
    for i in range(4):
        cost_layer(params[i], a, b, c, sa, sb, sc)
        for j in range(4):
            RX(params[4 + i], a[j])
            RX(params[4 + i], b[j])
            RX(params[4 + i], c[j])
        for j in range(2):
            RX(params[4 + i], sa[j])
            RX(params[4 + i], sb[j])
            RX(params[4 + i], sc[j])


DEFAULT_PARAMS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 8:
        raise ValueError("Four electric-grid layers require eight real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())
