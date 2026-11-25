"""Complete twenty-layer native network resiliency cost program."""

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
def cost_layer(gamma: float, z: QArray[QBit]) -> None:
    phase(
        0.025
        * (
            z[0]
            + z[1]
            + z[2]
            + z[3]
            + z[4]
            + z[5]
            + z[12]
            + z[13]
            + z[14]
            + z[15]
            + z[16]
            + z[17]
        )
        + 0.05 * (z[6] + z[18])
        + 0.25 * (z[7] * z[19] + z[8] * z[20] + z[9] * z[21] + z[11] * z[23])
        + z[1] * z[14]
        + z[2] * z[13]
        + 0.125
        * (
            (1 - 2 * z[7] + z[0] + z[1]) ** 2
            + (-2 * z[19] + z[12] + z[13]) ** 2
            + (-2 * z[8] + z[3] + z[5]) ** 2
            + (1 - 2 * z[20] + z[15] + z[17]) ** 2
            + (-2 * z[9] + z[0] + z[2] + z[3] + z[4]) ** 2
            + (-2 * z[21] + z[12] + z[14] + z[15] + z[16]) ** 2
            + (1 - 2 * z[10] + z[1] + z[2] + z[6]) ** 2
            + (1 - 2 * z[22] + z[13] + z[14] + z[18]) ** 2
            + (-2 * z[11] + z[4] + z[5] + z[6]) ** 2
            + (-2 * z[23] + z[16] + z[17] + z[18]) ** 2
            + 4
            - z[10]
            - z[7]
            - z[22]
            - z[20]
        ),
        gamma,
    )


@qfunc
def main(params: list[float], z: Output[QArray[QBit]]) -> None:
    allocate(24, z)
    hadamard_transform(z)
    for layer in range(20):
        cost_layer(params[2 * layer], z)
        for bit in range(24):
            RX(params[2 * layer + 1], z[bit])


DEFAULT_PARAMS = tuple(0.01 * (i + 1) for i in range(40))


def build_program(
    params: tuple[float, ...] = DEFAULT_PARAMS,
) -> synth.Program:
    if len(params) != 40:
        raise ValueError("Twenty resiliency layers require 40 real parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": params},
        max_workspace=0,
        memory_limit_bytes=2**30,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(program.report["resources"])
    print("State norm:", result.statevector().norm().item())
