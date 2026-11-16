"""Native traffic costs retain all three commodities and cubic congestion."""

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
def flow_cost(
    gamma: float,
    base: int,
    source0: int,
    source1: int,
    source2: int,
    sink: int,
    z: QArray[QBit],
) -> None:
    phase(
        (source0 - z[base] - z[base + 1]) ** 2
        + (z[base] - z[base + 2] - z[base + 3] + source1) ** 2
        + (z[base + 1] - z[base + 4] - z[base + 5] + source2) ** 2
        + (z[base + 3] + z[base + 5] - z[base + 6]) ** 2
        + (z[base + 2] + z[base + 4] - z[base + 7] - (1 - sink)) ** 2
        + (z[base + 6] + z[base + 7] - sink) ** 2,
        0.2 * gamma,
    )


@qfunc
def cost_layer(gamma: float, z: QArray[QBit]) -> None:
    for group in range(3):
        phase(
            0.133333333333333
            * (z[8 * group] + z[8 * group + 2] + z[8 * group + 5] + z[8 * group + 6])
            + 0.2 * z[8 * group + 1]
            + 0.066666666666667
            * (z[8 * group + 3] + z[8 * group + 4] + z[8 * group + 7]),
            gamma,
        )
    flow_cost(gamma, 0, 1, 0, 0, 0, z)
    flow_cost(gamma, 8, 0, 1, 0, 1, z)
    flow_cost(gamma, 16, 0, 0, 1, 1, z)
    for edge in range(8):
        phase(
            z[edge] * z[edge + 8]
            + z[edge] * z[edge + 16]
            + z[edge + 8] * z[edge + 16]
            - 2 * z[edge] * z[edge + 8] * z[edge + 16],
            0.2 * gamma,
        )


@qfunc
def main(params: list[float], z: Output[QArray[QBit]]) -> None:
    allocate(24, z)
    hadamard_transform(z)
    for layer in range(5):
        cost_layer(params[2 * layer], z)
        for bit in range(24):
            RX(params[2 * layer + 1], z[bit])


DEFAULT_PARAMS = tuple(0.01 * (i + 1) for i in range(10))


def build_program(
    params: tuple[float, ...] = DEFAULT_PARAMS,
) -> synth.Program:
    if len(params) != 10:
        raise ValueError("Five traffic layers require ten real parameters")
    return synth.synthesize(
        main,
        classical_inputs={"params": params},
        max_workspace=0,
        memory_limit_bytes=2**30,
    )


if __name__ == "__main__":
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())
