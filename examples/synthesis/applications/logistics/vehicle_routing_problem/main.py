"""Sixteen-bit twelve-layer native vehicle-routing program."""

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
    gamma: float,
    departure: list[float],
    arrival: list[float],
    distances: list[float],
    x: QArray[QBit],
) -> None:
    for city in range(4):
        phase(
            departure[city] * (x[4 * city] + x[4 * city + 2])
            + arrival[city] * (x[4 * city + 1] + x[4 * city + 3]),
            0.15 * gamma,
        )
        for other in range(4):
            phase(
                distances[4 * city + other]
                * (x[4 * city] * x[4 * other + 2] + x[4 * city + 1] * x[4 * other + 3]),
                0.15 * gamma,
            )
        phase(
            (x[4 * city] + x[4 * city + 1] + x[4 * city + 2] + x[4 * city + 3] - 1)
            ** 2,
            0.9 * gamma,
        )
    for slot in range(4):
        phase(
            (x[slot] + x[4 + slot] + x[8 + slot] + x[12 + slot] - 1) ** 2, 0.9 * gamma
        )


@qfunc
def main(
    params: list[float],
    departure: list[float],
    arrival: list[float],
    distances: list[float],
    x: Output[QArray[QBit]],
) -> None:
    allocate(16, x)
    hadamard_transform(x)
    for layer in range(12):
        cost(params[2 * layer], departure, arrival, distances, x)
        for bit in range(16):
            RX(0.1 * params[2 * layer + 1], x[bit])


DEPARTURE = (0.071298597812624, 0.195867966554767, 0.126824411268937, 0.196014621910967)
ARRIVAL = (0.113254916329654, 0.091194256233777, 0.168990876985014, 0.044163937342742)
DISTANCES = (
    0.0,
    0.128548621806211,
    0.074618713029614,
    0.157418483504471,
    0.128548621806211,
    0.0,
    0.140809501156594,
    0.109680667602677,
    0.074618713029614,
    0.140809501156594,
    0.0,
    0.210830604009408,
    0.157418483504471,
    0.109680667602677,
    0.210830604009408,
    0.0,
)


DEFAULT_PARAMS = tuple(0.01 * (i + 1) for i in range(24))


def build_program(
    params: tuple[float, ...] = DEFAULT_PARAMS,
) -> synth.Program:
    if len(params) != 24:
        raise ValueError("Twelve vehicle-routing layers require 24 real parameters")
    return synth.synthesize(
        main,
        classical_inputs={
            "params": params,
            "departure": DEPARTURE,
            "arrival": ARRIVAL,
            "distances": DISTANCES,
        },
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())
