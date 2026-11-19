"""Synthesize the generated finite task polynomial using the shared native qfunc."""

import math

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

from .problem import Problem


@qfunc
def schedule(
    bits: int,
    count: int,
    constant: float,
    first: list[int],
    second: list[int],
    weights: list[float],
    parameters: list[float],
    target: Output[QArray[QBit]],
) -> None:
    allocate(bits, target)
    hadamard_transform(target)
    for layer in range(8):
        phase(constant, -parameters[2 * layer])
        for item in range(count):
            phase(
                weights[item] * target[first[item]] * target[second[item]],
                -parameters[2 * layer],
            )
        for bit in range(bits):
            RX(parameters[2 * layer + 1], target[bit])


def build(problem: Problem, parameters: tuple[float, ...]) -> synth.Program:
    terms = [(support, weight) for support, weight in problem.polynomial if support]
    return synth.synthesize(
        schedule,
        classical_inputs={
            "bits": problem.width,
            "count": len(terms),
            "constant": dict(problem.polynomial).get((), 0.0),
            "first": [support[0] for support, _ in terms],
            "second": [support[-1] for support, _ in terms],
            "weights": [weight for _, weight in terms],
            "parameters": list(parameters),
        },
        max_qubits=problem.width,
        max_workspace=0,
        memory_limit_bytes=2**30,
    )


def initial_parameters(problem: Problem) -> tuple[float, ...]:
    ising: dict[tuple[int, ...], float] = {}
    for support, coefficient in problem.polynomial:
        for subset in range(1 << len(support)):
            key = tuple(bit for i, bit in enumerate(support) if subset >> i & 1)
            ising[key] = ising.get(key, 0) + coefficient * ((-1) ** len(key)) / (
                2 ** len(support)
            )
    magnitudes = sorted(
        abs(value) for key, value in ising.items() if key and abs(value) > 1e-12
    )
    largest = magnitudes[len(magnitudes) // 2 :]
    step = len(largest) / (2 * math.fsum(largest))
    return tuple(
        value
        for layer in range(8)
        for value in (step * layer / 7, step * (1 - layer / 7))
    )
