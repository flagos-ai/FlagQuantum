"""Recover a hidden five-bit integer from its linear phase oracle."""

import math
from collections import Counter
from random import Random

import flagquantum as fq
from flagquantum.drawer import draw
from flagquantum.synth import (
    PHASE,
    Output,
    Program,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    qfunc,
    synthesize,
)


@qfunc
def linear_phase(weight: float, bit: QBit) -> None:
    PHASE(math.pi * weight, bit)


@qfunc
def hidden_string(weights: list[float], x: QArray[QBit]) -> None:
    for index in range(len(weights)):
        linear_phase(weights[index], x[index])


@qfunc
def main(weights: list[float], x: Output[QArray[QBit]]) -> None:
    allocate(5, x)
    hadamard_transform(x)
    hidden_string(weights, x)
    hadamard_transform(x)


def build_program(secret: int = 13) -> Program:
    """Bind the classical secret; the oracle applies exp(i*pi*a dot x)."""
    if type(secret) is not int or not 0 <= secret < 32:
        raise ValueError("secret must be a five-bit integer in 0..31")
    weights = [float((secret >> index) & 1) for index in range(5)]
    return synthesize(main, classical_inputs={"weights": weights}, max_qubits=6)


def run_example(secret: int = 13) -> None:
    program = build_program(secret)
    print(draw(program.circuit))
    result = fq.run(program)
    probabilities = dict(program.probabilities(result, registers=("x",)))
    # Local seeded sampling replaces the upstream cloud sampling session.
    samples = Random(17).choices(
        list(probabilities), weights=list(probabilities.values()), k=1000
    )
    counts = Counter(samples)
    recovered = counts.most_common(1)[0][0][0]
    assert recovered == secret
    print("Secret integer:", recovered)
    print("Exact recovery probability:", probabilities[(recovered,)])
    print("Counts:", dict(counts))


if __name__ == "__main__":
    run_example()
