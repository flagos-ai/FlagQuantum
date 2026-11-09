"""Distinguish the two balanced predicates in the upstream tutorial."""

import math
from collections import Counter
from random import Random

import flagquantum as fq
from flagquantum.drawer import draw
from flagquantum.synth import (
    PHASE,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
    synthesize,
)


@qfunc
def high_bit_phase(angle: float, bit: QBit) -> None:
    PHASE(angle, bit)


@qfunc
def simple(x: Output[QArray[QBit]]) -> None:
    allocate(4, x)
    hadamard_transform(x)
    # On four bits, x > 7 is the most significant bit.
    high_bit_phase(math.pi, x[3])
    hadamard_transform(x)


@qfunc
def main(x: Output[QArray[QBit]]) -> None:
    allocate(3, x)
    hadamard_transform(x)
    # For x in 0..7, max(x ^ 4, x & 3) >= 4 iff the high bit is zero.
    phase(1, math.pi)
    high_bit_phase(-math.pi, x[2])
    hadamard_transform(x)


def classify(probabilities: dict[tuple[int, ...], float]) -> str:
    """Apply the promised constant-or-balanced decision to the zero outcome."""
    zero = probabilities.get((0,), 0.0)
    if abs(zero - 1) < 1e-12:
        return "constant"
    if abs(zero) < 1e-12:
        return "balanced"
    raise ValueError("distribution does not certify a constant-or-balanced promise")


def run_example() -> None:
    for label, definition in (("simple", simple), ("complex", main)):
        program = synthesize(definition, max_qubits=11)
        print(draw(program.circuit))
        result = fq.run(program)
        probabilities = dict(program.probabilities(result, registers=("x",)))
        print(label, classify(probabilities))
        samples = Random(17).choices(
            list(probabilities), weights=list(probabilities.values()), k=1000
        )
        print("Counts:", dict(Counter(samples)))


if __name__ == "__main__":
    run_example()
