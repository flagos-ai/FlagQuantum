"""A Boolean product phase compiled from native high-level arithmetic."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def main(a: Output[QBit], b: Output[QBit], c: Output[QBit], d: Output[QBit]) -> None:
    allocate(a)
    allocate(b)
    allocate(c)
    allocate(d)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    hadamard_transform(d)
    phase(a * b * c * d, 0.37)


def run_example() -> None:
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
