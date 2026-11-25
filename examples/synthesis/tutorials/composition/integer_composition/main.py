"""A repeated phase cost in Qmod, native qfunc and an independent Circuit.

The Qmod entry binds an exact integer. The equivalent native qfunc uses two
static helper calls; its existing Python parameter contract is unchanged."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def twist(a: Const[QBit], b: Const[QBit]) -> None:
    phase(a * b, 0.6)


@qfunc
def main(a: Output[QBit], b: Output[QBit]) -> None:
    allocate(a)
    allocate(b)
    hadamard_transform(a)
    hadamard_transform(b)
    twist(a, b)
    twist(a, b)


def run_example() -> None:
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
