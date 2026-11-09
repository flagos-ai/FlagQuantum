"""One-bit phase estimation with Qmod, existing scalar qfunc, and Circuit.

The native example composes existing operations. It does not advertise a native
array QPE export; the three unchanged full QPE originals have a separate validator."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    prepare_uniform_interval_state,
    qfunc,
)

ANGLE = math.pi / 2


@qfunc
def estimate(counter: Output[QBit], data: Output[QBit]) -> None:
    allocate(counter)
    allocate(data)
    prepare_uniform_interval_state(1, 2, data)
    hadamard_transform(counter)
    phase(counter * data, ANGLE)
    hadamard_transform(counter)


def run_example() -> None:
    program = synth.synthesize(estimate)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
