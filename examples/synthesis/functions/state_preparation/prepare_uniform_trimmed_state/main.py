"""Prepare a uniform state using native Python quantum semantics."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    prepare_uniform_interval_state,
    qfunc,
)


@qfunc
def main(x: Output[QNum]) -> None:
    allocate(4, x)
    prepare_uniform_interval_state(0, 3, x)


if __name__ == "__main__":
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
