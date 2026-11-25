"""Compute a coherent minimum without designing a comparison or selection circuit."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc


@qfunc
def main(a: Output[QNum], b: Output[Annotated[QNum, 3]], res: Output[QNum]) -> None:
    a |= 4
    allocate(b)
    hadamard_transform(b)
    res |= a if a < b else b


def build_program() -> synth.Program:
    return synth.synthesize(main)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("a", "b", "res"))))
