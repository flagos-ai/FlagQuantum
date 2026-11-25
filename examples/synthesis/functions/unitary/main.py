"""Apply a retained static complex matrix through native quantum functions."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, unitary

MATRIX = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1j, 0), (0, 0, 0, 1j))


@qfunc
def main(x: Output[QArray[QBit]]) -> None:
    allocate(2, x)
    unitary(MATRIX, x)


def build_program() -> synth.Program:
    return synth.synthesize(main)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=tuple(program.registers))))
