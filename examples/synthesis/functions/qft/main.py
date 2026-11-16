"""Synthesize and execute the native four-qubit Fourier transform."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qft, qfunc


@qfunc
def main(x: Output[QArray[QBit]]) -> None:
    allocate(4, x)
    qft(x)


def build_program() -> synth.Program:
    """Run the complete source Notebook's allocate/transform/synthesize workflow."""
    return synth.synthesize(main)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
