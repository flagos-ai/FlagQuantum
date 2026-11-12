"""Run a Hadamard test with a coherently controlled four-bit Fourier transform."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import H, Output, QArray, QBit, allocate, control, qft, qfunc


@qfunc
def main(expectation_value: Output[QBit], psi: Output[QArray[QBit]]) -> None:
    allocate(expectation_value)
    allocate(4, psi)
    H(expectation_value)
    control(expectation_value, lambda: qft(psi))
    H(expectation_value)


def build_program() -> synth.Program:
    """Retain the complete joint state; select the ancilla at readout time."""
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("expectation_value",))))
    print(program.report["resources"])
