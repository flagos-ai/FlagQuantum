"""Evolve a static native Observable with an ordered Pauli product formula."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, suzuki_trotter

HAMILTONIAN = 0.3 * (fq.Z(0) @ fq.Z(1)) + 0.7 * fq.X(0) + 0.2 * fq.X(1)


@qfunc
def main(time: float, qba: Output[QArray[QBit]]) -> None:
    allocate(2, qba)
    suzuki_trotter(HAMILTONIAN, time, 1, 10, qba)


def build_program(time: float = 1.0) -> synth.Program:
    return synth.synthesize(main, classical_inputs={"time": time})


def run_example() -> None:
    """Synthesize, execute and read named amplitudes as probabilities."""
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("qba",))))


if __name__ == "__main__":
    run_example()
