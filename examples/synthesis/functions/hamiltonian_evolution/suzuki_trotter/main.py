"""Evolve a static native Observable with an ordered Pauli product formula."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, suzuki_trotter

HAMILTONIAN = 1.5 * (fq.X(0) @ fq.X(1) @ fq.Z(2)) + 0.5 * (fq.Y(0) @ fq.X(1) @ fq.Z(2))


@qfunc
def main(time: float, qba: Output[QArray[QBit]]) -> None:
    allocate(3, qba)
    suzuki_trotter(HAMILTONIAN, time, 1, 1, qba)


def build_program(time: float = 0.37) -> synth.Program:
    return synth.synthesize(main, classical_inputs={"time": time})


def run_example() -> None:
    """Synthesize, execute and read named amplitudes as probabilities."""
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("qba",))))


if __name__ == "__main__":
    run_example()
