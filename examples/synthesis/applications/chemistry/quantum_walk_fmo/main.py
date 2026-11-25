"""Evolve the FMO Hamiltonian using native high-level quantum functions."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, qfunc, suzuki_trotter

from .physics import HAMILTONIAN


@qfunc
def main(qn: Output[QNum], time_point: float) -> None:
    allocate(3, qn)
    suzuki_trotter(HAMILTONIAN, time_point, 1, 10, qn)


def build_program(time_point: float = 0.037) -> synth.Program:
    return synth.synthesize(main, classical_inputs={"time_point": time_point})


def run_example() -> None:
    """Execute one time point and read named site populations."""
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("qn",))))


if __name__ == "__main__":
    run_example()
